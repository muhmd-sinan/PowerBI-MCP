import torch
import torch.nn as nn


class TimeframeBranch(nn.Module):
    def __init__(self, num_features: int, hidden_size: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(num_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
        )
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3),  # long, short, neutral
        )

    def forward(self, x):
        # x shape: [batch, sequence_length, num_features]
        x = x.permute(0, 2, 1)  # [batch, features, seq_len] for Conv1d
        x = self.conv(x)
        x = x.permute(0, 2, 1)  # [batch, reduced_seq, channels] for LSTM
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]  # last hidden state
        logits = self.head(x)
        return logits


class FuturesModel(nn.Module):
    def __init__(self, num_features: int, num_timeframes: int = 6, hidden_size: int = 128):
        super().__init__()
        self.num_timeframes = num_timeframes
        self.branches = nn.ModuleList([
            TimeframeBranch(num_features, hidden_size) for _ in range(num_timeframes)
        ])
        self.primary_indices = [0, 1, 2]  # 5m, 10m, 15m
        self.confluence_indices = [3, 4, 5]  # 30m, 1h, 2h

    def forward(self, x):
        # x shape: [batch, num_timeframes, sequence_length, num_features]
        tf_logits = []
        for i, branch in enumerate(self.branches):
            tf_input = x[:, i, :, :]  # [batch, seq_len, features]
            logits = branch(tf_input)
            tf_logits.append(logits)

        tf_probs = [torch.softmax(logits, dim=-1) for logits in tf_logits]
        primary_probs = [tf_probs[i] for i in self.primary_indices]
        confluence_probs = [tf_probs[i] for i in self.confluence_indices]

        # Primary vote: direction with highest avg probability across primary TFs
        primary_stack = torch.stack(primary_probs, dim=1)  # [batch, 3, 3]
        primary_avg = primary_stack.mean(dim=1)  # [batch, 3] (long, short, neutral)

        # Primary agreement: how many agree on the top direction
        primary_directions = torch.stack([p.argmax(dim=-1) for p in primary_probs], dim=1)
        top_direction = primary_avg.argmax(dim=-1)  # [batch]
        agreement = (primary_directions == top_direction.unsqueeze(1)).float().sum(dim=1)

        # Confidence from primary average probability of the chosen direction
        base_confidence = primary_avg.gather(1, top_direction.unsqueeze(1)).squeeze(1)

        # Confluence multiplier
        confluence_stack = torch.stack(confluence_probs, dim=1)
        confluence_avg = confluence_stack.mean(dim=1)
        confluence_direction = confluence_avg.argmax(dim=-1)

        # 1.2 if confluence agrees, 0.7 if disagrees, 1.0 if mixed/neutral
        multiplier = torch.where(
            confluence_direction == top_direction,
            torch.tensor(1.2, device=x.device),
            torch.where(
                confluence_direction == 2,  # neutral
                torch.tensor(1.0, device=x.device),
                torch.tensor(0.7, device=x.device),
            ),
        )

        final_confidence = (base_confidence * multiplier).clamp(0.0, 0.95)

        return {
            "direction": top_direction,  # 0=long, 1=short, 2=neutral
            "confidence": final_confidence,
            "agreement": agreement,  # how many primary TFs agree (2 or 3)
            "tf_probs": tf_probs,
            "primary_logits": [tf_logits[i] for i in self.primary_indices],
            "confluence_logits": [tf_logits[i] for i in self.confluence_indices],
        }

    def get_training_logits(self, x):
        # For training: return all per-timeframe logits for cross-entropy loss
        all_logits = []
        for i, branch in enumerate(self.branches):
            tf_input = x[:, i, :, :]
            logits = branch(tf_input)
            all_logits.append(logits)
        return all_logits
