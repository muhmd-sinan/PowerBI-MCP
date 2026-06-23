import logging
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.config import config
from src.data.database import SessionLocal, Candle
from src.models.cnn_lstm import FuturesModel
from src.models.labeler import generate_labels, compute_label_stats
from src.features.indicators import calculate_indicators

logger = logging.getLogger(__name__)


class CandleDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class Trainer:
    def __init__(self, symbol: str, device: str = None):
        self.symbol = symbol
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    def load_candles(self, timeframe: str) -> pd.DataFrame:
        db = SessionLocal()
        try:
            candles = (
                db.query(Candle)
                .filter(Candle.symbol == self.symbol, Candle.timeframe == timeframe)
                .order_by(Candle.timestamp.asc())
                .all()
            )
            data = [{
                "timestamp": c.timestamp,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
            } for c in candles]
            return pd.DataFrame(data)
        finally:
            db.close()

    def prepare_data(self, sequence_length: int = 60):
        logger.info(f"Preparing training data for {self.symbol}")
        all_timeframe_data = {}
        min_len = float("inf")

        for tf in config.ALL_TIMEFRAMES:
            df = self.load_candles(tf)
            if df.empty:
                raise ValueError(f"No candle data for {self.symbol} {tf}")
            df = calculate_indicators(df)
            df["funding_rate"] = 0.0
            df["open_interest"] = 0.0
            df = df.dropna().reset_index(drop=True)
            all_timeframe_data[tf] = df
            min_len = min(min_len, len(df))

        # Use primary timeframe (5m) for labels
        primary_df = all_timeframe_data[config.PRIMARY_TIMEFRAMES[0]]
        atr_col = primary_df["atr_14"] if "atr_14" in primary_df.columns else primary_df["ATRr_14"]
        labels = generate_labels(primary_df, atr_col)
        stats = compute_label_stats(labels)
        logger.info(f"Label distribution: {stats}")

        # Normalize features and build sequences
        feature_cols = [c for c in primary_df.columns if c not in ["timestamp"]]
        num_features = len(feature_cols)
        self.num_features = num_features

        # Compute and save scaler params
        scalers = {}
        for tf in config.ALL_TIMEFRAMES:
            df = all_timeframe_data[tf][feature_cols]
            scalers[tf] = {
                "min": df.min().to_dict(),
                "max": df.max().to_dict(),
            }
            # Min-max normalize
            range_vals = df.max() - df.min()
            range_vals[range_vals == 0] = 1
            all_timeframe_data[tf][feature_cols] = (df - df.min()) / range_vals

        self.scalers = scalers

        # Build sequences: [num_samples, 6_timeframes, seq_len, num_features]
        usable_len = min_len - sequence_length - config.FORWARD_WINDOW_CANDLES
        if usable_len <= 0:
            raise ValueError("Not enough data for training with given sequence length")

        X = np.zeros((usable_len, len(config.ALL_TIMEFRAMES), sequence_length, num_features))
        y = labels.values[sequence_length:sequence_length + usable_len]

        for tf_idx, tf in enumerate(config.ALL_TIMEFRAMES):
            df_vals = all_timeframe_data[tf][feature_cols].values
            for i in range(usable_len):
                X[i, tf_idx] = df_vals[i:i + sequence_length]

        # Chronological split: 70/15/15
        train_end = int(usable_len * 0.70)
        val_end = int(usable_len * 0.85)

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]

        logger.info(f"Data split — Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def train(
        self,
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 0.001,
        patience: int = 15,
        sequence_length: int = 60,
    ) -> dict:
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.prepare_data(sequence_length)

        train_ds = CandleDataset(X_train, y_train)
        val_ds = CandleDataset(X_val, y_val)
        test_ds = CandleDataset(X_test, y_test)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)
        test_loader = DataLoader(test_ds, batch_size=batch_size)

        self.model = FuturesModel(
            num_features=self.num_features,
            num_timeframes=len(config.ALL_TIMEFRAMES),
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        # Class weights for imbalanced labels
        class_counts = np.bincount(y_train, minlength=3).astype(float)
        class_counts[class_counts == 0] = 1
        weights = 1.0 / class_counts
        weights = weights / weights.sum() * 3
        class_weights = torch.FloatTensor(weights).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        logger.info(f"Training {self.symbol} model — {epochs} max epochs, patience={patience}")

        for epoch in range(epochs):
            # Train
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                all_logits = self.model.get_training_logits(X_batch)
                loss = sum(criterion(logits, y_batch) for logits in all_logits) / len(all_logits)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validate
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    all_logits = self.model.get_training_logits(X_batch)
                    loss = sum(criterion(logits, y_batch) for logits in all_logits) / len(all_logits)
                    val_loss += loss.item()
                    # Use primary TF average for accuracy
                    primary_logits = all_logits[:3]
                    avg_probs = torch.stack([torch.softmax(l, dim=-1) for l in primary_logits]).mean(0)
                    preds = avg_probs.argmax(dim=-1)
                    val_correct += (preds == y_batch).sum().item()
                    val_total += len(y_batch)

            val_loss /= len(val_loader)
            val_acc = val_correct / val_total
            scheduler.step(val_loss)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_acc)

            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                    f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        # Load best weights
        self.model.load_state_dict(best_state)
        self.model.eval()

        # Test evaluation
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                output = self.model(X_batch)
                preds = output["direction"]
                test_correct += (preds == y_batch).sum().item()
                test_total += len(y_batch)

        test_acc = test_correct / test_total
        logger.info(f"Test accuracy: {test_acc:.4f}")

        return {
            "test_accuracy": test_acc,
            "best_val_loss": best_val_loss,
            "epochs_trained": len(self.history["train_loss"]),
            "history": self.history,
        }

    def save(self) -> str:
        model_dir = config.MODEL_DIR
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / f"{self.symbol}_latest.pt"
        scaler_path = model_dir / f"{self.symbol}_scaler.json"

        torch.save({
            "model_state": self.model.state_dict(),
            "num_features": self.num_features,
            "num_timeframes": len(config.ALL_TIMEFRAMES),
            "symbol": self.symbol,
            "trained_at": datetime.utcnow().isoformat(),
            "history": self.history,
        }, model_path)

        with open(scaler_path, "w") as f:
            json.dump(self.scalers, f)

        logger.info(f"Model saved to {model_path}")
        return str(model_path)

    @staticmethod
    def load_model(symbol: str, device: str = None) -> "FuturesModel":
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_path = config.MODEL_DIR / f"{symbol}_latest.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"No model found for {symbol}")

        checkpoint = torch.load(model_path, map_location=device)
        model = FuturesModel(
            num_features=checkpoint["num_features"],
            num_timeframes=checkpoint["num_timeframes"],
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        return model
