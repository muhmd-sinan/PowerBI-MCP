import logging
from plyer import notification

logger = logging.getLogger(__name__)


def notify_signal(signal_data: dict):
    direction = signal_data["direction"]
    symbol = signal_data["symbol"]
    icon = "\U0001f7e2" if direction == "LONG" else "\U0001f534"
    title = f"{icon} {direction} {symbol}"

    entry = signal_data["entry_price"]
    tp1 = signal_data["tp1"]
    tp2 = signal_data["tp2"]
    tp3 = signal_data["tp3"]
    sl = signal_data["sl"]
    conf = signal_data["confidence"]

    body = (
        f"Entry: ${entry:.2f}\n"
        f"TP1: ${tp1:.2f} | TP2: ${tp2:.2f} | TP3: ${tp3:.2f}\n"
        f"SL: ${sl:.2f} | Conf: {conf:.0%}"
    )

    try:
        notification.notify(
            title=title,
            message=body,
            app_name="Futures Bot",
            timeout=15,
        )
        logger.info(f"Desktop notification sent: {title}")
    except Exception as e:
        logger.error(f"Failed to send notification: {e}")


def notify_reconnection(offline_minutes: float):
    try:
        notification.notify(
            title="Futures Bot — Reconnected",
            message=f"Back online after {offline_minutes:.1f} min. Gaps backfilled.",
            app_name="Futures Bot",
            timeout=10,
        )
    except Exception as e:
        logger.error(f"Reconnection notification failed: {e}")


def notify_degradation(symbol: str, win_rate: float):
    try:
        notification.notify(
            title=f"⚠️ {symbol} Model Degraded",
            message=f"Live win rate dropped to {win_rate:.0%} (last {30} signals). Consider retraining.",
            app_name="Futures Bot",
            timeout=15,
        )
    except Exception as e:
        logger.error(f"Degradation notification failed: {e}")
