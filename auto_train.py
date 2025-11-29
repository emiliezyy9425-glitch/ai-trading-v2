#!/usr/bin/env python3
"""Watch historical_data.csv and retrain models when it updates."""
import time
import subprocess
import sys
from pathlib import Path

try:  # pragma: no cover - optional dependency
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer
except Exception:  # pragma: no cover
    FileSystemEventHandler = object  # type: ignore
    Observer = None  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_FILE = PROJECT_ROOT / "data" / "historical_data.csv"
CONFIGS: list[str] = [
    "rf_config.json",
    "xgb_config.json",
    "lgb_config.json",
    "ppo_config.json",
    "transformer_config.json",
    "lstm_config.json",
]


def retrain_all() -> None:
    """Run retraining for all available configs."""
    script = PROJECT_ROOT / "retrain_models.py"
    for cfg in CONFIGS:
        cfg_path = PROJECT_ROOT / cfg
        if cfg_path.exists():
            subprocess.run([sys.executable, str(script), "--config", str(cfg_path)], check=True)


class _DataHandler(FileSystemEventHandler):
    """Trigger retraining when the historical data file changes."""
    def __init__(self) -> None:
        super().__init__()
        self._last_mtime = 0.0

    def _maybe_retrain(self) -> None:
        if DATA_FILE.exists():
            mtime = DATA_FILE.stat().st_mtime
            if mtime != self._last_mtime:
                self._last_mtime = mtime
                retrain_all()

    def on_modified(self, event):
        if Path(event.src_path) == DATA_FILE:
            self._maybe_retrain()

    def on_created(self, event):
        if Path(event.src_path) == DATA_FILE:
            self._maybe_retrain()


def main() -> None:
    if Observer is None:
        raise SystemExit("watchdog is required to use auto_train.py")
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    observer = Observer()
    handler = _DataHandler()
    observer.schedule(handler, DATA_FILE.parent, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        observer.stop()
        observer.join()


if __name__ == "__main__":
    main()
