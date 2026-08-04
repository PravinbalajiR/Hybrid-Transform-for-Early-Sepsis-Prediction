"""
logger.py — simple experiment logger
"""
import json
import time
from pathlib import Path
from datetime import datetime


class ExperimentLogger:
    """Logs metrics to a JSON lines file for easy post-hoc analysis."""

    def __init__(self, log_dir: str | Path, experiment_name: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.name = experiment_name
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{experiment_name}_{ts}.jsonl"
        self._start = time.time()
        print(f"[logger] Writing to {self.log_file}")

    def log(self, step: int, metrics: dict, phase: str = "train") -> None:
        record = {
            "step":    step,
            "phase":   phase,
            "elapsed": round(time.time() - self._start, 2),
            **metrics,
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    def summary(self, metrics: dict) -> None:
        record = {"type": "summary", **metrics}
        with open(self.log_file, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"[logger] Summary: {metrics}")
