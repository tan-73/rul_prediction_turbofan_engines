from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from inference.reliability import evaluate_reliability_log

def evaluate(csv_path: Path, catastrophic_error_threshold: float) -> dict:
    df = pd.read_csv(csv_path)
    return evaluate_reliability_log(df, catastrophic_error_threshold=catastrophic_error_threshold)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate reliability gating behavior from prediction logs.")
    parser.add_argument("--csv", type=Path, required=True, help="CSV with true_rul,predicted_rul,ri and optional decision.")
    parser.add_argument(
        "--cat-threshold",
        type=float,
        default=20.0,
        help="Absolute error threshold for catastrophic prediction failures.",
    )
    args = parser.parse_args()

    metrics = evaluate(args.csv, args.cat_threshold)
    print("Reliability Evaluation Summary")
    for key, value in metrics.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
