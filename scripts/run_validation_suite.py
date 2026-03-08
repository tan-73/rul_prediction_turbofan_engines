from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.model_service import ModelService


DEFAULT_FIXTURES = [
    Path("examples/sample_cmapss_engine.csv"),
    Path("examples/sample_cmapss_engine_dual.csv"),
    Path("examples/scenarios/scenario_stable_behavior.csv"),
    Path("examples/scenarios/scenario_noisy_behavior.csv"),
    Path("examples/scenarios/scenario_rapid_degradation.csv"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic validation suite for model outputs.")
    parser.add_argument("--reports-dir", type=Path, default=Path("reports/validation"))
    parser.add_argument(
        "--fixtures",
        nargs="*",
        default=[str(p) for p in DEFAULT_FIXTURES],
        help="CSV fixtures for regression sanity checks.",
    )
    parser.add_argument(
        "--include-artifact-backend",
        action="store_true",
        help="Also compare Baseline attention backend vs artifact backend (if artifact is available).",
    )
    return parser.parse_args()


def _decision_counts(result: Dict[str, object]) -> Dict[str, int]:
    counts: Dict[str, int] = {"ACCEPT": 0, "WARN": 0, "REJECT": 0}
    for engine_id in result["engine_ids"]:
        decision = str(result["per_engine_reliability"][engine_id]["decision"])
        if decision in counts:
            counts[decision] += 1
    return counts


def main() -> None:
    args = parse_args()
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    service = ModelService()

    rows: List[Dict[str, object]] = []
    golden: Dict[str, Dict[str, object]] = {}
    backend_rows: List[Dict[str, object]] = []

    for fixture_str in args.fixtures:
        fixture = Path(fixture_str)
        payload = fixture.read_bytes()
        fixture_key = str(fixture).replace("\\", "/")
        golden[fixture_key] = {}

        for mode in ["Baseline", "Physics-Informed"]:
            result = service.infer(payload, model_mode=mode)
            decision_counts = _decision_counts(result)

            row = {
                "fixture": fixture_key,
                "mode": mode,
                "engines": int(len(result["engine_ids"])),
                "overall_mean_rul": float(result["overall_mean_rul"]),
                "overall_reliability_index": float(result["overall_reliability_index"]),
                "accept_count": decision_counts["ACCEPT"],
                "warn_count": decision_counts["WARN"],
                "reject_count": decision_counts["REJECT"],
            }
            rows.append(row)
            golden[fixture_key][mode] = row

        if args.include_artifact_backend:
            try:
                attention = service.infer(payload, model_mode="Baseline", model_backend="attention")
                artifact = service.infer(payload, model_mode="Baseline", model_backend="artifact")
                shared = sorted(set(attention["engine_ids"]) & set(artifact["engine_ids"]))
                backend_rows.append(
                    {
                        "fixture": fixture_key,
                        "engines_shared": len(shared),
                        "attention_overall_mean_rul": float(attention["overall_mean_rul"]),
                        "artifact_overall_mean_rul": float(artifact["overall_mean_rul"]),
                        "delta_overall_mean_rul": float(artifact["overall_mean_rul"] - attention["overall_mean_rul"]),
                        "attention_overall_reliability_index": float(attention["overall_reliability_index"]),
                        "artifact_overall_reliability_index": float(artifact["overall_reliability_index"]),
                        "delta_overall_reliability_index": float(
                            artifact["overall_reliability_index"] - attention["overall_reliability_index"]
                        ),
                    }
                )
            except Exception as exc:
                backend_rows.append({"fixture": fixture_key, "error": str(exc)})

    df = pd.DataFrame(rows)
    csv_path = args.reports_dir / "validation_summary.csv"
    json_path = args.reports_dir / "validation_summary.json"
    golden_path = ROOT_DIR / "reproducibility" / "golden_inference_outputs.json"

    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    golden_path.parent.mkdir(parents=True, exist_ok=True)
    golden_path.write_text(json.dumps(golden, indent=2), encoding="utf-8")

    agg = {
        "fixtures": int(df["fixture"].nunique()),
        "rows": int(len(df)),
        "baseline_mean_rul": float(df[df["mode"] == "Baseline"]["overall_mean_rul"].mean()),
        "pi_mean_rul": float(df[df["mode"] == "Physics-Informed"]["overall_mean_rul"].mean()),
    }
    agg_path = args.reports_dir / "validation_aggregate.json"
    agg_path.write_text(json.dumps(agg, indent=2), encoding="utf-8")

    if backend_rows:
        backend_df = pd.DataFrame(backend_rows)
        backend_csv_path = args.reports_dir / "validation_backend_comparison.csv"
        backend_json_path = args.reports_dir / "validation_backend_comparison.json"
        backend_df.to_csv(backend_csv_path, index=False)
        backend_json_path.write_text(json.dumps(backend_rows, indent=2), encoding="utf-8")
        print(f"- backend_comparison_csv: {backend_csv_path}")
        print(f"- backend_comparison_json: {backend_json_path}")

    print("Validation suite complete.")
    print(f"- summary_csv: {csv_path}")
    print(f"- summary_json: {json_path}")
    print(f"- aggregate_json: {agg_path}")
    print(f"- golden_regression: {golden_path}")


if __name__ == "__main__":
    main()
