from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
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
    parser = argparse.ArgumentParser(description="Benchmark attention backend vs artifact backend for Baseline mode.")
    parser.add_argument(
        "--fixtures",
        nargs="*",
        default=[str(p) for p in DEFAULT_FIXTURES],
        help="CSV fixtures to benchmark.",
    )
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--bench-runs", type=int, default=5)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports/benchmark"))
    return parser.parse_args()


def _time_ms(fn, warmups: int, runs: int) -> tuple[float, float]:
    for _ in range(max(0, int(warmups))):
        fn()
    durations = []
    for _ in range(max(1, int(runs))):
        t0 = time.perf_counter()
        fn()
        durations.append((time.perf_counter() - t0) * 1000.0)
    return float(statistics.mean(durations)), float(statistics.stdev(durations) if len(durations) > 1 else 0.0)


def main() -> None:
    args = parse_args()
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    service = ModelService()

    rows: List[Dict[str, object]] = []

    for fixture_str in args.fixtures:
        fixture = Path(fixture_str)
        payload = fixture.read_bytes()
        fixture_key = str(fixture).replace("\\", "/")

        attention_result = service.infer(payload, model_mode="Baseline", model_backend="attention")
        try:
            artifact_result = service.infer(payload, model_mode="Baseline", model_backend="artifact")
        except Exception as exc:
            rows.append({"fixture": fixture_key, "error": str(exc)})
            continue

        att_ms_mean, att_ms_std = _time_ms(
            lambda: service.infer(payload, model_mode="Baseline", model_backend="attention"),
            args.warmup_runs,
            args.bench_runs,
        )
        art_ms_mean, art_ms_std = _time_ms(
            lambda: service.infer(payload, model_mode="Baseline", model_backend="artifact"),
            args.warmup_runs,
            args.bench_runs,
        )

        shared = sorted(set(attention_result["engine_ids"]) & set(artifact_result["engine_ids"]))
        decision_match = 0
        for engine_id in shared:
            a_dec = attention_result["per_engine_reliability"][engine_id]["decision"]
            b_dec = artifact_result["per_engine_reliability"][engine_id]["decision"]
            if a_dec == b_dec:
                decision_match += 1
        match_rate = float(decision_match / len(shared)) if shared else float("nan")

        rows.append(
            {
                "fixture": fixture_key,
                "engines_shared": len(shared),
                "attention_overall_mean_rul": float(attention_result["overall_mean_rul"]),
                "artifact_overall_mean_rul": float(artifact_result["overall_mean_rul"]),
                "delta_overall_mean_rul": float(
                    artifact_result["overall_mean_rul"] - attention_result["overall_mean_rul"]
                ),
                "attention_overall_reliability_index": float(attention_result["overall_reliability_index"]),
                "artifact_overall_reliability_index": float(artifact_result["overall_reliability_index"]),
                "delta_overall_reliability_index": float(
                    artifact_result["overall_reliability_index"] - attention_result["overall_reliability_index"]
                ),
                "decision_match_rate": match_rate,
                "attention_latency_ms_mean": att_ms_mean,
                "attention_latency_ms_std": att_ms_std,
                "artifact_latency_ms_mean": art_ms_mean,
                "artifact_latency_ms_std": art_ms_std,
                "speedup_attention_over_artifact": float(att_ms_mean / art_ms_mean) if art_ms_mean > 0 else float("nan"),
            }
        )

    out_df = pd.DataFrame(rows)
    csv_path = args.reports_dir / "backend_benchmark_summary.csv"
    json_path = args.reports_dir / "backend_benchmark_summary.json"
    out_df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print("Backend benchmark completed.")
    print(f"- summary_csv: {csv_path}")
    print(f"- summary_json: {json_path}")


if __name__ == "__main__":
    main()
