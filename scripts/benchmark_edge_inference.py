from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark TensorFlow vs TFLite edge inference for RUL.")
    parser.add_argument("--csv", type=Path, required=True, help="Input C-MAPSS CSV.")
    parser.add_argument("--mode", default="Baseline", help="Model mode: Baseline or Physics-Informed.")
    parser.add_argument("--tflite", type=Path, required=True, help="Path to exported .tflite file.")
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--bench-runs", type=int, default=15)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports/edge"))
    parser.add_argument("--true-rul-csv", type=Path, help="Optional CSV with columns engine_id,true_rul.")
    return parser.parse_args()


def _inverse_scale_rul(rul_pred_scaled: np.ndarray) -> np.ndarray:
    from inference.attention_model import EARLY_RUL

    scaled = np.asarray(rul_pred_scaled, dtype=float).reshape(-1)
    return np.clip(scaled * float(EARLY_RUL), 0.0, float(EARLY_RUL))


def _split_per_engine(values: np.ndarray, num_windows: List[int]) -> List[np.ndarray]:
    return np.split(values, np.cumsum(num_windows)[:-1])


def _aggregate(values: np.ndarray, engine_ids: List[int], num_windows: List[int]) -> Dict[int, float]:
    per_engine = _split_per_engine(values, num_windows)
    return {int(e): float(np.mean(v)) for e, v in zip(engine_ids, per_engine)}


def _reliability_from_windows(
    engine_ids: List[int], num_windows: List[int], raw_window_rul: np.ndarray, mean_rul: Dict[int, float]
) -> Dict[int, Dict[str, object]]:
    from inference.attention_model import EARLY_RUL
    from inference.reliability import compute_reliability_index, gate_prediction

    per_engine_windows = _split_per_engine(raw_window_rul, num_windows)
    out: Dict[int, Dict[str, object]] = {}
    for engine_id, windows in zip(engine_ids, per_engine_windows):
        metrics = compute_reliability_index(windows.tolist(), early_rul=float(EARLY_RUL))
        gate = gate_prediction(float(mean_rul[int(engine_id)]), float(metrics["ri"]), windows.tolist())
        out[int(engine_id)] = {**metrics, **gate}
    return out


def _infer_tf(model, x_tensor) -> np.ndarray:
    enc_outputs = model.encoder(x_tensor, training=False)
    dec_state = model.decoder.init_state(enc_outputs)
    dec_x = x_tensor[:, -1, tf.newaxis]
    y_pred, _ = model.decoder(dec_x, dec_state, training=False)
    return tf.squeeze(y_pred).numpy()


def _infer_tflite(interpreter, x_np: np.ndarray) -> np.ndarray:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.resize_tensor_input(input_details[0]["index"], x_np.shape)
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]["index"], x_np.astype(np.float32))
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]["index"])
    return np.asarray(out).reshape(-1)


def _time_runs(fn, warmups: int, runs: int) -> Tuple[float, float]:
    for _ in range(max(warmups, 0)):
        fn()
    durations = []
    for _ in range(max(runs, 1)):
        t0 = time.perf_counter()
        fn()
        durations.append((time.perf_counter() - t0) * 1000.0)
    return float(statistics.mean(durations)), float(statistics.stdev(durations) if len(durations) > 1 else 0.0)


def _optional_rss_mb() -> Optional[float]:
    try:
        import psutil  # type: ignore

        process = psutil.Process()
        return float(process.memory_info().rss / (1024.0 * 1024.0))
    except Exception:
        return None


def main() -> None:
    args = parse_args()

    import tensorflow as tf

    from inference.attention_model import get_model_weights_path, load_attention_model, preprocess_csv_for_inference

    args.reports_dir.mkdir(parents=True, exist_ok=True)

    x_tensor, engine_ids, num_windows = preprocess_csv_for_inference(args.csv)
    x_np = x_tensor.numpy()

    model = load_attention_model(get_model_weights_path(args.mode))
    interpreter = tf.lite.Interpreter(model_path=str(args.tflite))

    tf_fn = lambda: _infer_tf(model, x_tensor)
    tfl_fn = lambda: _infer_tflite(interpreter, x_np)

    tf_ms_mean, tf_ms_std = _time_runs(tf_fn, args.warmup_runs, args.bench_runs)
    tfl_ms_mean, tfl_ms_std = _time_runs(tfl_fn, args.warmup_runs, args.bench_runs)

    tf_scaled = np.atleast_1d(_infer_tf(model, x_tensor))
    tfl_scaled = np.atleast_1d(_infer_tflite(interpreter, x_np))

    tf_rul = _inverse_scale_rul(tf_scaled)
    tfl_rul = _inverse_scale_rul(tfl_scaled)

    tf_mean = _aggregate(tf_rul, engine_ids, num_windows)
    tfl_mean = _aggregate(tfl_rul, engine_ids, num_windows)

    tf_rel = _reliability_from_windows(engine_ids, num_windows, tf_rul, tf_mean)
    tfl_rel = _reliability_from_windows(engine_ids, num_windows, tfl_rul, tfl_mean)

    parity_rows = []
    decision_matches = []
    for engine_id in engine_ids:
        tf_dec = str(tf_rel[int(engine_id)]["decision"])
        tfl_dec = str(tfl_rel[int(engine_id)]["decision"])
        decision_matches.append(tf_dec == tfl_dec)
        parity_rows.append(
            {
                "engine_id": int(engine_id),
                "tf_pred_rul": float(tf_mean[int(engine_id)]),
                "tflite_pred_rul": float(tfl_mean[int(engine_id)]),
                "abs_delta_pred_rul": abs(float(tf_mean[int(engine_id)] - tfl_mean[int(engine_id)])),
                "tf_ri": float(tf_rel[int(engine_id)]["ri"]),
                "tflite_ri": float(tfl_rel[int(engine_id)]["ri"]),
                "abs_delta_ri": abs(float(tf_rel[int(engine_id)]["ri"] - tfl_rel[int(engine_id)]["ri"])),
                "tf_decision": tf_dec,
                "tflite_decision": tfl_dec,
                "decision_match": tf_dec == tfl_dec,
            }
        )
    parity_df = pd.DataFrame(parity_rows)

    metrics = {
        "mode": args.mode,
        "csv": str(args.csv),
        "tflite_model": str(args.tflite),
        "samples": int(len(x_np)),
        "engines": int(len(engine_ids)),
        "tf_latency_ms_mean": tf_ms_mean,
        "tf_latency_ms_std": tf_ms_std,
        "tflite_latency_ms_mean": tfl_ms_mean,
        "tflite_latency_ms_std": tfl_ms_std,
        "speedup_x_tf_over_tflite": float(tf_ms_mean / tfl_ms_mean) if tfl_ms_mean > 0 else float("nan"),
        "mean_abs_pred_delta": float(parity_df["abs_delta_pred_rul"].mean()) if not parity_df.empty else float("nan"),
        "mean_abs_ri_delta": float(parity_df["abs_delta_ri"].mean()) if not parity_df.empty else float("nan"),
        "decision_match_rate": float(np.mean(decision_matches)) if decision_matches else float("nan"),
        "rss_memory_mb": _optional_rss_mb(),
    }

    if args.true_rul_csv:
        true_df = pd.read_csv(args.true_rul_csv)
        required_cols = {"engine_id", "true_rul"}
        if not required_cols.issubset(set(true_df.columns)):
            raise ValueError(f"--true-rul-csv must contain columns {sorted(required_cols)}")
        truth = {int(r.engine_id): float(r.true_rul) for r in true_df.itertuples(index=False)}
        shared = [eid for eid in engine_ids if int(eid) in truth]
        if shared:
            tf_true = np.array([truth[int(e)] for e in shared], dtype=float)
            tf_pred = np.array([tf_mean[int(e)] for e in shared], dtype=float)
            tfl_pred = np.array([tfl_mean[int(e)] for e in shared], dtype=float)
            metrics["tf_mae"] = float(np.mean(np.abs(tf_pred - tf_true)))
            metrics["tflite_mae"] = float(np.mean(np.abs(tfl_pred - tf_true)))
            metrics["tf_rmse"] = float(math.sqrt(np.mean((tf_pred - tf_true) ** 2)))
            metrics["tflite_rmse"] = float(math.sqrt(np.mean((tfl_pred - tf_true) ** 2)))

    mode_slug = args.mode.strip().lower().replace("-", "_").replace(" ", "_")
    metrics_path = args.reports_dir / f"edge_benchmark_metrics_{mode_slug}.json"
    parity_path = args.reports_dir / f"edge_benchmark_parity_{mode_slug}.csv"
    summary_path = args.reports_dir / f"edge_benchmark_summary_{mode_slug}.csv"

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    parity_df.to_csv(parity_path, index=False)
    pd.DataFrame([metrics]).to_csv(summary_path, index=False)

    print("Edge benchmark completed.")
    print(f"- metrics_json: {metrics_path}")
    print(f"- parity_csv: {parity_path}")
    print(f"- summary_csv: {summary_path}")


if __name__ == "__main__":
    main()
