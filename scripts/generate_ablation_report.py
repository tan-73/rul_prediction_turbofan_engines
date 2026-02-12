from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from inference.attention_model import (
    get_model_weights_path,
    load_attention_model,
    predict_rul_detailed_from_csv,
)


DEFAULT_REPORTS_DIR = Path("reports")


def _nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = y_pred - y_true
    return float(np.sum(np.where(diff < 0, np.exp(-diff / 13.0) - 1.0, np.exp(diff / 10.0) - 1.0)))


def _decision_flip_rate(decisions: np.ndarray) -> float:
    if len(decisions) <= 1:
        return 0.0
    flips = np.sum(decisions[1:] != decisions[:-1])
    return float(flips / (len(decisions) - 1))


def _load_true_rul(rul_path: Path) -> np.ndarray:
    rul = pd.read_csv(rul_path, sep=r"\s+", header=None).iloc[:, 0].to_numpy(dtype=float)
    if rul.size == 0:
        raise ValueError(f"No RUL values found in: {rul_path}")
    return rul


def _load_freeze_config(path: Path) -> Dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Freeze config must be a JSON object: {path}")
    return payload


def _build_variant_rows(
    result: Dict[str, object],
    true_rul: np.ndarray,
    *,
    variant_name: str,
    model_mode: str,
    use_ri_gating: bool,
    catastrophic_threshold: float,
) -> pd.DataFrame:
    engine_ids: List[int] = sorted(int(v) for v in result["engine_ids"])
    if len(engine_ids) != len(true_rul):
        raise ValueError(
            f"True RUL rows ({len(true_rul)}) do not match predicted engines ({len(engine_ids)})."
        )

    rows: List[Dict[str, object]] = []
    for idx, engine_id in enumerate(engine_ids):
        true_value = float(true_rul[idx])
        raw_pred = float(result["per_engine_mean_rul"][engine_id])
        rel = result["per_engine_reliability"][engine_id]
        ri = float(rel["ri"])
        decision = str(rel["decision"])
        trusted_rul = float(rel["trusted_rul"])
        used_pred = trusted_rul if use_ri_gating else raw_pred
        abs_error = abs(used_pred - true_value)
        rows.append(
            {
                "variant": variant_name,
                "model_mode": model_mode,
                "engine_id": engine_id,
                "true_rul": true_value,
                "predicted_rul_raw": raw_pred,
                "predicted_rul_used": used_pred,
                "ri": ri,
                "decision": decision if use_ri_gating else "RAW",
                "abs_error_used": abs_error,
                "catastrophic_error_used": abs_error > catastrophic_threshold,
                "use_ri_gating": use_ri_gating,
            }
        )
    return pd.DataFrame(rows)


def _compute_metrics(df_variant: pd.DataFrame) -> Dict[str, float | str]:
    y_true = df_variant["true_rul"].to_numpy(dtype=float)
    y_pred = df_variant["predicted_rul_used"].to_numpy(dtype=float)
    ri = df_variant["ri"].to_numpy(dtype=float)
    errors = np.abs(y_pred - y_true)

    rmse = float(math.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae = float(np.mean(errors))
    nasa = _nasa_score(y_true, y_pred)
    corr = float(np.corrcoef(ri, errors)[0, 1]) if len(df_variant) > 1 else float("nan")
    catastrophic_rate = float(np.mean(df_variant["catastrophic_error_used"].to_numpy(dtype=bool)))

    decision_values = df_variant["decision"].to_numpy(dtype=str)
    uses_gating = bool(df_variant["use_ri_gating"].iloc[0])
    if uses_gating:
        acceptance_coverage = float(np.mean(decision_values == "ACCEPT"))
        flip_rate = _decision_flip_rate(decision_values)
    else:
        acceptance_coverage = 1.0
        flip_rate = 0.0

    return {
        "variant": str(df_variant["variant"].iloc[0]),
        "model_mode": str(df_variant["model_mode"].iloc[0]),
        "uses_ri_gating": uses_gating,
        "rmse": rmse,
        "mae": mae,
        "nasa_score": nasa,
        "ri_error_correlation": corr,
        "catastrophic_error_rate": catastrophic_rate,
        "decision_flip_rate": flip_rate,
        "acceptance_coverage": acceptance_coverage,
        "num_engines": int(len(df_variant)),
    }


def _save_plots(metrics_df: pd.DataFrame, per_engine_df: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(metrics_df))
    labels = metrics_df["variant"].tolist()

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()
    metric_fields = [
        ("rmse", "RMSE"),
        ("mae", "MAE"),
        ("catastrophic_error_rate", "Catastrophic Error Rate"),
        ("acceptance_coverage", "Acceptance Coverage"),
    ]
    for ax, (field, title) in zip(axes, metric_fields):
        ax.bar(x, metrics_df[field].to_numpy(dtype=float))
        ax.set_xticks(x, labels, rotation=15, ha="right")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "ablation_summary_metrics.png", dpi=220)
    plt.close(fig)

    for mode in ["Baseline", "Physics-Informed"]:
        subset = per_engine_df[per_engine_df["model_mode"] == mode]
        if subset.empty:
            continue
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        colors = np.where(subset["use_ri_gating"].to_numpy(dtype=bool), "#1f77b4", "#ff7f0e")
        ax.scatter(subset["ri"], subset["abs_error_used"], c=colors, alpha=0.8)
        ax.set_xlabel("Reliability Index (RI)")
        ax.set_ylabel("Absolute Error (Used Prediction)")
        ax.set_title(f"RI vs Error - {mode}")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        slug = mode.lower().replace("-", "_").replace(" ", "_")
        fig.savefig(figures_dir / f"ri_error_scatter_{slug}.png", dpi=220)
        plt.close(fig)


def run_report(
    *,
    test_path: Path,
    rul_path: Path,
    reports_dir: Path,
    catastrophic_threshold: float,
) -> Dict[str, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = reports_dir / "tables"
    figures_dir = reports_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    true_rul = _load_true_rul(rul_path)

    baseline_model = load_attention_model(get_model_weights_path("Baseline"))
    pi_model = load_attention_model(get_model_weights_path("Physics-Informed"))

    baseline_result = predict_rul_detailed_from_csv(test_path, model=baseline_model)
    pi_result = predict_rul_detailed_from_csv(test_path, model=pi_model)

    frames = [
        _build_variant_rows(
            baseline_result,
            true_rul,
            variant_name="Baseline",
            model_mode="Baseline",
            use_ri_gating=False,
            catastrophic_threshold=catastrophic_threshold,
        ),
        _build_variant_rows(
            baseline_result,
            true_rul,
            variant_name="Baseline+RI",
            model_mode="Baseline",
            use_ri_gating=True,
            catastrophic_threshold=catastrophic_threshold,
        ),
        _build_variant_rows(
            pi_result,
            true_rul,
            variant_name="PI",
            model_mode="Physics-Informed",
            use_ri_gating=False,
            catastrophic_threshold=catastrophic_threshold,
        ),
        _build_variant_rows(
            pi_result,
            true_rul,
            variant_name="PI+RI",
            model_mode="Physics-Informed",
            use_ri_gating=True,
            catastrophic_threshold=catastrophic_threshold,
        ),
    ]
    per_engine_df = pd.concat(frames, ignore_index=True)
    metrics = [_compute_metrics(group) for _, group in per_engine_df.groupby("variant", sort=False)]
    metrics_df = pd.DataFrame(metrics)

    per_engine_path = tables_dir / "ablation_per_engine_predictions.csv"
    metrics_path = tables_dir / "ablation_metrics.csv"
    per_engine_df.to_csv(per_engine_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    _save_plots(metrics_df, per_engine_df, figures_dir)

    return {
        "metrics_csv": metrics_path,
        "per_engine_csv": per_engine_path,
        "figures_dir": figures_dir,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ablation report for Baseline/PI with RI gating variants.")
    parser.add_argument("--test-path", type=Path, help="C-MAPSS test file (e.g., test_FD001.txt).")
    parser.add_argument("--rul-path", type=Path, help="RUL file matching test set (e.g., RUL_FD001.txt).")
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR, help="Output report directory.")
    parser.add_argument(
        "--cat-threshold",
        type=float,
        default=20.0,
        help="Absolute error threshold for catastrophic error rate.",
    )
    parser.add_argument(
        "--freeze-config",
        type=Path,
        help="Path to reproducibility freeze JSON (fills defaults for paths/threshold).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    freeze_config: Dict[str, object] = {}
    if args.freeze_config is not None:
        freeze_config = _load_freeze_config(args.freeze_config)

    test_path = args.test_path or (
        Path(str(freeze_config["test_path"])) if "test_path" in freeze_config else None
    )
    rul_path = args.rul_path or (Path(str(freeze_config["rul_path"])) if "rul_path" in freeze_config else None)
    if test_path is None or rul_path is None:
        raise ValueError("Both --test-path and --rul-path are required (directly or via --freeze-config).")

    reports_dir = args.reports_dir
    if args.reports_dir == DEFAULT_REPORTS_DIR and "reports_dir" in freeze_config:
        reports_dir = Path(str(freeze_config["reports_dir"]))

    cat_threshold = args.cat_threshold
    if args.cat_threshold == 20.0 and "catastrophic_threshold" in freeze_config:
        cat_threshold = float(freeze_config["catastrophic_threshold"])

    outputs = run_report(
        test_path=test_path,
        rul_path=rul_path,
        reports_dir=reports_dir,
        catastrophic_threshold=cat_threshold,
    )
    print("Ablation report generated.")
    for key, value in outputs.items():
        print(f"- {key}: {value}")
    manifest = {
        "test_path": str(test_path),
        "rul_path": str(rul_path),
        "reports_dir": str(reports_dir),
        "catastrophic_threshold": float(cat_threshold),
        "freeze_config": str(args.freeze_config) if args.freeze_config else None,
        "outputs": {k: str(v) for k, v in outputs.items()},
    }
    manifest_path = reports_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"- manifest: {manifest_path}")
    if freeze_config:
        freeze_copy_path = reports_dir / "freeze_config_used.json"
        freeze_copy_path.write_text(json.dumps(freeze_config, indent=2), encoding="utf-8")
        print(f"- freeze_config_used: {freeze_copy_path}")


if __name__ == "__main__":
    main()
