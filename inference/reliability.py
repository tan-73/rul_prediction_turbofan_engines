from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def _safe_ratio(numerator: float, denominator: float, eps: float = 1e-6) -> float:
    return float(numerator) / float(max(abs(denominator), eps))


def _score_from_ratio(ratio: float) -> float:
    return float(1.0 / (1.0 + max(ratio, 0.0)))


def compute_reliability_index(
    window_predictions: Sequence[float],
    *,
    early_rul: float = 125.0,
    delta_up: float = 1.0,
    w_stability: float = 0.6,
    w_physics: float = 0.4,
) -> Dict[str, float]:
    preds = np.asarray(window_predictions, dtype=float).reshape(-1)
    if preds.size == 0:
        raise ValueError("window_predictions cannot be empty.")

    pred_mean = float(np.mean(preds))
    pred_std = float(np.std(preds))
    pred_min = float(np.min(preds))
    pred_max = float(np.max(preds))

    cv = _safe_ratio(pred_std, pred_mean)
    spread = _safe_ratio(pred_max - pred_min, pred_mean)
    stability_cv_score = _score_from_ratio(cv)
    stability_spread_score = _score_from_ratio(spread)
    stability_score = float((stability_cv_score + stability_spread_score) / 2.0)

    diffs = np.diff(preds)
    monotonic_violation_rate = float(np.mean(diffs > delta_up)) if diffs.size else 0.0
    monotonic_score = float(max(0.0, 1.0 - monotonic_violation_rate))

    second_diff = np.diff(preds, n=2)
    diff_scale = float(np.mean(np.abs(diffs))) if diffs.size else 0.0
    smoothness_ratio = _safe_ratio(float(np.mean(np.abs(second_diff))) if second_diff.size else 0.0, diff_scale + 1e-6)
    smoothness_score = _score_from_ratio(smoothness_ratio)

    boundary_violations = np.logical_or(preds < 0.0, preds > early_rul)
    boundary_violation_rate = float(np.mean(boundary_violations))
    boundary_score = float(max(0.0, 1.0 - boundary_violation_rate))

    physics_score = float((0.5 * monotonic_score) + (0.3 * smoothness_score) + (0.2 * boundary_score))
    reliability_index = float((w_stability * stability_score) + (w_physics * physics_score))
    reliability_index = float(min(max(reliability_index, 0.0), 1.0))

    return {
        "ri": reliability_index,
        "stability_score": stability_score,
        "physics_score": physics_score,
        "cv": cv,
        "spread_ratio": spread,
        "monotonic_violation_rate": monotonic_violation_rate,
        "smoothness_ratio": smoothness_ratio,
        "boundary_violation_rate": boundary_violation_rate,
        "window_mean": pred_mean,
        "window_std": pred_std,
        "window_min": pred_min,
        "window_max": pred_max,
    }


def gate_prediction(
    predicted_rul: float,
    reliability_index: float,
    window_predictions: Sequence[float],
    *,
    accept_threshold: float = 0.75,
    warning_threshold: float = 0.45,
) -> Dict[str, float | str]:
    if accept_threshold <= warning_threshold:
        raise ValueError("accept_threshold must be greater than warning_threshold.")

    preds = np.asarray(window_predictions, dtype=float).reshape(-1)
    pred_std = float(np.std(preds)) if preds.size else 0.0
    conservative_fallback = float(max(0.0, predicted_rul - (2.0 * pred_std)))

    if reliability_index >= accept_threshold:
        decision = "ACCEPT"
        trusted_rul = float(predicted_rul)
        message = "Prediction accepted for operational use."
    elif reliability_index >= warning_threshold:
        decision = "WARN"
        trusted_rul = float(predicted_rul)
        message = "Prediction usable with degraded confidence."
    else:
        decision = "REJECT"
        trusted_rul = conservative_fallback
        message = "Prediction rejected. Conservative fallback policy applied."

    return {
        "decision": decision,
        "trusted_rul": float(trusted_rul),
        "fallback_rul": conservative_fallback,
        "message": message,
    }


def reason_codes(metrics: Dict[str, float]) -> List[str]:
    reasons: List[str] = []
    if metrics["cv"] > 0.35:
        reasons.append("HIGH_WINDOW_VARIANCE")
    if metrics["spread_ratio"] > 0.60:
        reasons.append("HIGH_WINDOW_SPREAD")
    if metrics["monotonic_violation_rate"] > 0.20:
        reasons.append("MONOTONICITY_WARN")
    if metrics["smoothness_ratio"] > 1.20:
        reasons.append("SMOOTHNESS_WARN")
    if metrics["boundary_violation_rate"] > 0.0:
        reasons.append("BOUNDARY_VIOLATION")
    if not reasons:
        reasons.append("CONSISTENT")
    return reasons


def evaluate_reliability_log(
    df,
    *,
    catastrophic_error_threshold: float = 20.0,
) -> Dict[str, float]:
    required = {"true_rul", "predicted_rul", "ri"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    true_rul = np.asarray(df["true_rul"], dtype=float)
    pred_rul = np.asarray(df["predicted_rul"], dtype=float)
    ri = np.asarray(df["ri"], dtype=float)
    errors = np.abs(pred_rul - true_rul)

    corr = float(np.corrcoef(ri, errors)[0, 1]) if len(df) > 1 else float("nan")
    catastrophic_mask = errors > float(catastrophic_error_threshold)

    out: Dict[str, float] = {
        "samples": float(len(df)),
        "mean_abs_error": float(np.mean(errors)),
        "ri_error_correlation": corr,
        "catastrophic_rate": float(np.mean(catastrophic_mask)),
    }

    if "decision" in df.columns:
        decisions = np.asarray(df["decision"], dtype=str)
        accepted = decisions == "ACCEPT"
        warn = decisions == "WARN"
        reject = decisions == "REJECT"

        out["accept_rate"] = float(np.mean(accepted))
        out["warn_rate"] = float(np.mean(warn))
        out["reject_rate"] = float(np.mean(reject))

        if len(decisions) > 1:
            flips = np.sum(decisions[1:] != decisions[:-1])
            out["decision_flip_rate"] = float(flips / (len(decisions) - 1))
        else:
            out["decision_flip_rate"] = 0.0

        if np.any(accepted):
            out["catastrophic_rate_accepted"] = float(np.mean(catastrophic_mask[accepted]))
        if np.any(reject):
            out["catastrophic_rate_rejected"] = float(np.mean(catastrophic_mask[reject]))

    return out
