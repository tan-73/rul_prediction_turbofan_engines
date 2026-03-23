"""SHAP-style Explainability for RUL Predictions.

Provides per-sensor attribution analysis to explain which sensors
contributed most to each RUL prediction.

Two analysis modes:
  1. **Attention-based** — extracts attention weights from the GRU model
     and maps them to sensor importance via gradient-weighted aggregation.
  2. **Statistical** — computes sensor importance using correlation with
     RUL and variance analysis when SHAP/attention weights aren't available.

This module is model-agnostic: it can work with any backend's predictions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Sensor metadata for human-readable output
SENSOR_META = {
    "s_2": {"name": "LPC Outlet Temp (T24)", "unit": "°R", "group": "thermal"},
    "s_3": {"name": "HPC Outlet Temp (T30)", "unit": "°R", "group": "thermal"},
    "s_4": {"name": "LPT Outlet Temp (T50)", "unit": "°R", "group": "thermal"},
    "s_7": {"name": "HPC Outlet Pressure (P30)", "unit": "psia", "group": "pressure"},
    "s_8": {"name": "Physical Fan Speed (Nf)", "unit": "rpm", "group": "mechanical"},
    "s_9": {"name": "Physical Core Speed (Nc)", "unit": "rpm", "group": "mechanical"},
    "s_11": {"name": "HPC Static Pressure (Ps30)", "unit": "psia", "group": "pressure"},
    "s_12": {"name": "Fuel Flow Ratio (phi)", "unit": "pps/psi", "group": "flow"},
    "s_13": {"name": "Corrected Fan Speed (NRf)", "unit": "rpm", "group": "mechanical"},
    "s_14": {"name": "Corrected Core Speed (NRc)", "unit": "rpm", "group": "mechanical"},
    "s_15": {"name": "Bypass Ratio (BPR)", "unit": "-", "group": "flow"},
    "s_17": {"name": "Bleed Enthalpy (htBleed)", "unit": "-", "group": "thermal"},
    "s_20": {"name": "HPT Coolant Bleed (W31)", "unit": "lbm/s", "group": "flow"},
    "s_21": {"name": "LPT Coolant Bleed (W32)", "unit": "lbm/s", "group": "flow"},
}

# Sensors used by the model (after feature selection)
ACTIVE_SENSORS = list(SENSOR_META.keys())


@dataclass
class SensorContribution:
    """Attribution result for a single sensor."""
    sensor_id: str
    name: str
    contribution: float  # positive = increases RUL risk, negative = healthy
    abs_contribution: float
    group: str
    unit: str
    direction: str  # "increases_risk" or "decreases_risk"
    rank: int


# ═══════════════════════════════════════════════════════════════
# Attention-Based Importance
# ═══════════════════════════════════════════════════════════════

def compute_attention_sensor_importance(
    attention_weights: List[float],
    sensor_window: np.ndarray,
) -> List[SensorContribution]:
    """Compute sensor importance by attention-weighting sensor variance.

    Uses attention weights (over timesteps) combined with per-sensor
    variance in the attended window to estimate sensor contributions.

    Args:
        attention_weights: shape (T,) — attention over window timesteps
        sensor_window: shape (T, n_sensors) — sensor values in the window
    """
    attn = np.array(attention_weights, dtype=float)
    window = np.array(sensor_window, dtype=float)

    if len(window.shape) == 1:
        n_sensors = len(ACTIVE_SENSORS)
        window = window.reshape(-1, n_sensors)

    T, n_sensors = window.shape
    if len(attn) != T:
        attn = np.ones(T) / T  # uniform fallback

    # Compute attention-weighted sensor variance
    # Higher variance in highly-attended timesteps → higher importance
    importance = np.zeros(n_sensors)
    for s in range(n_sensors):
        sensor_vals = window[:, s]
        # Weighted deviation from mean
        weighted_mean = np.average(sensor_vals, weights=attn)
        weighted_var = np.average((sensor_vals - weighted_mean) ** 2, weights=attn)
        # Trend in attended region (positive trend = increasing degradation)
        trend = np.polyfit(np.arange(T), sensor_vals, deg=1, w=attn)[0]
        importance[s] = np.sqrt(weighted_var) * np.sign(trend)

    return _build_contributions(importance)


# ═══════════════════════════════════════════════════════════════
# Statistical Importance (model-agnostic fallback)
# ═══════════════════════════════════════════════════════════════

def compute_statistical_sensor_importance(
    sensor_df: pd.DataFrame,
    rul_values: Optional[np.ndarray] = None,
) -> List[SensorContribution]:
    """Compute sensor importance using statistical analysis.

    Uses coefficient of variation + correlation with RUL (if available)
    to estimate which sensors carry the most degradation information.

    Args:
        sensor_df: DataFrame with sensor columns (s_2, s_3, ..., s_21)
        rul_values: Optional RUL targets for correlation analysis
    """
    importance = np.zeros(len(ACTIVE_SENSORS))

    for i, sensor in enumerate(ACTIVE_SENSORS):
        if sensor not in sensor_df.columns:
            continue
        vals = sensor_df[sensor].values.astype(float)

        # Coefficient of variation (higher CV = more informative)
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        cv = std_val / max(abs(mean_val), 1e-10)

        # Trend strength
        if len(vals) > 2:
            trend = np.polyfit(np.arange(len(vals)), vals, deg=1)[0]
        else:
            trend = 0.0

        # Correlation with RUL (if available)
        corr = 0.0
        if rul_values is not None and len(rul_values) == len(vals):
            valid = ~(np.isnan(vals) | np.isnan(rul_values))
            if valid.sum() > 2:
                corr = float(np.corrcoef(vals[valid], rul_values[valid])[0, 1])

        # Combined importance: CV + |trend| + |correlation|
        if rul_values is not None:
            importance[i] = 0.3 * cv + 0.3 * abs(trend) + 0.4 * abs(corr)
            importance[i] *= np.sign(corr) if corr != 0 else np.sign(trend)
        else:
            importance[i] = 0.5 * cv + 0.5 * abs(trend)
            importance[i] *= np.sign(trend)

    return _build_contributions(importance)


# ═══════════════════════════════════════════════════════════════
# Group-Level Analysis
# ═══════════════════════════════════════════════════════════════

def compute_group_importance(
    contributions: List[SensorContribution],
) -> Dict[str, Dict[str, float]]:
    """Aggregate sensor contributions by physical group."""
    groups: Dict[str, list] = {}
    for c in contributions:
        groups.setdefault(c.group, []).append(c.abs_contribution)

    result = {}
    for group, values in groups.items():
        result[group] = {
            "total": float(sum(values)),
            "mean": float(np.mean(values)),
            "max": float(max(values)),
            "count": len(values),
        }

    # Sort by total importance
    return dict(sorted(result.items(), key=lambda x: x[1]["total"], reverse=True))


def compute_anomaly_flags(
    sensor_df: pd.DataFrame,
    z_threshold: float = 2.5,
) -> Dict[str, str]:
    """Flag sensors with anomalous readings using z-score analysis.

    Args:
        sensor_df: DataFrame with sensor columns
        z_threshold: Z-score threshold for anomaly flagging

    Returns:
        Dict mapping sensor_id → anomaly description
    """
    anomalies = {}
    for sensor in ACTIVE_SENSORS:
        if sensor not in sensor_df.columns:
            continue
        vals = sensor_df[sensor].values.astype(float)
        if len(vals) < 3:
            continue

        mean_val = np.mean(vals)
        std_val = np.std(vals)
        if std_val < 1e-10:
            continue

        # Check last few readings for z-score anomaly
        last_vals = vals[-min(5, len(vals)):]
        z_scores = (last_vals - mean_val) / std_val

        max_z = np.max(np.abs(z_scores))
        if max_z > z_threshold:
            direction = "elevated" if z_scores[-1] > 0 else "depressed"
            label = SENSOR_META.get(sensor, {}).get("name", sensor)
            anomalies[sensor] = f"{direction} (z={max_z:.1f})"

    return anomalies


# ═══════════════════════════════════════════════════════════════
# Unified Explainer Interface
# ═══════════════════════════════════════════════════════════════

class SHAPExplainer:
    """Unified sensor attribution explainer.

    Uses attention weights when available, falls back to statistical analysis.
    """

    def explain(
        self,
        sensor_df: pd.DataFrame,
        attention_weights: Optional[List[float]] = None,
        sensor_window: Optional[np.ndarray] = None,
        rul_values: Optional[np.ndarray] = None,
    ) -> Dict[str, object]:
        """Compute sensor attributions and anomaly flags.

        Returns dict with contributions, group importance, and anomaly flags.
        """
        # Choose analysis mode
        if attention_weights is not None and sensor_window is not None:
            contributions = compute_attention_sensor_importance(
                attention_weights, sensor_window
            )
            analysis_mode = "attention_weighted"
        else:
            contributions = compute_statistical_sensor_importance(
                sensor_df, rul_values
            )
            analysis_mode = "statistical"

        # Group analysis
        group_importance = compute_group_importance(contributions)

        # Anomaly detection
        anomalies = compute_anomaly_flags(sensor_df)

        # Top contributors
        top_risk = [c for c in contributions if c.direction == "increases_risk"][:5]
        top_healthy = [c for c in contributions if c.direction == "decreases_risk"][:5]

        return {
            "contributions": [
                {
                    "sensor_id": c.sensor_id,
                    "name": c.name,
                    "contribution": round(c.contribution, 4),
                    "abs_contribution": round(c.abs_contribution, 4),
                    "group": c.group,
                    "direction": c.direction,
                    "rank": c.rank,
                }
                for c in contributions
            ],
            "top_risk_factors": [
                {"sensor": c.sensor_id, "name": c.name, "value": round(c.contribution, 4)}
                for c in top_risk
            ],
            "top_healthy_factors": [
                {"sensor": c.sensor_id, "name": c.name, "value": round(c.contribution, 4)}
                for c in top_healthy
            ],
            "group_importance": group_importance,
            "anomalies": anomalies,
            "analysis_mode": analysis_mode,
            "n_sensors_analyzed": len(contributions),
        }


# ═══════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════

def _build_contributions(importance: np.ndarray) -> List[SensorContribution]:
    """Convert raw importance array into sorted SensorContribution list."""
    contributions = []
    abs_imp = np.abs(importance)

    # Normalize to [-1, 1]
    max_val = np.max(abs_imp) if np.max(abs_imp) > 0 else 1.0
    normalized = importance / max_val

    sorted_indices = np.argsort(abs_imp)[::-1]

    for rank, idx in enumerate(sorted_indices):
        if idx >= len(ACTIVE_SENSORS):
            continue
        sensor = ACTIVE_SENSORS[idx]
        meta = SENSOR_META.get(sensor, {"name": sensor, "unit": "-", "group": "other"})
        val = float(normalized[idx])
        contributions.append(SensorContribution(
            sensor_id=sensor,
            name=meta["name"],
            contribution=val,
            abs_contribution=abs(val),
            group=meta["group"],
            unit=meta["unit"],
            direction="increases_risk" if val > 0 else "decreases_risk",
            rank=rank + 1,
        ))

    return contributions
