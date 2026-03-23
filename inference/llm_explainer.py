"""LLM-Powered Explanation Engine for RUL predictions.

Generates natural language maintenance reports from prediction results.

Two modes:
  1. **LLM mode** — uses Google Gemini API for rich, contextual explanations
  2. **Template mode** — deterministic template-based explanations (no API needed)

The engine takes prediction context (RUL, RI, gate decision, sensor anomalies,
attention weights) and produces human-readable maintenance recommendations.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

# Sensor human-readable names
SENSOR_LABELS = {
    "s_2": "LPC outlet temp (T24)",
    "s_3": "HPC outlet temp (T30)",
    "s_4": "LPT outlet temp (T50)",
    "s_7": "HPC outlet pressure (P30)",
    "s_8": "Physical fan speed (Nf)",
    "s_9": "Physical core speed (Nc)",
    "s_11": "HPC static pressure (Ps30)",
    "s_12": "Fuel flow ratio (phi)",
    "s_13": "Corrected fan speed (NRf)",
    "s_14": "Corrected core speed (NRc)",
    "s_15": "Bypass ratio (BPR)",
    "s_17": "Bleed enthalpy (htBleed)",
    "s_20": "HPT coolant bleed (W31)",
    "s_21": "LPT coolant bleed (W32)",
}

# Urgency thresholds
URGENCY_LEVELS = {
    "CRITICAL": {"rul_max": 20, "color": "#ef4444", "icon": "🔴"},
    "HIGH": {"rul_max": 40, "color": "#f59e0b", "icon": "🟠"},
    "MODERATE": {"rul_max": 70, "color": "#eab308", "icon": "🟡"},
    "LOW": {"rul_max": 125, "color": "#10b981", "icon": "🟢"},
}


@dataclass
class ExplanationContext:
    """Input context for generating an explanation."""
    engine_id: int
    predicted_rul: float
    trusted_rul: float
    reliability_index: float
    decision: str
    reason_codes: List[str]
    window_predictions: List[float]
    attention_weights: Optional[List[float]] = None
    sensor_anomalies: Optional[Dict[str, str]] = None
    physics_risk: float = 0.0
    cpc_score: float = 1.0
    current_cycle: int = 0
    model_backend: str = "attention"
    model_mode: str = "Baseline"


def _get_urgency(rul: float) -> str:
    for level, cfg in URGENCY_LEVELS.items():
        if rul <= cfg["rul_max"]:
            return level
    return "LOW"


def _get_urgency_info(rul: float) -> Dict[str, str]:
    level = _get_urgency(rul)
    return {"level": level, **URGENCY_LEVELS[level]}


def _format_attention_insight(weights: List[float]) -> str:
    if not weights:
        return ""
    arr = np.array(weights)
    top_k = min(3, len(arr))
    top_indices = np.argsort(arr)[-top_k:][::-1]
    top_vals = arr[top_indices]
    parts = [f"timestep {idx+1} ({val:.1%})" for idx, val in zip(top_indices, top_vals)]
    return f"The attention model focused most on {', '.join(parts)} in the sensor window."


def _format_reason_codes(codes: List[str]) -> str:
    descriptions = {
        "HIGH_WINDOW_VARIANCE": "High variance across prediction windows indicates unstable estimation.",
        "HIGH_WINDOW_SPREAD": "Large spread in window predictions suggests inconsistent degradation patterns.",
        "MONOTONICITY_WARN": "RUL predictions are not consistently decreasing, violating expected degradation monotonicity.",
        "SMOOTHNESS_WARN": "Prediction trajectory shows erratic behavior suggesting sensor noise or model instability.",
        "BOUNDARY_VIOLATION": "Some predictions fall outside physical bounds (0–125 cycles).",
        "PHYSICS_RISK_HIGH": "Sensor readings indicate thermodynamic constraint violations.",
        "CONSISTENT": "Predictions are stable and consistent across all evaluation criteria.",
    }
    lines = []
    for code in codes:
        desc = descriptions.get(code, code.replace("_", " ").title())
        lines.append(f"• **{code}**: {desc}")
    return "\n".join(lines)


def _format_sensor_anomalies(anomalies: Dict[str, str]) -> str:
    if not anomalies:
        return "No sensor anomalies detected."
    lines = []
    for sensor, status in anomalies.items():
        label = SENSOR_LABELS.get(sensor, sensor)
        lines.append(f"• **{label}** ({sensor}): {status}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# Template-Based Explanation (always available)
# ═══════════════════════════════════════════════════════════════

def generate_template_explanation(ctx: ExplanationContext) -> str:
    """Generate a structured maintenance explanation using templates."""
    urgency = _get_urgency_info(ctx.predicted_rul)
    rul_delta = ctx.predicted_rul - ctx.trusted_rul

    lines = []

    # Header
    lines.append(f"## {urgency['icon']} Maintenance Brief — Engine U-{ctx.engine_id:03d}")
    lines.append("")

    # Summary
    lines.append(f"**Urgency Level:** {urgency['level']}")
    lines.append(f"**Gate Decision:** {ctx.decision}")
    lines.append(f"**Model:** {ctx.model_backend} ({ctx.model_mode})")
    lines.append("")

    # RUL Assessment
    lines.append("### RUL Assessment")
    lines.append(f"- **Predicted RUL:** {ctx.predicted_rul:.1f} cycles remaining")
    lines.append(f"- **Trusted RUL:** {ctx.trusted_rul:.1f} cycles (post-gating)")
    if abs(rul_delta) > 0.5:
        lines.append(f"- **Gating Adjustment:** {rul_delta:+.1f} cycles (conservative correction applied)")
    lines.append(f"- **Reliability Index:** {ctx.reliability_index:.2f}")
    if ctx.cpc_score < 1.0:
        lines.append(f"- **CPC Score:** {ctx.cpc_score:.2f} (physics consistency)")
    lines.append("")

    # Decision Rationale
    lines.append("### Decision Rationale")
    if ctx.decision == "ACCEPT":
        lines.append("The prediction is **accepted for operational use**. "
                     f"RI={ctx.reliability_index:.2f} exceeds the acceptance threshold (0.75). "
                     "Prediction windows are stable and physically consistent.")
    elif ctx.decision == "WARN":
        lines.append("The prediction is **usable with degraded confidence**. "
                     f"RI={ctx.reliability_index:.2f} falls between warning (0.45) and acceptance (0.75) thresholds. "
                     "Consider increased monitoring frequency.")
    else:
        lines.append("The prediction is **rejected for operational use**. "
                     f"RI={ctx.reliability_index:.2f} is below the warning threshold (0.45). "
                     "A conservative fallback RUL has been applied. Immediate inspection recommended.")
    lines.append("")

    # Diagnostics
    lines.append("### Diagnostic Flags")
    lines.append(_format_reason_codes(ctx.reason_codes))
    lines.append("")

    # Attention Insight
    if ctx.attention_weights:
        insight = _format_attention_insight(ctx.attention_weights)
        if insight:
            lines.append("### Model Attention Insight")
            lines.append(insight)
            lines.append("")

    # Sensor Anomalies
    if ctx.sensor_anomalies:
        lines.append("### Sensor Anomaly Flags")
        lines.append(_format_sensor_anomalies(ctx.sensor_anomalies))
        lines.append("")

    # Recommendations
    lines.append("### Recommended Actions")
    if ctx.predicted_rul < 20:
        lines.append("1. **Schedule immediate maintenance** — engine approaching end of useful life")
        lines.append("2. **Ground the engine** if alternate is available")
        lines.append("3. **Request borescope inspection** of high-pressure turbine blades")
    elif ctx.predicted_rul < 40:
        lines.append("1. **Schedule maintenance within 2 weeks** — degradation is progressing")
        lines.append("2. **Increase monitoring frequency** to every flight cycle")
        lines.append("3. **Review maintenance history** for recurring patterns")
    elif ctx.predicted_rul < 70:
        lines.append("1. **Plan maintenance** in the upcoming maintenance window")
        lines.append("2. **Continue standard monitoring** with trend analysis")
        lines.append("3. **Flag for next scheduled inspection**")
    else:
        lines.append("1. **No immediate action required** — engine is in healthy operating range")
        lines.append("2. **Continue routine monitoring**")
        lines.append("3. **Review again at next scheduled interval**")
    lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# LLM-Powered Explanation (optional Gemini API)
# ═══════════════════════════════════════════════════════════════

def _build_llm_prompt(ctx: ExplanationContext) -> str:
    """Build a structured prompt for the LLM."""
    urgency = _get_urgency(ctx.predicted_rul)
    window_str = ", ".join(f"{w:.1f}" for w in ctx.window_predictions[-5:])

    prompt = f"""You are an aviation maintenance AI expert analyzing turbofan engine health data.

Generate a concise, professional maintenance brief for Engine U-{ctx.engine_id:03d}.

Data:
- Predicted RUL: {ctx.predicted_rul:.1f} cycles
- Trusted RUL: {ctx.trusted_rul:.1f} cycles
- Reliability Index: {ctx.reliability_index:.2f}
- Gate Decision: {ctx.decision}
- Urgency: {urgency}
- Physics Risk: {ctx.physics_risk:.2f}
- CPC Score: {ctx.cpc_score:.2f}
- Window Predictions (last 5): [{window_str}]
- Reason Codes: {', '.join(ctx.reason_codes)}
- Model: {ctx.model_backend} ({ctx.model_mode})
- Current Cycle: {ctx.current_cycle}

Instructions:
1. Start with a one-line summary of the engine's health status
2. Explain why the gate decision was made (reference RI value and thresholds)
3. Highlight any concerning sensor or prediction patterns
4. Provide 2-3 specific, actionable maintenance recommendations
5. Keep the tone professional and suitable for a flight operations team
6. Use markdown formatting with headers and bullet points
7. Keep total response under 200 words
"""
    return prompt


def generate_llm_explanation(
    ctx: ExplanationContext,
    api_key: Optional[str] = None,
) -> Optional[str]:
    """Generate explanation using Google Gemini API.

    Returns None if API key is not available or API call fails.
    Falls back to template mode externally.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        return None

    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = _build_llm_prompt(ctx)
        response = model.generate_content(prompt)
        return response.text
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════
# Unified Explainer Interface
# ═══════════════════════════════════════════════════════════════

class MaintenanceExplainer:
    """Unified maintenance explanation generator.

    Tries LLM (Gemini) first, falls back to template-based explanations.
    """

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    @property
    def has_llm(self) -> bool:
        return self._api_key is not None

    @property
    def mode(self) -> str:
        return "llm" if self.has_llm else "template"

    def explain(
        self,
        ctx: ExplanationContext,
        force_template: bool = False,
    ) -> Dict[str, str]:
        """Generate maintenance explanation.

        Returns dict with 'text' (markdown), 'mode' (llm/template), and 'urgency'.
        """
        urgency_info = _get_urgency_info(ctx.predicted_rul)

        if not force_template and self.has_llm:
            llm_text = generate_llm_explanation(ctx, self._api_key)
            if llm_text:
                return {
                    "text": llm_text,
                    "mode": "llm",
                    "urgency": urgency_info["level"],
                    "urgency_icon": urgency_info["icon"],
                    "urgency_color": urgency_info["color"],
                }

        template_text = generate_template_explanation(ctx)
        return {
            "text": template_text,
            "mode": "template",
            "urgency": urgency_info["level"],
            "urgency_icon": urgency_info["icon"],
            "urgency_color": urgency_info["color"],
        }
