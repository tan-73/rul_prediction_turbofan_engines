from __future__ import annotations

import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    px = None
    go = None

from backend.model_service import ModelService
from inference.attention_model import RAW_COLUMN_NAMES, maintenance_status
from inference.reliability import evaluate_reliability_log
from inference.cvae_trajectory import TrajectoryGenerator
from inference.llm_explainer import MaintenanceExplainer, ExplanationContext
from inference.shap_explainer import SHAPExplainer

# ═══════════════════════════════════════════════════════════════
# Page Config & Theme
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="PhysGen-RUL — Aero-Engine Prognostics",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS
CSS_PATH = Path(__file__).parent / "assets" / "theme.css"
if CSS_PATH.exists():
    st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

# Constants
COMPARE_MODE = "Compare (Baseline vs PI)"
LIVE_STATE_FILE = Path("logs") / "live_state.json"
LIVE_PREDICTIONS_FILE = Path("logs") / "mqtt_predictions.csv"
NODERED_STATE_FILE = Path("logs") / "nodered_live_state.json"
DECISION_COLORS = {"ACCEPT": "#10b981", "WARN": "#f59e0b", "REJECT": "#ef4444", "RAW": "#3b82f6"}
SENSOR_GROUP_COLORS = {
    "thermal": "#f97316",
    "pressure": "#38bdf8",
    "mechanical": "#a78bfa",
    "flow": "#22c55e",
    "other": "#94a3b8",
}

PLOTLY_TEMPLATE = "plotly_dark"
DEFAULT_HOURS_PER_CYCLE = 1.0


# ═══════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════
def _decision_badge(decision: str) -> str:
    cls = {"ACCEPT": "badge-accept", "WARN": "badge-warn", "REJECT": "badge-reject"}.get(decision, "badge-warn")
    icon = {"ACCEPT": "✅", "WARN": "⚠️", "REJECT": "🔴"}.get(decision, "❓")
    return f'<span class="{cls}">{icon} {decision}</span>'


def _ri_bar(value: float) -> str:
    pct = max(0, min(100, value * 100))
    cls = "ri-fill-high" if value >= 0.75 else "ri-fill-mid" if value >= 0.45 else "ri-fill-low"
    return f'<div class="ri-bar"><div class="ri-bar-fill {cls}" style="width:{pct}%"></div></div>'


def _gate_color(decision: str) -> str:
    return f'<span class="gate-{decision.lower()}">{decision}</span>'


def _cycles_to_hours(cycles: float, hours_per_cycle: float) -> float:
    return float(max(cycles, 0.0) * max(hours_per_cycle, 0.0))


def _format_hours(hours: float) -> str:
    if hours >= 24.0:
        return f"{hours:.1f} h ({hours / 24.0:.1f} days)"
    return f"{hours:.1f} h"


@st.cache_resource
def get_model_service() -> ModelService:
    return ModelService()

@st.cache_resource
def get_trajectory_gen() -> TrajectoryGenerator:
    return TrajectoryGenerator()

@st.cache_resource
def get_explainer() -> MaintenanceExplainer:
    return MaintenanceExplainer()

@st.cache_resource
def get_shap() -> SHAPExplainer:
    return SHAPExplainer()


def _read_live_state() -> dict:
    if not LIVE_STATE_FILE.exists():
        return {}
    try:
        return json.loads(LIVE_STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_live_predictions(limit: int = 250) -> pd.DataFrame:
    if not LIVE_PREDICTIONS_FILE.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(LIVE_PREDICTIONS_FILE)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
        df = df.sort_values("timestamp_utc")
    return df.tail(limit).reset_index(drop=True)


def build_reliability_df(result: dict) -> pd.DataFrame:
    rows = []
    for engine_id in result["engine_ids"]:
        rel = result["per_engine_reliability"][engine_id]
        rows.append({
            "engine_id": engine_id,
            "ri": rel["ri"],
            "decision": rel["decision"],
            "trusted_rul": rel["trusted_rul"],
            "raw_pred_rul": result["per_engine_mean_rul"][engine_id],
            "window_std": rel["window_std"],
            "monotonic_violation_rate": rel["monotonic_violation_rate"],
            "smoothness_ratio": rel["smoothness_ratio"],
            "reason_codes": ", ".join(rel["reason_codes"]),
        })
    return pd.DataFrame(rows)


def _make_plotly_dark(fig):
    """Apply consistent dark styling to plotly figures."""
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.8)",
        font=dict(family="Inter, sans-serif", color="#e2e8f0"),
        margin=dict(l=40, r=20, t=40, b=30),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="rgba(51,65,85,0.5)", zerolinecolor="rgba(51,65,85,0.5)")
    fig.update_yaxes(gridcolor="rgba(51,65,85,0.5)", zerolinecolor="rgba(51,65,85,0.5)")
    return fig


def _lookup_engine_map(mapping: dict, engine_id: int, default=None):
    if not isinstance(mapping, dict):
        return default
    return mapping.get(engine_id, mapping.get(str(engine_id), default))


def _build_model_core_payload(
    result: dict,
    engine_id: int,
    *,
    model_backend: str,
    model_mode: str,
    view_mode: str,
) -> dict:
    raw_df = result.get("raw_df", pd.DataFrame())
    if isinstance(raw_df, list):
        raw_df = pd.DataFrame(raw_df)

    engine_df = pd.DataFrame()
    if not raw_df.empty and "unit_nr" in raw_df.columns:
        engine_df = raw_df[raw_df["unit_nr"].astype(int) == int(engine_id)].sort_values("time_cycles")

    rel = _lookup_engine_map(result.get("per_engine_reliability", {}), engine_id, {})
    pred_rul = float(_lookup_engine_map(result.get("per_engine_mean_rul", {}), engine_id, 0.0) or 0.0)
    trusted_rul = float(rel.get("trusted_rul", pred_rul) or 0.0)
    ri = float(rel.get("ri", 0.0) or 0.0)
    decision = str(rel.get("decision", "RAW"))
    attention = _lookup_engine_map(result.get("per_engine_last_attention", {}), engine_id, []) or []
    windows = _lookup_engine_map(result.get("per_engine_window_rul", {}), engine_id, []) or []
    physics = _lookup_engine_map(result.get("per_engine_physics", {}), engine_id, {})

    shap_result = {"contributions": [], "anomalies": {}, "group_importance": {}, "analysis_mode": "unavailable"}
    if not engine_df.empty:
        try:
            sensor_cols_for_window = [c for c in engine_df.columns if str(c).startswith("s_")]
            sensor_window = engine_df[sensor_cols_for_window].values[-30:] if len(engine_df) >= 30 else None
            shap_result = get_shap().explain(
                sensor_df=engine_df,
                attention_weights=attention if attention else None,
                sensor_window=sensor_window,
            )
        except Exception:
            shap_result = {"contributions": [], "anomalies": {}, "group_importance": {}, "analysis_mode": "fallback"}

    contrib_by_sensor = {c["sensor_id"]: c for c in shap_result.get("contributions", [])}
    sensor_cols = [c for c in RAW_COLUMN_NAMES if c.startswith("s_")]
    latest = engine_df.iloc[-1].to_dict() if not engine_df.empty else {}
    sensors = []
    for idx, sensor in enumerate(sensor_cols):
        contrib = contrib_by_sensor.get(sensor, {})
        group = str(contrib.get("group", "other"))
        value = float(latest.get(sensor, 0.0) or 0.0)
        sensors.append(
            {
                "id": sensor,
                "label": sensor.upper(),
                "value": round(value, 3),
                "group": group,
                "color": SENSOR_GROUP_COLORS.get(group, SENSOR_GROUP_COLORS["other"]),
                "contribution": float(contrib.get("contribution", 0.0) or 0.0),
                "rank": int(contrib.get("rank", idx + 1) or idx + 1),
                "anomaly": shap_result.get("anomalies", {}).get(sensor, ""),
            }
        )

    trajectory = get_trajectory_gen().generate(
        current_rul=pred_rul,
        reliability_index=ri,
        window_predictions=[float(v) for v in windows] if windows else [pred_rul],
        n_trajectories=24,
        trajectory_length=28,
    )

    return {
        "engineId": int(engine_id),
        "backend": model_backend,
        "mode": model_mode,
        "viewMode": view_mode,
        "predictedRul": round(pred_rul, 3),
        "trustedRul": round(trusted_rul, 3),
        "ri": round(ri, 4),
        "decision": decision,
        "gateColor": DECISION_COLORS.get(decision, "#3b82f6"),
        "reasonCodes": rel.get("reason_codes", []),
        "windowStd": round(float(rel.get("window_std", 0.0) or 0.0), 4),
        "physicsRisk": round(float(rel.get("physics_risk", physics.get("mean_physics_risk", 0.0)) or 0.0), 4),
        "cpc": round(float(rel.get("cpc", physics.get("mean_cpc", rel.get("physics_score", 1.0))) or 1.0), 4),
        "attention": [float(v) for v in attention],
        "windows": [float(v) for v in windows],
        "sensors": sensors,
        "groupImportance": shap_result.get("group_importance", {}),
        "shapMode": shap_result.get("analysis_mode", "unavailable"),
        "trajectory": {
            "mode": trajectory.get("generation_mode", "monte_carlo"),
            "mean": trajectory.get("mean", []),
            "ci95Lower": trajectory.get("ci_95_lower", []),
            "ci95Upper": trajectory.get("ci_95_upper", []),
        },
        "cycle": int(latest.get("time_cycles", 0) or 0),
    }


def _render_model_core_component(payload: dict, height: int = 900) -> None:
    data_json = json.dumps(payload)
    html = f"""
<div id="model-core-root">
  <canvas id="model-core-canvas"></canvas>
  <div class="hud top-left">
    <div class="kicker">ENGINE U-{payload['engineId']:03d}</div>
    <div class="title">Model Digital Core</div>
    <div class="sub">{payload['backend']} · {payload['mode']} · {payload['viewMode']}</div>
  </div>
  <div class="hud top-right">
    <div class="metric"><span>RUL</span><b>{payload['predictedRul']:.1f}</b></div>
    <div class="metric"><span>Trusted</span><b>{payload['trustedRul']:.1f}</b></div>
    <div class="metric"><span>RI</span><b>{payload['ri']:.2f}</b></div>
    <div class="pill" style="border-color:{payload['gateColor']};color:{payload['gateColor']}">{payload['decision']}</div>
  </div>
  <div class="hud bottom-left">
    <div class="legend"><i style="background:#f97316"></i> Thermal</div>
    <div class="legend"><i style="background:#38bdf8"></i> Pressure</div>
    <div class="legend"><i style="background:#a78bfa"></i> Mechanical</div>
    <div class="legend"><i style="background:#22c55e"></i> Flow</div>
  </div>
  <div class="hud detail-card" id="detail-card">
    <div class="detail-kicker">Hover or click an element</div>
    <div class="detail-title" id="detail-title">Model Core</div>
    <div class="detail-body" id="detail-body">Sensor nodes, model layers, RI gate, and future RUL trajectory are interactive.</div>
  </div>
  <div class="hud controls">
    <button id="zoom-in" type="button">+</button>
    <button id="zoom-out" type="button">-</button>
    <button id="reset-view" type="button">Reset</button>
  </div>
</div>
<script>
const DATA = {data_json};
const root = document.getElementById("model-core-root");
const canvas = document.getElementById("model-core-canvas");
const ctx = canvas.getContext("2d");
const titleEl = document.getElementById("detail-title");
const bodyEl = document.getElementById("detail-body");
let w = 0, h = 0, dpr = window.devicePixelRatio || 1;
let pointer = {{x: 0, y: 0, rawX: 0, rawY: 0, inside: false}};
let selected = null;
let hover = null;
let paused = false;
let zoom = 1.18;
let hitTargets = [];
function resize() {{
  const rect = root.getBoundingClientRect();
  w = rect.width; h = rect.height;
  canvas.width = Math.floor(w * dpr);
  canvas.height = Math.floor(h * dpr);
  canvas.style.width = w + "px";
  canvas.style.height = h + "px";
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}}
window.addEventListener("resize", resize);
root.addEventListener("pointermove", (e) => {{
  const r = root.getBoundingClientRect();
  pointer.rawX = e.clientX - r.left;
  pointer.rawY = e.clientY - r.top;
  pointer.x = (pointer.rawX - w / 2) / w;
  pointer.y = (pointer.rawY - h / 2) / h;
  pointer.inside = true;
  hover = hitTest(pointer.rawX, pointer.rawY);
  root.style.cursor = hover ? "pointer" : "default";
  updateDetail(hover || selected);
}});
root.addEventListener("pointerleave", () => {{ pointer.inside = false; hover = null; root.style.cursor = "default"; updateDetail(selected); }});
root.addEventListener("click", () => {{
  selected = hover || selected;
  updateDetail(selected);
}});
root.addEventListener("dblclick", () => {{ selected = null; updateDetail(null); }});
root.addEventListener("wheel", (e) => {{
  e.preventDefault();
  zoom = clamp(zoom + (e.deltaY < 0 ? 0.08 : -0.08), 0.78, 1.72);
}}, {{passive:false}});
document.getElementById("zoom-in").addEventListener("click", () => zoom = clamp(zoom + 0.12, 0.78, 1.72));
document.getElementById("zoom-out").addEventListener("click", () => zoom = clamp(zoom - 0.12, 0.78, 1.72));
document.getElementById("reset-view").addEventListener("click", () => {{ zoom = 1.18; selected = null; updateDetail(null); }});
resize();

function clamp(v, lo, hi) {{ return Math.max(lo, Math.min(hi, v)); }}
function fmt(v, digits=3) {{
  const n = Number(v);
  return Number.isFinite(n) ? n.toFixed(digits) : String(v ?? "-");
}}
function hexToRgb(hex) {{
  const clean = hex.replace("#", "");
  const num = parseInt(clean, 16);
  return {{r:(num>>16)&255, g:(num>>8)&255, b:num&255}};
}}
function describeTarget(target) {{
  if (!target) {{
    return {{
      title: "Model Core",
      body: `Backend ${{DATA.backend}} in ${{DATA.mode}} mode. RUL ${{fmt(DATA.predictedRul, 1)}}, trusted RUL ${{fmt(DATA.trustedRul, 1)}}, RI ${{fmt(DATA.ri, 2)}}.`
    }};
  }}
  if (target.kind === "sensor") {{
    const s = target.data;
    return {{
      title: `${{s.id.toUpperCase()}} · ${{s.group}} sensor`,
      body: `Latest value ${{fmt(s.value, 3)}}. SHAP-style contribution ${{fmt(s.contribution, 3)}}. Rank ${{s.rank}}. ${{s.anomaly ? "Anomaly: " + s.anomaly + "." : "No anomaly flag."}}`
    }};
  }}
  if (target.kind === "core") {{
    return {{
      title: "Inference Core",
      body: `${{DATA.backend}} produced raw RUL ${{fmt(DATA.predictedRul, 1)}} cycles. This value remains separate from post-prediction gating.`
    }};
  }}
  if (target.kind === "gate") {{
    return {{
      title: "Reliability Gate",
      body: `Decision ${{DATA.decision}} with RI ${{fmt(DATA.ri, 3)}}. Trusted RUL ${{fmt(DATA.trustedRul, 1)}}. Reason codes: ${{(DATA.reasonCodes || []).join(", ") || "CONSISTENT"}}.`
    }};
  }}
  if (target.kind === "window") {{
    return {{
      title: "30-Cycle Window",
      body: `Window predictions: ${{(DATA.windows || []).map(v => fmt(v, 1)).join(", ") || "unavailable"}}. Window std ${{fmt(DATA.windowStd, 3)}}.`
    }};
  }}
  if (target.kind === "trajectory") {{
    return {{
      title: "Future RUL Trajectory",
      body: `${{DATA.trajectory.mode}} fan generated from current RUL and reliability. The line shows expected future degradation across upcoming cycles.`
    }};
  }}
  if (target.kind === "physics") {{
    return {{
      title: "Physics Consistency Field",
      body: `CPC ${{fmt(DATA.cpc, 2)}} and physics risk ${{fmt(DATA.physicsRisk, 2)}}. This layer is emphasized for physics-informed backends.`
    }};
  }}
  return {{title: target.label || "Element", body: "Interactive model element."}};
}}
function updateDetail(target) {{
  const d = describeTarget(target);
  titleEl.textContent = d.title;
  bodyEl.textContent = d.body;
}}
function registerHit(kind, x, y, r, data=null, label="") {{
  hitTargets.push({{kind, x, y, r, data, label}});
}}
function hitTest(x, y) {{
  for (let i = hitTargets.length - 1; i >= 0; i--) {{
    const t = hitTargets[i];
    const dx = x - t.x, dy = y - t.y;
    if (dx * dx + dy * dy <= t.r * t.r) return t;
  }}
  return null;
}}
function glowCircle(x, y, r, color, alpha=1) {{
  const c = hexToRgb(color);
  const g = ctx.createRadialGradient(x, y, 0, x, y, r * 3.2);
  g.addColorStop(0, `rgba(${{c.r}},${{c.g}},${{c.b}},${{alpha}})`);
  g.addColorStop(0.45, `rgba(${{c.r}},${{c.g}},${{c.b}},${{alpha * 0.22}})`);
  g.addColorStop(1, `rgba(${{c.r}},${{c.g}},${{c.b}},0)`);
  ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, r * 3.2, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = color; ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
}}
function drawRing(cx, cy, rx, ry, color, alpha, width=1.2) {{
  ctx.save();
  ctx.strokeStyle = color; ctx.globalAlpha = alpha; ctx.lineWidth = width;
  ctx.beginPath(); ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2); ctx.stroke();
  ctx.restore();
}}
function drawLabel(text, x, y, color = "#dbeafe", align = "center", size = 13) {{
  ctx.save(); ctx.fillStyle = color; ctx.font = `${{size}}px Inter, Segoe UI, sans-serif`; ctx.textAlign = align;
  ctx.shadowColor = "rgba(0,0,0,.8)"; ctx.shadowBlur = 8; ctx.fillText(text, x, y); ctx.restore();
}}
function nodePosition(i, count, t, cx, cy, rx, ry, tilt) {{
  const angle = (Math.PI * 2 * i / count) + t * (0.095 + (i % 4) * 0.012);
  const depth = Math.sin(angle + tilt);
  return {{
    x: cx + Math.cos(angle) * rx + pointer.x * depth * 56,
    y: cy + Math.sin(angle + tilt) * ry + pointer.y * depth * 42,
    depth
  }};
}}
function drawTrajectory(cx, cy, t) {{
  const mean = DATA.trajectory.mean || [];
  const lo = DATA.trajectory.ci95Lower || [];
  const hi = DATA.trajectory.ci95Upper || [];
  if (mean.length < 2) return;
  const startX = Math.min(w - 420, cx + 235 * zoom), startY = cy - 120 * zoom, width = Math.min(420, w * 0.34), scaleY = 2.15 * zoom;
  registerHit("trajectory", startX + width * .5, startY + 120, Math.max(90, width * .38));
  ctx.save(); ctx.lineWidth = 1; ctx.globalAlpha = DATA.viewMode === "Trajectory View" ? 0.9 : 0.42;
  ctx.strokeStyle = "rgba(96,165,250,.24)";
  for (let band of [lo, hi]) {{
    ctx.beginPath();
    band.forEach((v, i) => {{
      const x = startX + i / (band.length - 1) * width;
      const y = startY + (125 - v) * scaleY;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }});
    ctx.stroke();
  }}
  ctx.strokeStyle = "#06b6d4"; ctx.lineWidth = 2.4; ctx.globalAlpha = 0.9;
  ctx.beginPath();
  mean.forEach((v, i) => {{
    const x = startX + i / (mean.length - 1) * width;
    const y = startY + (125 - v) * scaleY + Math.sin(t * 1.8 + i * .4) * 2;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }});
  ctx.stroke();
  drawLabel("future RUL fan", startX + width * .52, startY - 18, "#93c5fd", "center", 14);
  ctx.restore();
}}
function draw(tMs) {{
  const t = tMs / 1000;
  hitTargets = [];
  ctx.clearRect(0, 0, w, h);
  const bg = ctx.createLinearGradient(0, 0, w, h);
  bg.addColorStop(0, "#020617"); bg.addColorStop(.45, "#08111f"); bg.addColorStop(1, "#111827");
  ctx.fillStyle = bg; ctx.fillRect(0, 0, w, h);
  const cx = w * .46 + pointer.x * 26, cy = h * .52 + pointer.y * 18;
  const base = Math.min(w, h);
  const scene = base * zoom;
  const rxOuter = scene * .43, ryOuter = scene * .22;
  const rxMid = scene * .31, ryMid = scene * .142;
  const rxCore = scene * .18, ryCore = scene * .085;
  registerHit("gate", cx, cy, rxCore * 1.14);
  registerHit("window", cx, cy + ryMid * .75, rxMid * .72);
  registerHit("physics", cx, cy, rxCore * 1.62);
  drawRing(cx, cy, rxOuter, ryOuter, "rgba(148,163,184,.55)", .75, 1.2);
  drawRing(cx, cy, rxOuter * .88, ryOuter * 1.32, "rgba(56,189,248,.34)", .7, 1);
  drawRing(cx, cy, rxMid, ryMid, "rgba(139,92,246,.54)", .9, 2.5);
  drawRing(cx, cy, rxCore, ryCore, DATA.gateColor, .9, 4.5);
  drawLabel("sensor shell", cx - rxOuter - 30, cy - ryOuter - 26, "#94a3b8", "left", 15);
  drawLabel("30-cycle window", cx - rxMid, cy + ryMid + 38, "#c4b5fd", "left", 15);
  drawLabel("RI gate", cx + rxCore - 24, cy - ryCore - 28, DATA.gateColor, "left", 15);

  const sensors = DATA.sensors || [];
  const activeSensors = sensors.slice().sort((a,b) => a.rank - b.rank);
  activeSensors.forEach((s, i) => {{
    const p = nodePosition(i, activeSensors.length, t, cx, cy, rxOuter, ryOuter, i % 2 ? .2 : -.34);
    const risk = clamp(Math.abs(s.contribution || 0), 0, 1);
    const isActive = (selected && selected.kind === "sensor" && selected.data.id === s.id) || (hover && hover.kind === "sensor" && hover.data.id === s.id);
    const size = 7.5 + risk * 9 + (s.anomaly ? 5 : 0) + (isActive ? 5 : 0);
    ctx.globalAlpha = p.depth < -0.55 ? .42 : .95;
    if (DATA.viewMode === "Attention View" || DATA.viewMode === "Architecture View") {{
      ctx.strokeStyle = `rgba(148,163,184,${{0.08 + risk * 0.22}})`;
      ctx.lineWidth = 1 + risk * 2;
      ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.quadraticCurveTo(cx, cy - 30, cx, cy); ctx.stroke();
    }}
    if (isActive) {{
      ctx.strokeStyle = "#f8fafc"; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.arc(p.x, p.y, size + 8, 0, Math.PI * 2); ctx.stroke();
    }}
    glowCircle(p.x, p.y, size, s.anomaly ? "#ef4444" : s.color, .95);
    registerHit("sensor", p.x, p.y, Math.max(18, size + 12), s);
    if (risk > .45 || s.anomaly || isActive || i < 8) drawLabel(s.id, p.x, p.y - size - 10, "#e2e8f0", "center", isActive ? 15 : 12);
    ctx.globalAlpha = 1;
  }});

  const attn = DATA.attention || [];
  if (attn.length) {{
    const maxA = Math.max(...attn, .001);
    attn.forEach((v, i) => {{
      const a0 = -Math.PI * .95 + i / attn.length * Math.PI * 1.9;
      const a1 = -Math.PI * .95 + (i + .72) / attn.length * Math.PI * 1.9;
      ctx.strokeStyle = `rgba(245,158,11,${{DATA.viewMode === "Attention View" ? .18 + (v/maxA) * .78 : .10 + (v/maxA) * .28}})`;
      ctx.lineWidth = 2 + (v / maxA) * 7;
      ctx.beginPath(); ctx.ellipse(cx, cy, rxMid + 8, ryMid + 18, 0, a0 + t * .07, a1 + t * .07); ctx.stroke();
    }});
  }}

  for (let i = 0; i < 42; i++) {{
    const phase = (t * .34 + i / 42) % 1;
    const x = cx - rxOuter + phase * rxOuter * 2;
    const y = cy + Math.sin(phase * Math.PI * 2 + i) * ryOuter * .48;
    const alpha = .12 + Math.sin(phase * Math.PI) * .38;
    glowCircle(x, y, 1.4 + (i % 3), "#60a5fa", alpha);
  }}

  const coreRadius = scene * .105;
  registerHit("core", cx, cy, coreRadius + 20);
  const coreGrad = ctx.createRadialGradient(cx - 25, cy - 25, 2, cx, cy, scene * .18);
  coreGrad.addColorStop(0, "rgba(255,255,255,.92)");
  coreGrad.addColorStop(.18, "rgba(125,211,252,.9)");
  coreGrad.addColorStop(.55, "rgba(59,130,246,.44)");
  coreGrad.addColorStop(1, "rgba(15,23,42,.12)");
  ctx.fillStyle = coreGrad; ctx.beginPath(); ctx.arc(cx, cy, coreRadius, 0, Math.PI * 2); ctx.fill();
  ctx.strokeStyle = DATA.gateColor; ctx.lineWidth = 2.5; ctx.beginPath();
  ctx.arc(cx, cy, coreRadius + 17, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * clamp(DATA.ri, 0, 1)); ctx.stroke();
  drawLabel(DATA.backend.toUpperCase(), cx, cy - 8, "#f8fafc", "center", 18);
  drawLabel(`RUL ${{DATA.predictedRul.toFixed(1)}}`, cx, cy + 18, "#bae6fd", "center", 16);

  if (DATA.viewMode === "Physics View") {{
    const pulse = .5 + Math.sin(t * 3) * .5;
    ctx.strokeStyle = `rgba(239,68,68,${{.18 + DATA.physicsRisk * .62 + pulse * .12}})`;
    ctx.lineWidth = 8 + DATA.physicsRisk * 16;
    ctx.beginPath(); ctx.arc(cx, cy, scene * (.15 + DATA.physicsRisk * .12), 0, Math.PI * 2); ctx.stroke();
  }}
  drawTrajectory(cx, cy, t);
  if (!paused) requestAnimationFrame(draw);
}}
updateDetail(null);
requestAnimationFrame(draw);
</script>
<style>
#model-core-root {{
  position: relative;
  height: {height}px;
  overflow: hidden;
  border: 1px solid rgba(148,163,184,.22);
  border-radius: 18px;
  background: #020617;
  font-family: Inter, Segoe UI, sans-serif;
  color: #e5e7eb;
  box-shadow: inset 0 0 80px rgba(14,165,233,.08), 0 24px 80px rgba(0,0,0,.25);
  min-height: 760px;
}}
#model-core-canvas {{ position: absolute; inset: 0; }}
.hud {{
  position: absolute;
  padding: 12px 14px;
  border: 1px solid rgba(148,163,184,.18);
  background: rgba(2,6,23,.54);
  backdrop-filter: blur(14px);
  border-radius: 14px;
  box-shadow: 0 10px 30px rgba(0,0,0,.24);
}}
.top-left {{ top: 16px; left: 16px; }}
.top-right {{ top: 16px; right: 16px; display: grid; grid-template-columns: repeat(2, minmax(72px, 1fr)); gap: 8px; }}
.bottom-left {{ bottom: 16px; left: 16px; display: grid; gap: 6px; }}
.detail-card {{ right: 16px; bottom: 16px; width: min(360px, calc(100% - 32px)); color: #cbd5e1; }}
.detail-kicker {{ color:#38bdf8; font-size:10px; letter-spacing:.08em; text-transform:uppercase; font-weight:800; }}
.detail-title {{ color:#f8fafc; font-size:18px; font-weight:800; margin-top:4px; }}
.detail-body {{ font-size:13px; line-height:1.45; margin-top:7px; }}
.controls {{ left: 50%; bottom: 16px; transform: translateX(-50%); display:flex; gap:8px; padding:8px; }}
.controls button {{
  appearance:none; border:1px solid rgba(148,163,184,.25); background:rgba(15,23,42,.72); color:#e2e8f0;
  border-radius:10px; min-width:38px; height:34px; padding:0 12px; font-weight:800; cursor:pointer;
}}
.controls button:hover {{ border-color: rgba(56,189,248,.7); color:#f8fafc; }}
.kicker {{ color: #38bdf8; font-size: 11px; letter-spacing: .08em; font-weight: 800; }}
.title {{ font-size: 22px; font-weight: 800; margin-top: 2px; }}
.sub {{ color: #cbd5e1; font-size: 12px; margin-top: 3px; }}
.metric {{ min-width: 74px; }}
.metric span {{ display:block; color:#94a3b8; font-size:11px; }}
.metric b {{ display:block; color:#f8fafc; font-size:18px; }}
.pill {{ grid-column: 1 / -1; border: 1px solid; border-radius: 999px; padding: 6px 10px; text-align:center; font-size: 12px; font-weight: 800; }}
.legend {{ color:#cbd5e1; font-size: 12px; }}
.legend i {{ display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:7px; box-shadow:0 0 12px currentColor; }}
@media (max-width: 760px) {{
  #model-core-root {{ height: 720px; border-radius: 12px; }}
  .top-right {{ left: 16px; right: auto; top: 112px; }}
  .detail-card {{ left: 16px; right: 16px; width:auto; }}
  .controls {{ bottom: 168px; }}
}}
</style>
"""
    components.html(html, height=height + 8, scrolling=False)


# ═══════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<p class="hero-title">✈️ PhysGen-RUL</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Physics-Integrated Generative Edge-AI<br>IEEE IES GenAI Challenge 2026 · NASA C-MAPSS FD001</p>', unsafe_allow_html=True)
    st.divider()

    st.markdown("### Monitor")
    page = st.radio(
        "Navigation",
        [
            "🏠 Fleet Overview",
            "📡 Live Digital Twin",
            "📈 RUL Trajectories",
            "🧬 Model Internals",
            "🔬 Batch Inference",
            "⚙️ Settings",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("### Inference")

    service = get_model_service()
    backends = service.list_backends()
    backend_names = [b["name"] for b in backends]
    backend_descs = {b["name"]: b["description"] for b in backends}
    selected_backend = st.selectbox(
        "Runtime Backend",
        options=backend_names,
        index=0,
        format_func=lambda x: f"{x}  —  {backend_descs.get(x, '')}",
    )
    model_mode = st.selectbox("Model Mode", ["Baseline", "Physics-Informed", COMPARE_MODE], index=0)
    hours_per_cycle = st.number_input(
        "Estimated Hours per Cycle",
        min_value=0.01,
        max_value=24.0,
        value=DEFAULT_HOURS_PER_CYCLE,
        step=0.25,
        help="C-MAPSS predicts RUL in cycles. This factor converts predicted cycles into approximate flight hours.",
    )

    st.divider()
    st.markdown("### Status")
    live_state = _read_live_state()
    if live_state:
        st.markdown('<span class="status-dot live"></span> MQTT Connected', unsafe_allow_html=True)
        st.caption(f"Unit: {live_state.get('engine_id', '-')} · Cycle: {live_state.get('received_cycles_for_engine', 0)}")
    else:
        st.markdown('<span class="status-dot offline"></span> MQTT Offline', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# Page: Fleet Overview
# ═══════════════════════════════════════════════════════════════
if page == "🏠 Fleet Overview":
    st.markdown('<p class="section-header">Fleet Overview Dashboard</p>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload sensor CSV (C-MAPSS format)", type=["csv", "txt"], key="fleet_upload")
    if uploaded:
        file_bytes = uploaded.getvalue()
        sig = (uploaded.name, len(file_bytes), model_mode, selected_backend)
        if st.session_state.get("fleet_sig") != sig:
            st.session_state["fleet_sig"] = sig
            st.session_state["fleet_bytes"] = file_bytes
            st.session_state.pop("fleet_result", None)

        if st.button("🚀 Run Fleet Inference", type="primary"):
            with st.spinner("Running inference across fleet..."):
                try:
                    if model_mode == COMPARE_MODE:
                        st.session_state["fleet_result"] = service.compare(file_bytes, model_backend=selected_backend)
                        st.session_state["fleet_compare"] = True
                    else:
                        st.session_state["fleet_result"] = service.infer(file_bytes, model_mode=model_mode, model_backend=selected_backend)
                        st.session_state["fleet_compare"] = False
                except Exception as exc:
                    st.error(f"Inference failed: {exc}")

    if "fleet_result" in st.session_state and not st.session_state.get("fleet_compare", False):
        result = st.session_state["fleet_result"]
        predictions = result["per_engine_mean_rul"]
        overall_rul = float(result["overall_mean_rul"])
        overall_ri = float(result["overall_reliability_index"])
        engine_ids = result["engine_ids"]

        # Top metric row
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Fleet Engines", len(predictions), help="Total engines in uploaded data")
        m2.metric("Mean RUL", f"{overall_rul:.1f}", help="Fleet average remaining useful life (cycles)")
        m3.metric("Reliability Index", f"{overall_ri:.2f}", help="Fleet mean RI (0-1)")
        m4.metric("Est. Time Before Risk", _format_hours(_cycles_to_hours(overall_rul, hours_per_cycle)), help="Approximate flight time derived from RUL cycles and the sidebar hours-per-cycle setting.")

        # Count decisions
        rel_df = build_reliability_df(result)
        rel_df["est_hours_to_risk"] = rel_df["trusted_rul"].apply(lambda v: _cycles_to_hours(float(v), hours_per_cycle))
        decision_counts = rel_df["decision"].value_counts().to_dict()
        m5.metric("Accept Gate", f"{decision_counts.get('ACCEPT', 0)}", help=f"RI >= 0.75")
        m6.metric("Warn / Reject", f"{decision_counts.get('WARN', 0)} / {decision_counts.get('REJECT', 0)}")

        st.divider()

        # Fleet engine table
        col_table, col_chart = st.columns([1, 1])
        with col_table:
            st.markdown('<p class="section-header">Fleet Engine Table</p>', unsafe_allow_html=True)
            table_data = []
            for eid in engine_ids:
                rel = result["per_engine_reliability"][eid]
                rul = float(predictions[eid])
                cpc_val = rel.get("cpc", rel.get("physics_score", "-"))
                table_data.append({
                    "Unit": f"U-{eid:03d}",
                    "Cycles": int(result["num_test_windows_list"][engine_ids.index(eid)]),
                    "Mean RUL": f"{rul:.0f}",
                    "Est. Hours": _format_hours(_cycles_to_hours(float(rel["trusted_rul"]), hours_per_cycle)),
                    "RI": f"{float(rel['ri']):.2f}",
                    "CPC": f"{float(cpc_val):.2f}" if isinstance(cpc_val, (int, float)) else str(cpc_val),
                    "Gate": rel["decision"],
                    "Mode": model_mode.split("(")[0].strip(),
                })
            fleet_df = pd.DataFrame(table_data)
            st.dataframe(fleet_df, use_container_width=True, height=min(400, 40 + len(fleet_df) * 35))

        with col_chart:
            st.markdown('<p class="section-header">RUL Distribution</p>', unsafe_allow_html=True)
            if px is not None:
                bar_df = pd.DataFrame({"engine_id": [f"U-{e:03d}" for e in engine_ids], "rul": [float(predictions[e]) for e in engine_ids], "decision": [result["per_engine_reliability"][e]["decision"] for e in engine_ids]})
                fig = px.bar(bar_df, x="engine_id", y="rul", color="decision", color_discrete_map=DECISION_COLORS, title="Per-Engine RUL with Gate Decision")
                _make_plotly_dark(fig)
                st.plotly_chart(fig, use_container_width=True)

        # Reliability gating section
        st.divider()
        st.markdown('<p class="section-header">Reliability Gating</p>', unsafe_allow_html=True)
        g1, g2 = st.columns([1, 1])
        with g1:
            if px is not None:
                counts = rel_df["decision"].value_counts().reset_index()
                counts.columns = ["decision", "count"]
                pie = px.pie(counts, names="decision", values="count", color="decision", color_discrete_map=DECISION_COLORS, hole=0.5, title="Fleet Gate Distribution")
                _make_plotly_dark(pie)
                pie.update_traces(textposition="inside", textinfo="percent+label")
                st.plotly_chart(pie, use_container_width=True)

        with g2:
            if px is not None:
                scatter = px.scatter(rel_df, x="ri", y="raw_pred_rul", color="decision", size="window_std", hover_data=["engine_id", "trusted_rul"], color_discrete_map=DECISION_COLORS, title="Reliability vs Prediction")
                _make_plotly_dark(scatter)
                st.plotly_chart(scatter, use_container_width=True)

        # Detailed per-engine analysis
        st.divider()
        st.markdown('<p class="section-header">Per-Engine Deep Dive</p>', unsafe_allow_html=True)
        sel_engine = st.selectbox("Select Engine", engine_ids, format_func=lambda x: f"U-{x:03d}")
        if sel_engine is not None:
            sel_rel = result["per_engine_reliability"][sel_engine]
            sel_windows = result["per_engine_window_rul"][sel_engine]
            sel_attention = result["per_engine_last_attention"][sel_engine]

            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Predicted RUL", f"{float(predictions[sel_engine]):.1f}")
            d2.metric("Reliability Index", f"{float(sel_rel['ri']):.2f}")
            d3.markdown(f"**Gate Decision**<br>{_decision_badge(sel_rel['decision'])}", unsafe_allow_html=True)
            d4.metric("Est. Time Before Risk", _format_hours(_cycles_to_hours(float(sel_rel["trusted_rul"]), hours_per_cycle)))

            if px is not None:
                dc1, dc2 = st.columns(2)
                with dc1:
                    win_df = pd.DataFrame({"window": range(1, len(sel_windows)+1), "rul": sel_windows})
                    fig_w = px.area(win_df, x="window", y="rul", title=f"U-{sel_engine:03d} Window-Level RUL", color_discrete_sequence=["#06b6d4"])
                    _make_plotly_dark(fig_w)
                    st.plotly_chart(fig_w, use_container_width=True)
                with dc2:
                    att_df = pd.DataFrame({"timestep": range(1, len(sel_attention)+1), "weight": sel_attention})
                    fig_a = px.bar(att_df, x="timestep", y="weight", color="weight", color_continuous_scale="Sunset", title=f"U-{sel_engine:03d} Attention Weights")
                    _make_plotly_dark(fig_a)
                    st.plotly_chart(fig_a, use_container_width=True)

            # ── GenAI: Trajectory Fan Plot ──
            st.divider()
            st.markdown('<p class="section-header">🔮 Probabilistic RUL Trajectories (cVAE / Monte Carlo)</p>', unsafe_allow_html=True)
            traj_gen = get_trajectory_gen()
            traj_result = traj_gen.generate(
                current_rul=float(predictions[sel_engine]),
                reliability_index=float(sel_rel["ri"]),
                window_predictions=sel_windows,
                n_trajectories=50,
                trajectory_length=30,
            )
            if px is not None and go is not None:
                steps = list(range(1, len(traj_result["mean"]) + 1))
                fig_fan = go.Figure()
                # 95% CI band
                fig_fan.add_trace(go.Scatter(
                    x=steps + steps[::-1],
                    y=traj_result["ci_95_upper"] + traj_result["ci_95_lower"][::-1],
                    fill="toself", fillcolor="rgba(59,130,246,0.1)",
                    line=dict(color="rgba(0,0,0,0)"), name="95% CI", showlegend=True,
                ))
                # 50% CI band
                fig_fan.add_trace(go.Scatter(
                    x=steps + steps[::-1],
                    y=traj_result["ci_50_upper"] + traj_result["ci_50_lower"][::-1],
                    fill="toself", fillcolor="rgba(139,92,246,0.25)",
                    line=dict(color="rgba(0,0,0,0)"), name="50% CI", showlegend=True,
                ))
                # Median trajectory
                fig_fan.add_trace(go.Scatter(
                    x=steps, y=traj_result["median"],
                    line=dict(color="#06b6d4", width=3), name="Median",
                ))
                # Mean trajectory
                fig_fan.add_trace(go.Scatter(
                    x=steps, y=traj_result["mean"],
                    line=dict(color="#f59e0b", width=2, dash="dash"), name="Mean",
                ))
                # Sample trajectories (faint)
                for i in range(min(8, len(traj_result["trajectories"]))):
                    fig_fan.add_trace(go.Scatter(
                        x=steps, y=traj_result["trajectories"][i],
                        line=dict(color="rgba(148,163,184,0.15)", width=1),
                        showlegend=False, hoverinfo="skip",
                    ))
                fig_fan.update_layout(title=f"U-{sel_engine:03d} Probabilistic RUL Fan ({traj_result['generation_mode'].upper()})", xaxis_title="Future Cycles", yaxis_title="Predicted RUL")
                _make_plotly_dark(fig_fan)
                st.plotly_chart(fig_fan, use_container_width=True)

                fc1, fc2, fc3 = st.columns(3)
                fc1.metric("Mode", traj_result["generation_mode"].upper())
                if "mean_time_to_failure" in traj_result:
                    fc2.metric("Mean Time to Failure", f"{traj_result['mean_time_to_failure']:.0f} cycles")
                if "physics_violation_rate" in traj_result:
                    fc3.metric("Physics Violation Rate", f"{traj_result['physics_violation_rate']:.1%}")

            # ── GenAI: SHAP Sensor Attribution ──
            st.divider()
            st.markdown('<p class="section-header">🔬 Sensor Attribution (SHAP-style)</p>', unsafe_allow_html=True)

            raw_df = result["raw_df"]
            if isinstance(raw_df, list):
                raw_df = pd.DataFrame(raw_df)
            engine_df = raw_df[raw_df["unit_nr"] == sel_engine].sort_values("time_cycles")

            shap_ex = get_shap()
            shap_result = shap_ex.explain(
                sensor_df=engine_df,
                attention_weights=sel_attention if sel_attention else None,
                sensor_window=engine_df[[c for c in engine_df.columns if c.startswith("s_")]].values[-30:] if len(engine_df) >= 30 else None,
            )

            if px is not None:
                contrib_df = pd.DataFrame(shap_result["contributions"])
                if not contrib_df.empty:
                    contrib_df = contrib_df.sort_values("contribution")
                    colors = ["#ef4444" if d == "increases_risk" else "#10b981" for d in contrib_df["direction"]]
                    fig_shap = go.Figure(go.Bar(
                        x=contrib_df["contribution"], y=contrib_df["name"],
                        orientation="h", marker_color=colors,
                    ))
                    fig_shap.update_layout(title=f"U-{sel_engine:03d} Sensor Attribution", xaxis_title="Contribution (← healthy | risk →)", yaxis_title="")
                    _make_plotly_dark(fig_shap)
                    fig_shap.update_layout(height=max(350, len(contrib_df) * 30))
                    st.plotly_chart(fig_shap, use_container_width=True)

                    # Group importance
                    if shap_result["group_importance"]:
                        grp_df = pd.DataFrame([
                            {"group": g.title(), "importance": v["total"]}
                            for g, v in shap_result["group_importance"].items()
                        ])
                        fig_grp = px.bar(grp_df, x="group", y="importance", color="group",
                                        color_discrete_sequence=["#f59e0b", "#ef4444", "#06b6d4", "#8b5cf6"],
                                        title="Sensor Group Importance")
                        _make_plotly_dark(fig_grp)
                        st.plotly_chart(fig_grp, use_container_width=True)

                    # Anomaly flags
                    if shap_result["anomalies"]:
                        st.warning(f"⚠️ **Anomalous sensors detected:** {', '.join(f'{k} ({v})' for k, v in shap_result['anomalies'].items())}")

            # ── GenAI: AI Maintenance Brief ──
            st.divider()
            st.markdown('<p class="section-header">🤖 AI Maintenance Brief</p>', unsafe_allow_html=True)

            explainer = get_explainer()
            expl_ctx = ExplanationContext(
                engine_id=sel_engine,
                predicted_rul=float(predictions[sel_engine]),
                trusted_rul=float(sel_rel["trusted_rul"]),
                reliability_index=float(sel_rel["ri"]),
                decision=sel_rel["decision"],
                reason_codes=sel_rel["reason_codes"],
                window_predictions=sel_windows,
                attention_weights=sel_attention,
                sensor_anomalies=shap_result.get("anomalies"),
                model_backend=selected_backend,
                model_mode=model_mode,
            )
            explanation = explainer.explain(expl_ctx)

            badge_cls = {"CRITICAL": "badge-reject", "HIGH": "badge-reject", "MODERATE": "badge-warn", "LOW": "badge-accept"}
            st.markdown(f'{explanation["urgency_icon"]} **Urgency: {explanation["urgency"]}** · Generated via `{explanation["mode"]}` engine', unsafe_allow_html=True)
            st.markdown(explanation["text"])

            # Sensor trends
            sensor_cols = [c for c in RAW_COLUMN_NAMES if c.startswith("s_")]
            st.markdown('<p class="section-header">📊 Sensor Telemetry</p>', unsafe_allow_html=True)
            sel_sensors = st.multiselect("Sensors", sensor_cols, default=["s_2", "s_3", "s_4", "s_7", "s_11", "s_15"])
            if sel_sensors and px is not None:
                trend = engine_df[["time_cycles"] + sel_sensors].melt(id_vars="time_cycles", var_name="sensor", value_name="reading")
                fig_t = px.line(trend, x="time_cycles", y="reading", color="sensor", title=f"U-{sel_engine:03d} Sensor Telemetry")
                _make_plotly_dark(fig_t)
                st.plotly_chart(fig_t, use_container_width=True)

        # Download
        st.download_button("📥 Download Fleet Report (CSV)", data=rel_df.to_csv(index=False).encode(), file_name="fleet_report.csv", mime="text/csv")

    # Compare mode
    if "fleet_result" in st.session_state and st.session_state.get("fleet_compare", False):
        compare = st.session_state["fleet_result"]
        bl = compare["baseline"]
        pi = compare["physics_informed"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Baseline RUL", f"{float(bl['overall_mean_rul']):.1f}")
        m2.metric("PI RUL", f"{float(pi['overall_mean_rul']):.1f}", delta=f"{float(pi['overall_mean_rul'])-float(bl['overall_mean_rul']):.1f}")
        m3.metric("Baseline RI", f"{float(bl['overall_reliability_index']):.2f}")
        m4.metric("PI RI", f"{float(pi['overall_reliability_index']):.2f}", delta=f"{float(pi['overall_reliability_index'])-float(bl['overall_reliability_index']):.2f}")

        deltas = pd.DataFrame(compare["engine_deltas"])
        if not deltas.empty and px is not None:
            fig_cmp = px.bar(deltas.melt(id_vars=["engine_id"], value_vars=["baseline_pred_rul", "pi_pred_rul"], var_name="mode", value_name="rul"), x="engine_id", y="rul", color="mode", barmode="group", title="Baseline vs PI per Engine", color_discrete_sequence=["#3b82f6", "#f59e0b"])
            _make_plotly_dark(fig_cmp)
            st.plotly_chart(fig_cmp, use_container_width=True)
            st.dataframe(deltas, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# Page: Live Digital Twin
# ═══════════════════════════════════════════════════════════════
elif page == "📡 Live Digital Twin":
    st.markdown('<p class="section-header">Live Digital Twin Feed (MQTT)</p>', unsafe_allow_html=True)

    lcol1, lcol2, lcol3 = st.columns([1, 1, 2])
    refresh_now = lcol1.button("🔄 Refresh")
    auto_refresh = lcol2.checkbox("Auto-refresh", value=False, key="mqtt_auto")
    refresh_sec = int(lcol3.slider("Interval (sec)", 1, 10, 2))

    live_state = _read_live_state()
    live_df = _read_live_predictions()
    latest = (live_state.get("latest_prediction") or {}) if live_state else {}
    if not latest and not live_df.empty:
        latest = live_df.iloc[-1].to_dict()

    is_buffering = live_state.get("buffering", False) if live_state else False
    cycles_left = live_state.get("cycles_until_first_prediction", 0) if live_state else 0
    has_pred = "predicted_rul" in latest

    lm = st.columns(7)
    lm[0].metric("Engine", str(live_state.get("engine_id", "-")) if live_state else "-")
    lm[1].metric("Cycles", int(live_state.get("received_cycles_for_engine", 0)) if live_state else 0)
    
    if is_buffering and not has_pred:
        lm[2].metric("Predicted RUL", "Wait...", delta=f"{cycles_left} cycles left", delta_color="off")
        lm[3].metric("Trusted RUL", "Wait...")
        lm[4].metric("RI", "Wait...")
        lm[5].metric("Decision", "BUFFERING")
        lm[6].metric("Est. Hours", "Wait...")
    else:
        lm[2].metric("Predicted RUL", f"{float(latest.get('predicted_rul', 0)):.1f}")
        lm[3].metric("Trusted RUL", f"{float(latest.get('trusted_rul', 0)):.1f}")
        lm[4].metric("RI", f"{float(latest.get('ri', 0)):.3f}")
        lm[5].metric("Decision", str(latest.get("decision", "-")))
        lm[6].metric("Est. Hours", _format_hours(_cycles_to_hours(float(latest.get("trusted_rul", 0)), hours_per_cycle)))

    if live_state:
        st.caption(f"Last update: {live_state.get('updated_at_utc', '-')}")

    if not live_df.empty:
        if px is not None:
            cols_plot = [c for c in ["predicted_rul", "trusted_rul"] if c in live_df.columns]
            x_col = "timestamp_utc" if "timestamp_utc" in live_df.columns else live_df.index.name or "index"
            if x_col == "index":
                live_df = live_df.reset_index()
            fig_live = px.line(live_df, x=x_col, y=cols_plot, title="Live RUL / Trusted RUL Trajectory", color_discrete_sequence=["#06b6d4", "#8b5cf6"])
            _make_plotly_dark(fig_live)
            st.plotly_chart(fig_live, use_container_width=True)

            if "ri" in live_df.columns:
                fig_ri = px.line(live_df, x=x_col, y="ri", title="Reliability Index over Time", color_discrete_sequence=["#10b981"])
                _make_plotly_dark(fig_ri)
                st.plotly_chart(fig_ri, use_container_width=True)

        if "decision" in live_df.columns:
            counts = live_df["decision"].value_counts().reset_index()
            counts.columns = ["decision", "count"]
            if px is not None:
                fig_dec = px.pie(counts, names="decision", values="count", color="decision", color_discrete_map=DECISION_COLORS, hole=0.45, title="Decision Distribution")
                _make_plotly_dark(fig_dec)
                st.plotly_chart(fig_dec, use_container_width=True)

        with st.expander("📋 Recent Live Rows"):
            st.dataframe(live_df.tail(40), use_container_width=True)
    else:
        st.info("No live MQTT predictions yet. Start ingest + digital twin streamer and refresh.")

    with st.expander("📖 Start MQTT Pipeline"):
        st.code("python ingestion\\mqtt_secure_ingest.py --broker 127.0.0.1 --port 1883 --topic engines/fd001/raw --model-mode Baseline --insecure-no-tls", language="powershell")
        st.code("python ingestion\\digital_twin_streamer.py --broker 127.0.0.1 --port 1883 --topic engines/fd001/raw --interval-sec 0.5 --cycles 3000", language="powershell")

    if auto_refresh and not refresh_now:
        time.sleep(max(refresh_sec, 1))
        st.rerun()


# ═══════════════════════════════════════════════════════════════
# Page: RUL Trajectories (Streaming Replay)
# ═══════════════════════════════════════════════════════════════
elif page == "📈 RUL Trajectories":
    st.markdown('<p class="section-header">Streaming Replay — Cycle-by-Cycle RUL</p>', unsafe_allow_html=True)

    replay_file = st.file_uploader("Upload sensor CSV", type=["csv", "txt"], key="replay_upload")
    if replay_file:
        replay_bytes = replay_file.getvalue()
        # Quick peek to get engine list
        try:
            peek = service.infer(replay_bytes, model_mode="Baseline", model_backend="attention")
            engines = peek["engine_ids"]
        except Exception:
            engines = [1]

        rc1, rc2 = st.columns(2)
        replay_engine = rc1.selectbox("Engine", engines, format_func=lambda x: f"U-{x:03d}")
        replay_step = rc2.slider("Step size", 1, 5, 1)

        if st.button("▶️ Run Replay", type="primary"):
            with st.spinner("Simulating real-time RUL updates..."):
                try:
                    replay = service.replay(replay_bytes, model_mode=model_mode, engine_id=int(replay_engine), step=replay_step, model_backend=selected_backend)
                    stream_df = pd.DataFrame(replay["rows"])
                    if not stream_df.empty and "trusted_rul" in stream_df.columns:
                        stream_df["est_hours_to_risk"] = stream_df["trusted_rul"].apply(lambda v: _cycles_to_hours(float(v), hours_per_cycle))
                    st.session_state["replay_df"] = stream_df
                except Exception as exc:
                    st.error(f"Replay failed: {exc}")

        if "replay_df" in st.session_state:
            stream_df = st.session_state["replay_df"]
            if px is not None:
                long = stream_df.melt(id_vars=["time_cycles"], value_vars=["predicted_rul", "trusted_rul"], var_name="series", value_name="rul")
                fig_s = px.line(
                    long,
                    x="time_cycles",
                    y="rul",
                    color="series",
                    line_dash="series",
                    markers=True,
                    title="Streaming Replay: Raw vs Trusted RUL",
                    color_discrete_map={"predicted_rul": "#06b6d4", "trusted_rul": "#8b5cf6"},
                    line_dash_map={"predicted_rul": "dash", "trusted_rul": "solid"},
                )
                _make_plotly_dark(fig_s)
                st.plotly_chart(fig_s, use_container_width=True)

                if stream_df["predicted_rul"].round(6).equals(stream_df["trusted_rul"].round(6)):
                    st.caption("`predicted_rul` overlaps exactly with `trusted_rul` here because the gate did not adjust the raw prediction.")

                if "reliability_index" in stream_df.columns:
                    fig_ri = px.line(stream_df, x="time_cycles", y="reliability_index", color="decision" if "decision" in stream_df.columns else None, title="Reliability Trajectory", color_discrete_map=DECISION_COLORS)
                    _make_plotly_dark(fig_ri)
                    st.plotly_chart(fig_ri, use_container_width=True)

            st.dataframe(stream_df, use_container_width=True)
            st.download_button("📥 Download Replay", data=stream_df.to_csv(index=False).encode(), file_name="replay_trajectory.csv", mime="text/csv")

    # Demo scenarios
    with st.expander("🎯 Demo Scenarios"):
        scenario_dir = Path("examples") / "scenarios"
        for label, fname in [("Stable", "scenario_stable_behavior.csv"), ("Noisy", "scenario_noisy_behavior.csv"), ("Rapid Degradation", "scenario_rapid_degradation.csv")]:
            p = scenario_dir / fname
            if p.exists():
                st.download_button(f"📥 {label}", data=p.read_bytes(), file_name=fname, mime="text/csv", key=f"dl_{fname}")


# ═══════════════════════════════════════════════════════════════
# Page: Model Internals
# ═══════════════════════════════════════════════════════════════
elif page == "🧬 Model Internals":
    st.markdown('<p class="section-header">Model Internals — Spatial Digital Core</p>', unsafe_allow_html=True)

    intro_l, intro_r = st.columns([1.5, 1])
    with intro_l:
        st.markdown(
            """
            This view turns one engine prediction into an interactive systems map:
            sensor nodes orbit the model core, attention arcs show recent timestep focus,
            the Reliability Index becomes the gate ring, and the future RUL fan projects outward.
            """
        )
    with intro_r:
        st.info("Use a C-MAPSS CSV with at least 30 cycles per selected engine. The visualization uses the same inference result shape as the rest of the dashboard.")

    source_mode = st.radio(
        "Source",
        ["Upload CSV", "Use Fleet Overview result", "Use demo sample"],
        horizontal=True,
        label_visibility="collapsed",
    )

    internals_bytes = None
    internals_name = ""
    if source_mode == "Upload CSV":
        internals_file = st.file_uploader("Upload sensor CSV", type=["csv", "txt"], key="internals_upload")
        if internals_file:
            internals_bytes = internals_file.getvalue()
            internals_name = internals_file.name
    elif source_mode == "Use Fleet Overview result":
        internals_bytes = st.session_state.get("fleet_bytes")
        internals_name = str(st.session_state.get("fleet_sig", ["fleet_result"])[0]) if internals_bytes else ""
        if not internals_bytes:
            st.warning("Run Fleet Overview inference first, or choose another source.")
    else:
        sample_path = Path("examples") / "sample_cmapss_engine_dual.csv"
        if sample_path.exists():
            internals_bytes = sample_path.read_bytes()
            internals_name = str(sample_path)
        else:
            st.error("Demo sample not found.")

    vc1, vc2, vc3 = st.columns([1, 1, 1])
    view_mode = vc1.selectbox(
        "Visual Mode",
        ["Architecture View", "Attention View", "Physics View", "Trajectory View"],
        index=0,
        help="Changes which internal layer is emphasized in the spatial scene.",
    )
    visual_height = int(vc2.slider("Viewport Height", 760, 1120, 940, 20))
    render_now = vc3.button("Render Digital Core", type="primary", disabled=internals_bytes is None)

    if internals_bytes is not None:
        sig = (internals_name, len(internals_bytes), model_mode, selected_backend)
        if render_now or st.session_state.get("internals_sig") != sig:
            with st.spinner("Building model internals view..."):
                try:
                    baseline_only_backend = selected_backend in {"artifact", "pi-lightgbm"}
                    effective_mode = "Baseline" if model_mode == COMPARE_MODE or baseline_only_backend else model_mode
                    st.session_state["internals_result"] = service.infer(
                        internals_bytes,
                        model_mode=effective_mode,
                        model_backend=selected_backend,
                    )
                    st.session_state["internals_sig"] = sig
                    st.session_state["internals_mode"] = effective_mode
                    st.session_state["internals_backend"] = selected_backend
                except Exception as exc:
                    st.error(f"Could not build internals view: {exc}")

    if "internals_result" in st.session_state:
        internals_result = st.session_state["internals_result"]
        engine_ids = internals_result.get("engine_ids", [])
        if engine_ids:
            ec1, ec2, ec3, ec4 = st.columns(4)
            selected_engine = ec1.selectbox("Engine", engine_ids, format_func=lambda x: f"U-{int(x):03d}", key="internals_engine")
            rel = _lookup_engine_map(internals_result.get("per_engine_reliability", {}), int(selected_engine), {})
            pred = float(_lookup_engine_map(internals_result.get("per_engine_mean_rul", {}), int(selected_engine), 0.0) or 0.0)
            ec2.metric("Predicted RUL", f"{pred:.1f}")
            ec3.metric("Reliability Index", f"{float(rel.get('ri', 0.0)):.2f}")
            ec4.markdown(f"**Gate**<br>{_decision_badge(str(rel.get('decision', 'RAW')))}", unsafe_allow_html=True)

            payload = _build_model_core_payload(
                internals_result,
                int(selected_engine),
                model_backend=st.session_state.get("internals_backend", selected_backend),
                model_mode=st.session_state.get("internals_mode", model_mode),
                view_mode=view_mode,
            )
            _render_model_core_component(payload, height=visual_height)

            with st.expander("What The Layers Mean"):
                st.markdown(
                    """
                    - **Sensor shell:** active C-MAPSS sensors grouped by thermal, pressure, mechanical, and flow behavior.
                    - **Data flow particles:** normalized sensor windows moving toward inference.
                    - **Attention arcs:** recent 30-cycle timestep focus from the attention backend when available.
                    - **Model core:** selected backend and current RUL estimate.
                    - **Gate ring:** post-prediction Reliability Index, colored by `ACCEPT`, `WARN`, or `REJECT`.
                    - **Future fan:** probabilistic RUL trajectory generated by the cVAE/Monte Carlo trajectory module.
                    """
                )


# ═══════════════════════════════════════════════════════════════
# Page: Batch Inference (detailed)
# ═══════════════════════════════════════════════════════════════
elif page == "🔬 Batch Inference":
    st.markdown('<p class="section-header">Batch Inference & Analytics</p>', unsafe_allow_html=True)

    batch_file = st.file_uploader("Upload sensor CSV", type=["csv", "txt"], key="batch_upload")
    if batch_file:
        batch_bytes = batch_file.getvalue()
        preview = pd.read_csv(io.BytesIO(batch_bytes), nrows=10)
        st.dataframe(preview, use_container_width=True)

        if st.button("🚀 Run Inference", type="primary"):
            with st.spinner("Processing..."):
                try:
                    result = service.infer(batch_bytes, model_mode=model_mode, model_backend=selected_backend)
                    st.session_state["batch_result"] = result
                except Exception as exc:
                    st.error(f"Error: {exc}")

    if "batch_result" in st.session_state:
        result = st.session_state["batch_result"]
        rel_df = build_reliability_df(result)

        st.markdown('<p class="section-header">Reliability Gating Table</p>', unsafe_allow_html=True)
        st.dataframe(rel_df, use_container_width=True)

        # Ground truth evaluation
        st.markdown('<p class="section-header">Ground Truth Evaluation (Optional)</p>', unsafe_allow_html=True)
        gt_file = st.file_uploader("Upload ground truth CSV (engine_id, true_rul)", type=["csv"], key="gt_eval")
        if gt_file:
            try:
                gt_df = pd.read_csv(gt_file)
                if {"engine_id", "true_rul"}.issubset(set(gt_df.columns)):
                    export_df = rel_df[["engine_id", "raw_pred_rul", "ri", "decision", "trusted_rul"]].rename(columns={"raw_pred_rul": "predicted_rul"})
                    eval_df = export_df.merge(gt_df[["engine_id", "true_rul"]], on="engine_id", how="inner")
                    if not eval_df.empty:
                        eval_df["abs_error"] = (eval_df["predicted_rul"] - eval_df["true_rul"]).abs()
                        metrics = evaluate_reliability_log(eval_df, catastrophic_error_threshold=20.0)
                        e1, e2, e3, e4 = st.columns(4)
                        e1.metric("MAE", f"{metrics['mean_abs_error']:.2f}")
                        e2.metric("RI-Error Corr", f"{metrics['ri_error_correlation']:.3f}")
                        e3.metric("Catastrophic Rate", f"{metrics['catastrophic_rate']:.2%}")
                        e4.metric("Accept Rate", f"{metrics.get('accept_rate', 0):.2%}")
            except Exception as exc:
                st.error(f"Evaluation error: {exc}")

        # Sensor correlation & distribution
        raw_df = result["raw_df"]
        if isinstance(raw_df, list):
            raw_df = pd.DataFrame(raw_df)

        st.markdown('<p class="section-header">Sensor Analytics</p>', unsafe_allow_html=True)
        sensor_cols = [c for c in RAW_COLUMN_NAMES if c.startswith("s_")]

        ac1, ac2 = st.columns(2)
        with ac1:
            st.caption("Correlation Matrix (first 12 sensors)")
            corr_cols = [c for c in sensor_cols if c in raw_df.columns][:12]
            corr = raw_df[corr_cols].corr()
            if px is not None:
                fig_corr = px.imshow(corr, color_continuous_scale="RdBu_r", title="Sensor Correlation")
                _make_plotly_dark(fig_corr)
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.dataframe(corr.round(3), use_container_width=True)

        with ac2:
            hist_sensor = st.selectbox("Sensor distribution", sensor_cols, index=1)
            if px is not None:
                fig_h = px.histogram(raw_df, x=hist_sensor, nbins=20, color_discrete_sequence=["#8b5cf6"], title=f"{hist_sensor} Distribution")
                _make_plotly_dark(fig_h)
                st.plotly_chart(fig_h, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# Page: Settings
# ═══════════════════════════════════════════════════════════════
elif page == "⚙️ Settings":
    st.markdown('<p class="section-header">System Settings</p>', unsafe_allow_html=True)

    st.markdown("#### Available Backends")
    for b in backends:
        st.markdown(f"- **{b['name']}**: {b['description']}")

    st.markdown("#### Validation Commands")
    st.code("python scripts\\run_validation_suite.py", language="powershell")
    st.code("python -m pytest tests\\test_inference_regression.py -q", language="powershell")

    st.markdown("#### MQTT Pipeline")
    st.code("python ingestion\\mqtt_secure_ingest.py --broker 127.0.0.1 --port 1883 --topic engines/fd001/raw --model-mode Baseline --insecure-no-tls", language="powershell")
    st.code("python ingestion\\digital_twin_streamer.py --broker 127.0.0.1 --port 1883 --topic engines/fd001/raw --interval-sec 0.5 --cycles 3000", language="powershell")

    st.markdown("#### Headless Inference")
    st.code("python scripts\\run_headless_inference.py --csv examples\\sample_cmapss_engine.csv --mode Baseline --backend attention", language="powershell")
