from __future__ import annotations

import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

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

PLOTLY_TEMPLATE = "plotly_dark"


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
        ["🏠 Fleet Overview", "📡 Live Digital Twin", "📈 RUL Trajectories", "🔬 Batch Inference", "⚙️ Settings"],
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
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Fleet Engines", len(predictions), help="Total engines in uploaded data")
        m2.metric("Mean RUL", f"{overall_rul:.1f}", help="Fleet average remaining useful life (cycles)")
        m3.metric("Reliability Index", f"{overall_ri:.2f}", help="Fleet mean RI (0-1)")

        # Count decisions
        rel_df = build_reliability_df(result)
        decision_counts = rel_df["decision"].value_counts().to_dict()
        m4.metric("Accept Gate", f"{decision_counts.get('ACCEPT', 0)}", help=f"RI ≥ 0.75")
        m5.metric("Warn / Reject", f"{decision_counts.get('WARN', 0)} / {decision_counts.get('REJECT', 0)}")

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

            d1, d2, d3 = st.columns(3)
            d1.metric("Predicted RUL", f"{float(predictions[sel_engine]):.1f}")
            d2.metric("Reliability Index", f"{float(sel_rel['ri']):.2f}")
            d3.markdown(f"**Gate Decision**<br>{_decision_badge(sel_rel['decision'])}", unsafe_allow_html=True)

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

    lm = st.columns(6)
    lm[0].metric("Engine", str(live_state.get("engine_id", "-")) if live_state else "-")
    lm[1].metric("Cycles", int(live_state.get("received_cycles_for_engine", 0)) if live_state else 0)
    
    if is_buffering and not has_pred:
        lm[2].metric("Predicted RUL", "Wait...", delta=f"{cycles_left} cycles left", delta_color="off")
        lm[3].metric("Trusted RUL", "Wait...")
        lm[4].metric("RI", "Wait...")
        lm[5].metric("Decision", "BUFFERING")
    else:
        lm[2].metric("Predicted RUL", f"{float(latest.get('predicted_rul', 0)):.1f}")
        lm[3].metric("Trusted RUL", f"{float(latest.get('trusted_rul', 0)):.1f}")
        lm[4].metric("RI", f"{float(latest.get('ri', 0)):.3f}")
        lm[5].metric("Decision", str(latest.get("decision", "-")))

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
