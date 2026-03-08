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
except ImportError:  # pragma: no cover - optional visualization dependency.
    px = None

from inference.attention_model import (
    RAW_COLUMN_NAMES,
    get_model_weights_path,
    load_attention_model,
    maintenance_status,
    predict_rul_detailed_from_csv,
    simulate_realtime_engine_from_df,
)
from inference.reliability import evaluate_reliability_log


st.set_page_config(page_title="Aircraft Engine RUL Predictor", layout="wide")
st.title("Aircraft Engine RUL Predictor")
st.write("Upload a CSV file containing C-MAPSS style engine sensor readings.")
COMPARE_MODE = "Compare (Baseline vs PI)"
DECISION_COLORS = {"ACCEPT": "#2ca02c", "WARN": "#ff7f0e", "REJECT": "#d62728", "RAW": "#1f77b4"}
LIVE_STATE_FILE = Path("logs") / "live_state.json"
LIVE_PREDICTIONS_FILE = Path("logs") / "mqtt_predictions.csv"


def _show_plotly_hint_once() -> None:
    if px is None:
        st.info("Install `plotly` for colorful interactive charts: `pip install plotly`")


def _render_decision_distribution(df: pd.DataFrame, column: str = "decision") -> None:
    counts = df[column].value_counts().rename_axis(column).to_frame("count").reset_index()
    if counts.empty:
        return
    if px is not None:
        fig = px.pie(
            counts,
            names=column,
            values="count",
            color=column,
            color_discrete_map=DECISION_COLORS,
            hole=0.45,
            title="Decision Distribution",
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(counts.set_index(column)["count"])


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


@st.cache_resource
def get_model(model_mode: str):
    return load_attention_model(get_model_weights_path(model_mode))


def build_reliability_df(result: dict) -> pd.DataFrame:
    reliability_rows = []
    for engine_id in result["engine_ids"]:
        rel = result["per_engine_reliability"][engine_id]
        reliability_rows.append(
            {
                "engine_id": engine_id,
                "ri": rel["ri"],
                "decision": rel["decision"],
                "trusted_rul": rel["trusted_rul"],
                "raw_pred_rul": result["per_engine_mean_rul"][engine_id],
                "window_std": rel["window_std"],
                "monotonic_violation_rate": rel["monotonic_violation_rate"],
                "smoothness_ratio": rel["smoothness_ratio"],
                "reason_codes": ", ".join(rel["reason_codes"]),
            }
        )
    return pd.DataFrame(reliability_rows)


st.subheader("Live Digital Twin Feed (MQTT)")
with st.expander("Start MQTT Digital Twin Pipeline", expanded=False):
    st.code(
        "python ingestion\\mqtt_secure_ingest.py --broker 127.0.0.1 --port 1883 --topic engines/fd001/raw --model-mode Baseline --insecure-no-tls",
        language="powershell",
    )
    st.code(
        "python ingestion\\digital_twin_streamer.py --broker 127.0.0.1 --port 1883 --topic engines/fd001/raw --interval-sec 0.5 --cycles 3000",
        language="powershell",
    )

live_a, live_b, live_c = st.columns([1, 1, 2])
refresh_now = live_a.button("Refresh Live Feed")
auto_refresh = live_b.checkbox("Auto-refresh", value=False, key="mqtt_auto_refresh")
refresh_sec = int(live_c.slider("Refresh interval (sec)", min_value=1, max_value=10, value=2))

live_state = _read_live_state()
live_pred_df = _read_live_predictions()
latest_pred = (live_state.get("latest_prediction") or {}) if live_state else {}
if not latest_pred and not live_pred_df.empty:
    latest_pred = live_pred_df.iloc[-1].to_dict()

live_metrics = st.columns(6)
live_metrics[0].metric("Engine", str(live_state.get("engine_id", "-")) if live_state else "-")
live_metrics[1].metric("Cycles", int(live_state.get("received_cycles_for_engine", 0)) if live_state else 0)
live_metrics[2].metric("Predicted RUL", f"{float(latest_pred.get('predicted_rul', 0.0)):.2f}")
live_metrics[3].metric("Trusted RUL", f"{float(latest_pred.get('trusted_rul', 0.0)):.2f}")
live_metrics[4].metric("RI", f"{float(latest_pred.get('ri', 0.0)):.3f}")
live_metrics[5].metric("Decision", str(latest_pred.get("decision", "-")))

if live_state:
    st.caption(f"Last update: {live_state.get('updated_at_utc', '-')}")

if not live_pred_df.empty:
    if "timestamp_utc" in live_pred_df.columns and px is not None:
        live_plot_df = live_pred_df.copy()
        fig_live = px.line(
            live_plot_df,
            x="timestamp_utc",
            y=[c for c in ["predicted_rul", "trusted_rul", "ri"] if c in live_plot_df.columns],
            title="Live RUL / RI Trajectory",
        )
        st.plotly_chart(fig_live, use_container_width=True)
    else:
        cols = [c for c in ["predicted_rul", "trusted_rul", "ri"] if c in live_pred_df.columns]
        if cols:
            st.line_chart(live_pred_df[cols])
    if "decision" in live_pred_df.columns:
        _render_decision_distribution(live_pred_df, column="decision")
    with st.expander("Recent Live Rows", expanded=False):
        st.dataframe(live_pred_df.tail(40), use_container_width=True)
else:
    st.info("No live MQTT predictions yet. Start ingest + digital twin and refresh.")

if auto_refresh and not refresh_now:
    time.sleep(max(refresh_sec, 1))
    st.rerun()


uploaded_file = st.file_uploader("Upload sensor CSV", type=["csv", "txt"])
model_mode = st.selectbox("Model Mode", options=["Baseline", "Physics-Informed", COMPARE_MODE], index=0)
scenario_dir = Path("examples") / "scenarios"
with st.expander("Demo Scenarios", expanded=False):
    st.caption("Curated replay scenario files for quick testing.")
    scenario_map = {
        "Stable behavior": scenario_dir / "scenario_stable_behavior.csv",
        "Noisy behavior": scenario_dir / "scenario_noisy_behavior.csv",
        "Rapid degradation": scenario_dir / "scenario_rapid_degradation.csv",
    }
    for label, path in scenario_map.items():
        if path.exists():
            st.download_button(
                f"Download {label}",
                data=path.read_bytes(),
                file_name=path.name,
                mime="text/csv",
                key=f"dl_{path.name}",
            )
        else:
            st.write(f"{label}: `{path}` not found yet.")

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_signature = (uploaded_file.name, len(file_bytes), model_mode)
    if st.session_state.get("active_file_signature") != file_signature:
        st.session_state["active_file_signature"] = file_signature
        st.session_state["active_file_bytes"] = file_bytes
        st.session_state["active_model_mode"] = model_mode
        st.session_state.pop("inference_result", None)

    try:
        preview_df = pd.read_csv(io.BytesIO(file_bytes), nrows=10)
    except Exception:
        try:
            preview_df = pd.read_csv(io.BytesIO(file_bytes), sep=r"\s+", engine="python", nrows=10)
        except Exception as exc:
            st.error(f"Could not parse uploaded file for preview: {exc}")
            st.stop()

    st.subheader("Uploaded Data Preview")
    st.dataframe(preview_df)

    if st.button("Run RUL Inference"):
        with st.spinner("Running attention model inference..."):
            try:
                payload = io.BytesIO(st.session_state["active_file_bytes"])
                if model_mode == COMPARE_MODE:
                    baseline_model = get_model("Baseline")
                    pi_model = get_model("Physics-Informed")
                    baseline_result = predict_rul_detailed_from_csv(payload, model=baseline_model)
                    payload.seek(0)
                    pi_result = predict_rul_detailed_from_csv(payload, model=pi_model)
                    st.session_state["compare_result"] = {
                        "baseline": baseline_result,
                        "pi": pi_result,
                    }
                    st.session_state.pop("inference_result", None)
                else:
                    model = get_model(model_mode)
                    result = predict_rul_detailed_from_csv(payload, model=model)
                    st.session_state["inference_result"] = result
                    st.session_state.pop("compare_result", None)
            except FileNotFoundError as exc:
                st.error(str(exc))
                st.stop()
            except ValueError as exc:
                st.error(f"Input format error: {exc}")
                st.info(
                    "Expected data must include either named C-MAPSS columns "
                    "(unit_nr, time_cycles, op_setting_1..3, s_1..s_21) or at least 26 raw columns."
                )
                st.stop()
            except Exception as exc:
                st.error(f"Unexpected error during inference: {exc}")
                st.stop()

    if st.session_state.get("active_model_mode") == COMPARE_MODE and "compare_result" in st.session_state:
        compare_result = st.session_state["compare_result"]
        baseline_result = compare_result["baseline"]
        pi_result = compare_result["pi"]
        engine_ids = sorted(set(baseline_result["engine_ids"]) & set(pi_result["engine_ids"]))

        rows = []
        for engine_id in engine_ids:
            b_rel = baseline_result["per_engine_reliability"][engine_id]
            p_rel = pi_result["per_engine_reliability"][engine_id]
            b_pred = float(baseline_result["per_engine_mean_rul"][engine_id])
            p_pred = float(pi_result["per_engine_mean_rul"][engine_id])
            b_ri = float(b_rel["ri"])
            p_ri = float(p_rel["ri"])
            rows.append(
                {
                    "engine_id": int(engine_id),
                    "baseline_pred_rul": b_pred,
                    "pi_pred_rul": p_pred,
                    "pred_rul_delta_pi_minus_baseline": p_pred - b_pred,
                    "baseline_ri": b_ri,
                    "pi_ri": p_ri,
                    "ri_delta_pi_minus_baseline": p_ri - b_ri,
                    "baseline_decision": b_rel["decision"],
                    "pi_decision": p_rel["decision"],
                    "decision_delta": "CHANGED" if b_rel["decision"] != p_rel["decision"] else "SAME",
                }
            )
        compare_df = pd.DataFrame(rows)

        baseline_overall = float(baseline_result["overall_mean_rul"])
        pi_overall = float(pi_result["overall_mean_rul"])
        baseline_ri = float(baseline_result["overall_reliability_index"])
        pi_ri = float(pi_result["overall_reliability_index"])
        decision_change_rate = (
            float((compare_df["decision_delta"] == "CHANGED").mean()) if not compare_df.empty else 0.0
        )

        st.subheader("Side-by-Side Baseline vs PI")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Baseline Overall RUL", f"{baseline_overall:.2f}")
        c2.metric("PI Overall RUL", f"{pi_overall:.2f}", delta=f"{pi_overall - baseline_overall:.2f}")
        c3.metric("Baseline Overall RI", f"{baseline_ri:.2f}")
        c4.metric("PI Overall RI", f"{pi_ri:.2f}", delta=f"{pi_ri - baseline_ri:.2f}")
        st.caption(f"Decision change rate across engines: {decision_change_rate:.2%}")

        if not compare_df.empty:
            st.dataframe(compare_df, use_container_width=True)
            st.download_button(
                "Download Baseline vs PI Comparison (CSV)",
                data=compare_df.to_csv(index=False).encode("utf-8"),
                file_name="baseline_vs_pi_comparison.csv",
                mime="text/csv",
            )
            if px is not None:
                fig_compare = px.bar(
                    compare_df.melt(
                        id_vars=["engine_id"],
                        value_vars=["baseline_pred_rul", "pi_pred_rul"],
                        var_name="mode",
                        value_name="predicted_rul",
                    ),
                    x="engine_id",
                    y="predicted_rul",
                    color="mode",
                    barmode="group",
                    title="Baseline vs PI Predicted RUL",
                    color_discrete_sequence=["#4c78a8", "#f58518"],
                )
                st.plotly_chart(fig_compare, use_container_width=True)
                fig_delta = px.scatter(
                    compare_df,
                    x="ri_delta_pi_minus_baseline",
                    y="pred_rul_delta_pi_minus_baseline",
                    color="decision_delta",
                    title="PI-Baseline Delta Map",
                    color_discrete_map={"SAME": "#1f77b4", "CHANGED": "#d62728"},
                )
                st.plotly_chart(fig_delta, use_container_width=True)
            else:
                _show_plotly_hint_once()
                st.bar_chart(compare_df.set_index("engine_id")[["baseline_pred_rul", "pi_pred_rul"]])
                st.bar_chart(compare_df.set_index("engine_id")[["pred_rul_delta_pi_minus_baseline"]])
            _render_decision_distribution(compare_df.rename(columns={"decision_delta": "decision"}), column="decision")

        compare_tab_baseline, compare_tab_pi = st.tabs(["Baseline Details", "PI Details"])
        with compare_tab_baseline:
            baseline_reliability_df = build_reliability_df(baseline_result)
            st.dataframe(baseline_reliability_df, use_container_width=True)
        with compare_tab_pi:
            pi_reliability_df = build_reliability_df(pi_result)
            st.dataframe(pi_reliability_df, use_container_width=True)

    if "inference_result" in st.session_state:
        result = st.session_state["inference_result"]
        active_mode = st.session_state.get("active_model_mode", model_mode)
        model = get_model(active_mode)
        predictions = result["per_engine_mean_rul"]
        overall_rul = float(result["overall_mean_rul"])
        overall_ri = float(result["overall_reliability_index"])
        raw_df = result["raw_df"]

        st.subheader("Predicted Remaining Useful Life (RUL)")
        metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)
        metric_col_1.metric("Overall Predicted RUL", f"{overall_rul:.2f}")
        metric_col_2.metric("Fleet Engines", len(predictions))
        metric_col_3.metric("Status", maintenance_status(overall_rul))
        metric_col_4.metric("Reliability Index (RI)", f"{overall_ri:.2f}")
        st.caption(f"Active model mode: {st.session_state.get('active_model_mode', model_mode)}")

        normalized = max(0.0, min(1.0, overall_rul / 125.0))
        st.progress(normalized, text="RUL score normalized to 0-125")

        overview_tab, reliability_tab, insights_tab = st.tabs(
            ["Prediction Overview", "Reliability Gating", "Advanced Insights"]
        )

        with overview_tab:
            result_df = pd.DataFrame(
                [{"engine_id": int(engine_id), "predicted_rul": float(rul)} for engine_id, rul in predictions.items()]
            )
            st.subheader("Per Engine Prediction")
            st.dataframe(result_df, use_container_width=True)
            if len(result_df) > 1:
                if px is not None:
                    fig = px.bar(
                        result_df.sort_values("predicted_rul", ascending=False),
                        x="engine_id",
                        y="predicted_rul",
                        color="predicted_rul",
                        color_continuous_scale="Turbo",
                        title="Per Engine RUL (Color Encoded)",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    box_fig = px.box(result_df, y="predicted_rul", points="all", title="Fleet RUL Spread")
                    st.plotly_chart(box_fig, use_container_width=True)
                else:
                    _show_plotly_hint_once()
                    st.bar_chart(result_df.set_index("engine_id")["predicted_rul"])

        with reliability_tab:
            reliability_df = build_reliability_df(result)
            export_df = reliability_df[["engine_id", "raw_pred_rul", "ri", "decision", "trusted_rul"]].rename(
                columns={"raw_pred_rul": "predicted_rul"}
            )
            st.subheader("Reliability-Aware Gating Summary")
            st.dataframe(reliability_df, use_container_width=True)
            st.download_button(
                "Download Prediction Reliability Log (CSV)",
                data=export_df.to_csv(index=False).encode("utf-8"),
                file_name="prediction_reliability_log.csv",
                mime="text/csv",
            )

            _render_decision_distribution(reliability_df, column="decision")
            if px is not None:
                scatter = px.scatter(
                    reliability_df,
                    x="ri",
                    y="raw_pred_rul",
                    color="decision",
                    size="window_std",
                    hover_data=["engine_id", "trusted_rul"],
                    color_discrete_map=DECISION_COLORS,
                    title="Reliability vs Raw Prediction",
                )
                st.plotly_chart(scatter, use_container_width=True)

            st.subheader("Reliability Evaluation (Optional Ground Truth)")
            gt_file = st.file_uploader(
                "Upload ground truth CSV with columns: engine_id,true_rul",
                type=["csv"],
                key="gt_eval_file",
            )
            if gt_file is not None:
                try:
                    gt_df = pd.read_csv(gt_file)
                    required_gt = {"engine_id", "true_rul"}
                    if not required_gt.issubset(set(gt_df.columns)):
                        st.error("Ground truth CSV must contain columns: engine_id,true_rul")
                    else:
                        eval_df = export_df.merge(gt_df[["engine_id", "true_rul"]], on="engine_id", how="inner")
                        if eval_df.empty:
                            st.warning("No overlapping engine_id values between predictions and ground truth.")
                        else:
                            eval_df["abs_error"] = (eval_df["predicted_rul"] - eval_df["true_rul"]).abs()
                            eval_metrics = evaluate_reliability_log(eval_df, catastrophic_error_threshold=20.0)
                            e1, e2, e3, e4 = st.columns(4)
                            e1.metric("Mean Absolute Error", f"{eval_metrics['mean_abs_error']:.2f}")
                            e2.metric("RI-Error Correlation", f"{eval_metrics['ri_error_correlation']:.3f}")
                            e3.metric("Catastrophic Error Rate", f"{eval_metrics['catastrophic_rate']:.2%}")
                            e4.metric("Accept Rate", f"{eval_metrics.get('accept_rate', 0.0):.2%}")

                            if px is not None:
                                eval_plot = px.scatter(
                                    eval_df,
                                    x="ri",
                                    y="abs_error",
                                    color="decision" if "decision" in eval_df.columns else None,
                                    color_discrete_map=DECISION_COLORS,
                                    title="RI vs Absolute Error",
                                )
                                st.plotly_chart(eval_plot, use_container_width=True)
                            else:
                                st.scatter_chart(eval_df.set_index("ri")[["abs_error"]])
                            st.dataframe(eval_df, use_container_width=True)
                except Exception as exc:
                    st.error(f"Could not evaluate reliability metrics: {exc}")

            st.subheader("Streaming Replay (Cycle-by-Cycle)")
            stream_engine = st.selectbox(
                "Engine for streaming replay",
                options=result["engine_ids"],
                index=0,
                key="stream_engine",
            )
            stream_step = st.slider("Replay step (cycles)", min_value=1, max_value=5, value=1)
            if st.button("Run Streaming Replay"):
                with st.spinner("Simulating real-time RUL updates..."):
                    stream_df = simulate_realtime_engine_from_df(
                        raw_df=raw_df,
                        engine_id=int(stream_engine),
                        model=model,
                        step=int(stream_step),
                    )
                st.dataframe(stream_df, use_container_width=True)
                if px is not None:
                    stream_long = stream_df.melt(
                        id_vars=["time_cycles"],
                        value_vars=["predicted_rul", "trusted_rul"],
                        var_name="series",
                        value_name="rul",
                    )
                    stream_fig = px.line(
                        stream_long,
                        x="time_cycles",
                        y="rul",
                        color="series",
                        title="Streaming Replay: Raw vs Trusted RUL",
                        color_discrete_sequence=["#17becf", "#bc5090"],
                    )
                    st.plotly_chart(stream_fig, use_container_width=True)
                    ri_fig = px.line(
                        stream_df,
                        x="time_cycles",
                        y="reliability_index",
                        color="decision",
                        title="Streaming Replay: Reliability Trajectory",
                        color_discrete_map=DECISION_COLORS,
                    )
                    st.plotly_chart(ri_fig, use_container_width=True)
                else:
                    st.line_chart(stream_df.set_index("time_cycles")[["predicted_rul", "trusted_rul"]])
                    st.line_chart(stream_df.set_index("time_cycles")[["reliability_index"]])

        with insights_tab:
            with st.expander("Advanced Insights", expanded=True):
                dq_col_1, dq_col_2, dq_col_3, dq_col_4 = st.columns(4)
                dq_col_1.metric("Rows", len(raw_df))
                dq_col_2.metric("Missing Values", int(raw_df.isna().sum().sum()))
                dq_col_3.metric("Duplicate Rows", int(raw_df.duplicated().sum()))
                dq_col_4.metric("Unique Engines", int(raw_df["unit_nr"].nunique()))

                cycle_check = (
                    raw_df.groupby("unit_nr")["time_cycles"]
                    .agg(["min", "max", "count"])
                    .rename(columns={"count": "observed_rows"})
                )
                cycle_check["expected_rows"] = cycle_check["max"] - cycle_check["min"] + 1
                cycle_check["missing_cycle_rows"] = cycle_check["expected_rows"] - cycle_check["observed_rows"]
                st.subheader("Cycle Continuity Check")
                st.dataframe(cycle_check, use_container_width=True)

                selected_engine = st.selectbox(
                    "Select engine for detailed visualizations",
                    options=result["engine_ids"],
                    index=0,
                )

                selected_window_preds = result["per_engine_window_rul"][selected_engine]
                selected_rel = result["per_engine_reliability"][selected_engine]
                window_df = pd.DataFrame(
                    {
                        "window_index": np.arange(1, len(selected_window_preds) + 1),
                        "predicted_rul": selected_window_preds,
                    }
                )

                st.subheader(f"Engine {selected_engine}: Window-Level RUL")
                if px is not None:
                    window_fig = px.area(
                        window_df,
                        x="window_index",
                        y="predicted_rul",
                        title=f"Engine {selected_engine}: Window-Level RUL",
                        color_discrete_sequence=["#00a896"],
                    )
                    st.plotly_chart(window_fig, use_container_width=True)
                else:
                    st.line_chart(window_df.set_index("window_index")["predicted_rul"])
                st.caption(
                    f"Uncertainty summary - mean: {np.mean(selected_window_preds):.2f}, "
                    f"min: {np.min(selected_window_preds):.2f}, max: {np.max(selected_window_preds):.2f}, "
                    f"std: {np.std(selected_window_preds):.2f}"
                )
                st.caption(
                    f"Reliability decision: {selected_rel['decision']} | "
                    f"RI={selected_rel['ri']:.2f} | "
                    f"Reasons={', '.join(selected_rel['reason_codes'])}"
                )

                attention = result["per_engine_last_attention"][selected_engine]
                attention_df = pd.DataFrame(
                    {"time_step": np.arange(1, len(attention) + 1), "attention_weight": attention}
                )
                st.subheader(f"Engine {selected_engine}: Attention Across Last {len(attention)} Timesteps")
                if px is not None:
                    attention_fig = px.bar(
                        attention_df,
                        x="time_step",
                        y="attention_weight",
                        color="attention_weight",
                        color_continuous_scale="Sunset",
                        title=f"Engine {selected_engine}: Attention Weights",
                    )
                    st.plotly_chart(attention_fig, use_container_width=True)
                else:
                    st.area_chart(attention_df.set_index("time_step")["attention_weight"])

                engine_df = raw_df[raw_df["unit_nr"] == selected_engine].sort_values("time_cycles")
                sensor_columns = [col for col in RAW_COLUMN_NAMES if col.startswith("s_")]

                st.subheader(f"Engine {selected_engine}: Sensor Trends")
                selected_sensors = st.multiselect(
                    "Sensors to visualize",
                    options=sensor_columns,
                    default=["s_2", "s_3", "s_4", "s_7"],
                )
                if selected_sensors:
                    trend_df = engine_df[["time_cycles"] + selected_sensors].set_index("time_cycles")
                    if px is not None:
                        trend_long = trend_df.reset_index().melt(
                            id_vars=["time_cycles"], var_name="sensor", value_name="reading"
                        )
                        trend_fig = px.line(
                            trend_long,
                            x="time_cycles",
                            y="reading",
                            color="sensor",
                            title=f"Engine {selected_engine}: Sensor Trends",
                        )
                        st.plotly_chart(trend_fig, use_container_width=True)
                    else:
                        st.line_chart(trend_df)

                st.subheader("Sensor Correlation Matrix (EDA)")
                corr_cols = [col for col in sensor_columns if col in engine_df.columns][:12]
                corr = engine_df[corr_cols].corr()
                try:
                    import matplotlib  # noqa: F401

                    st.dataframe(corr.style.background_gradient(cmap="coolwarm"), use_container_width=True)
                except ImportError:
                    st.info("Install `matplotlib` to enable colored correlation styling.")
                    st.dataframe(corr.round(3), use_container_width=True)

                st.subheader("Sensor Distribution Snapshot (EDA)")
                hist_sensor = st.selectbox("Sensor for distribution", options=sensor_columns, index=1)
                hist_counts, hist_edges = np.histogram(engine_df[hist_sensor].values, bins=12)
                hist_df = pd.DataFrame(
                    {
                        "bin_center": (hist_edges[:-1] + hist_edges[1:]) / 2,
                        "count": hist_counts,
                    }
                )
                if px is not None:
                    hist_fig = px.bar(
                        hist_df,
                        x="bin_center",
                        y="count",
                        color="count",
                        color_continuous_scale="Tealrose",
                        title=f"{hist_sensor} Distribution",
                    )
                    st.plotly_chart(hist_fig, use_container_width=True)
                else:
                    st.bar_chart(hist_df.set_index("bin_center")["count"])
