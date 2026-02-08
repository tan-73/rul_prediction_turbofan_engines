from __future__ import annotations

import io

import numpy as np
import pandas as pd
import streamlit as st

from inference.attention_model import (
    RAW_COLUMN_NAMES,
    load_attention_model,
    maintenance_status,
    predict_rul_detailed_from_csv,
)


st.set_page_config(page_title="Aircraft Engine RUL Predictor", layout="wide")
st.title("Aircraft Engine RUL Predictor")
st.write("Upload a CSV file containing C-MAPSS style engine sensor readings.")


@st.cache_resource
def get_model():
    return load_attention_model()


uploaded_file = st.file_uploader("Upload sensor CSV", type=["csv", "txt"])

if uploaded_file is not None:
    try:
        preview_bytes = uploaded_file.getvalue()
        preview_df = pd.read_csv(io.BytesIO(preview_bytes), nrows=10)
    except Exception:
        try:
            preview_df = pd.read_csv(io.BytesIO(uploaded_file.getvalue()), sep=r"\s+", engine="python", nrows=10)
        except Exception as exc:
            st.error(f"Could not parse uploaded file for preview: {exc}")
            st.stop()

    st.subheader("Uploaded Data Preview")
    st.dataframe(preview_df)

    if st.button("Run RUL Inference"):
        with st.spinner("Running attention model inference..."):
            try:
                model = get_model()
                result = predict_rul_detailed_from_csv(io.BytesIO(uploaded_file.getvalue()), model=model)
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

        predictions = result["per_engine_mean_rul"]
        overall_rul = float(result["overall_mean_rul"])
        raw_df = result["raw_df"]

        st.subheader("Predicted Remaining Useful Life (RUL)")
        metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
        metric_col_1.metric("Overall Predicted RUL", f"{overall_rul:.2f}")
        metric_col_2.metric("Fleet Engines", len(predictions))
        metric_col_3.metric("Status", maintenance_status(overall_rul))

        normalized = max(0.0, min(1.0, overall_rul / 125.0))
        st.progress(normalized, text="RUL score normalized to 0-125")

        overview_tab, insights_tab = st.tabs(["Prediction Overview", "Advanced Insights"])

        with overview_tab:
            result_df = pd.DataFrame(
                [{"engine_id": int(engine_id), "predicted_rul": float(rul)} for engine_id, rul in predictions.items()]
            )
            st.subheader("Per Engine Prediction")
            st.dataframe(result_df, use_container_width=True)
            if len(result_df) > 1:
                st.bar_chart(result_df.set_index("engine_id")["predicted_rul"])

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
                window_df = pd.DataFrame(
                    {
                        "window_index": np.arange(1, len(selected_window_preds) + 1),
                        "predicted_rul": selected_window_preds,
                    }
                )

                st.subheader(f"Engine {selected_engine}: Window-Level RUL")
                st.line_chart(window_df.set_index("window_index")["predicted_rul"])
                st.caption(
                    f"Uncertainty summary — mean: {np.mean(selected_window_preds):.2f}, "
                    f"min: {np.min(selected_window_preds):.2f}, max: {np.max(selected_window_preds):.2f}, "
                    f"std: {np.std(selected_window_preds):.2f}"
                )

                attention = result["per_engine_last_attention"][selected_engine]
                attention_df = pd.DataFrame(
                    {"time_step": np.arange(1, len(attention) + 1), "attention_weight": attention}
                )
                st.subheader(f"Engine {selected_engine}: Attention Across Last {len(attention)} Timesteps")
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
                st.bar_chart(hist_df.set_index("bin_center")["count"])
