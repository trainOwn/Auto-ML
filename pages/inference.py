import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="keras")

import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import plotly.express as px

from pages.styles import (
    inject_css, render_page_title, render_step, render_divider,
    render_metrics_row, render_result_banner, render_sidebar,
    PLOTLY_TEMPLATE, PLOTLY_COLORS,
)

inject_css()
render_sidebar()

SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved_models")

render_page_title("Inference", "Load a trained model and make predictions on new data")

# ── Step 1: Select model ─────────────────────────────
render_step(1, "Select Trained Model")

if not os.path.isdir(SAVE_DIR) or not os.listdir(SAVE_DIR):
    st.warning("No saved models found. Train a model first on the **Train Model** page.")
    st.stop()

model_dirs = sorted(
    [d for d in os.listdir(SAVE_DIR) if os.path.isdir(os.path.join(SAVE_DIR, d))],
    reverse=True,
)
if not model_dirs:
    st.warning("No saved models found.")
    st.stop()

selected_model = st.selectbox("Saved model", model_dirs)
model_path = os.path.join(SAVE_DIR, selected_model)
meta_path = os.path.join(model_path, "metadata.json")

if not os.path.exists(meta_path):
    st.error("metadata.json not found.")
    st.stop()

with open(meta_path) as f:
    metadata = json.load(f)

is_lstm = metadata.get("is_lstm", False)
is_binary = metadata.get("is_binary", False)
n_classes = metadata.get("n_classes", 0)
problem_type = metadata["problem_type"]

# Model info card
st.markdown("""<div class="info-card">
<h4>Model Details</h4>
</div>""", unsafe_allow_html=True)

ic1, ic2, ic3, ic4 = st.columns(4)
ic1.metric("Algorithm", metadata["algorithm"])
if problem_type == "Classification":
    sub = "Binary" if is_binary else "Multiclass"
    ic2.metric("Type", f"{sub} ({n_classes}cls)")
else:
    ic2.metric("Type", "Regression")
ic3.metric("Target", metadata["target_column"])
if problem_type == "Classification":
    ic4.metric("Accuracy", f"{metadata.get('accuracy', 0):.4f}")
else:
    ic4.metric("R\u00b2", f"{metadata.get('r2_score', 0):.4f}")

render_metrics_row([
    ("Train Samples", f"{metadata['train_size']:,}", ""),
    ("Test Samples", f"{metadata['test_size']:,}", ""),
    ("Features", f"{len(metadata['input_columns'])}", ""),
    ("F1 Score" if problem_type == "Classification" else "RMSE",
     f"{metadata.get('f1_score', metadata.get('rmse', 0)):.4f}",
     "green" if metadata.get("f1_score", metadata.get("r2_score", 0)) > 0.9 else "amber"),
])

render_divider()

# ── Load artifacts ───────────────────────────────────
if is_lstm:
    # import tensorflow as tf
    # tf.get_logger().setLevel("ERROR")
    from tensorflow import keras
    model = keras.models.load_model(os.path.join(model_path, "model.keras"))
else:
    model = joblib.load(os.path.join(model_path, "model.joblib"))

scaler = joblib.load(os.path.join(model_path, "scaler.joblib"))

label_encoder_y = None
if metadata.get("has_label_encoder_y"):
    label_encoder_y = joblib.load(os.path.join(model_path, "label_encoder_y.joblib"))

label_encoders_x = {}
if metadata.get("has_label_encoders_x"):
    label_encoders_x = joblib.load(os.path.join(model_path, "label_encoders_x.joblib"))

input_columns = metadata["input_columns"]


def _predict(scaled_data):
    if is_lstm:
        n_features = scaled_data.shape[1]
        lstm_in = scaled_data.values.reshape(-1, 1, n_features) if hasattr(scaled_data, "values") else scaled_data.reshape(-1, 1, n_features)
        raw = model.predict(lstm_in, verbose=0)
        if problem_type == "Classification":
            if is_binary:
                proba = raw.ravel()
                return (proba >= 0.5).astype(int), np.column_stack([1 - proba, proba])
            else:
                return np.argmax(raw, axis=1), raw
        return raw.ravel(), None
    preds = model.predict(scaled_data)
    proba = model.predict_proba(scaled_data) if (problem_type == "Classification" and hasattr(model, "predict_proba")) else None
    return preds, proba


def _decode(preds):
    return label_encoder_y.inverse_transform(preds) if label_encoder_y else preds


def _class_names():
    if label_encoder_y:
        return label_encoder_y.classes_
    if not is_lstm and hasattr(model, "classes_"):
        return model.classes_
    return [str(i) for i in range(n_classes)]


# ── Step 2: Input data ──────────────────────────────
render_step(2, "Provide Input Data")

input_method = st.radio("Input method", ["Manual Entry", "Upload CSV"], horizontal=True)

if input_method == "Manual Entry":
    with st.expander("Enter feature values", expanded=True):
        input_values = {}
        cols = st.columns(3)
        for i, col_name in enumerate(input_columns):
            with cols[i % 3]:
                if col_name in label_encoders_x:
                    options = list(label_encoders_x[col_name].classes_)
                    input_values[col_name] = st.selectbox(col_name, options, key=f"inf_{col_name}")
                else:
                    input_values[col_name] = st.number_input(col_name, value=0.0, format="%.4f", key=f"inf_{col_name}")

    if st.button("Predict", type="primary", use_container_width=True):
        input_df = pd.DataFrame([input_values])
        for col, le in label_encoders_x.items():
            if col in input_df.columns:
                input_df[col] = le.transform(input_df[col].astype(str))
        input_scaled = pd.DataFrame(scaler.transform(input_df), columns=input_columns)

        preds, proba = _predict(input_scaled)
        decoded = _decode(preds)

        render_divider()
        render_step(3, "Prediction Result")
        render_result_banner("Predicted Value", str(decoded[0]))

        if proba is not None and problem_type == "Classification":
            classes = _class_names()
            proba_df = pd.DataFrame({
                "Class": [str(c) for c in classes],
                "Probability": proba[0],
            }).sort_values("Probability", ascending=False)

            p1, p2 = st.columns(2)
            with p1:
                st.dataframe(proba_df, hide_index=True, width="stretch")
            with p2:
                fig = px.bar(proba_df, x="Class", y="Probability", color="Class",
                              template=PLOTLY_TEMPLATE, color_discrete_sequence=PLOTLY_COLORS)
                fig.update_layout(height=320, margin=dict(t=30, b=30), showlegend=False)
                st.plotly_chart(fig, width="stretch")

else:
    uploaded = st.file_uploader("Upload CSV for batch prediction", type=["csv"], key="inf_upload")
    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
        st.success(f"Uploaded **{uploaded.name}** &mdash; {batch_df.shape[0]:,} rows")
        with st.expander("Preview uploaded data", expanded=True):
            st.dataframe(batch_df.head(), width="stretch")

        missing = [c for c in input_columns if c not in batch_df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            st.stop()

        if st.button("Run Batch Prediction", type="primary", use_container_width=True):
            batch_input = batch_df[input_columns].copy()
            for col, le in label_encoders_x.items():
                if col in batch_input.columns:
                    batch_input[col] = le.transform(batch_input[col].astype(str))
            batch_scaled = pd.DataFrame(scaler.transform(batch_input), columns=input_columns)

            preds, proba = _predict(batch_scaled)
            decoded = _decode(preds)

            result_df = batch_df.copy()
            result_df["Prediction"] = decoded

            if proba is not None and problem_type == "Classification":
                classes = _class_names()
                for i, cls in enumerate(classes):
                    result_df[f"Prob_{cls}"] = proba[:, i]

            render_divider()
            render_step(3, "Batch Results")

            # Summary metrics
            if problem_type == "Classification":
                pred_counts = pd.Series(decoded).value_counts()
                render_metrics_row([
                    ("Total Rows", f"{len(result_df):,}", ""),
                    ("Unique Predictions", f"{pred_counts.shape[0]}", ""),
                    ("Most Common", str(pred_counts.index[0]), "green"),
                    ("Least Common", str(pred_counts.index[-1]), "amber"),
                ])

                fig_dist = px.histogram(result_df, x="Prediction", color="Prediction",
                                         template=PLOTLY_TEMPLATE, color_discrete_sequence=PLOTLY_COLORS)
                fig_dist.update_layout(height=300, margin=dict(t=30, b=30), title="Prediction Distribution")
                st.plotly_chart(fig_dist, width="stretch")

            with st.expander("Full Results Table", expanded=True):
                st.dataframe(result_df, width="stretch")

            csv_out = result_df.to_csv(index=False)
            st.download_button(
                "Download Predictions CSV",
                csv_out,
                file_name="predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )
