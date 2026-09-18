import os
import json

import streamlit as st
import pandas as pd
import plotly.express as px

from pages.styles import (
    inject_css,
    render_page_title,
    render_step,
    render_divider,
    render_result_banner,
    render_sidebar,
    PLOTLY_TEMPLATE,
    PLOTLY_COLORS,
)

inject_css()
render_sidebar()

SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved_models")

render_page_title(
    "Compare Models", "See how different models stack up on the same dataset"
)

# ── Load all saved models' metadata ──────────────────
render_step(1, "Select Dataset")

if not os.path.isdir(SAVE_DIR) or not os.listdir(SAVE_DIR):
    st.warning(
        "No saved models found. Train a model first on the **Train Model** page."
    )
    st.stop()

rows = []
for model_dir in sorted(os.listdir(SAVE_DIR), reverse=True):
    meta_path = os.path.join(SAVE_DIR, model_dir, "metadata.json")
    if not os.path.exists(meta_path):
        continue
    with open(meta_path) as f:
        meta = json.load(f)
    rows.append(
        {
            "Model": model_dir,
            "Dataset": meta.get("dataset_name", "Unknown"),
            "Algorithm": meta.get("algorithm", "Unknown"),
            "Problem Type": meta.get("problem_type", "Unknown"),
            "Target": meta.get("target_column", "Unknown"),
            "Train Size": meta.get("train_size", 0),
            "Test Size": meta.get("test_size", 0),
            "Accuracy": meta.get("accuracy"),
            "F1 Score": meta.get("f1_score"),
            "R2 Score": meta.get("r2_score"),
            "RMSE": meta.get("rmse"),
            "Timestamp": meta.get("timestamp", ""),
        }
    )

if not rows:
    st.warning("No valid model metadata found.")
    st.stop()

all_models_df = pd.DataFrame(rows)
all_models_df["Score"] = all_models_df["Accuracy"].fillna(all_models_df["R2 Score"])

dataset_names = sorted(all_models_df["Dataset"].unique())
selected_dataset = st.selectbox("Dataset", dataset_names)

dataset_df = all_models_df[all_models_df["Dataset"] == selected_dataset].reset_index(
    drop=True
)

render_divider()

# ── Select models to compare ─────────────────────────
render_step(2, "Select Models to Compare")

model_options = dataset_df["Model"].tolist()
selected_models = st.multiselect(
    f"Models trained on **{selected_dataset}**",
    model_options,
    default=model_options,
)

if len(selected_models) < 2:
    st.info("Select at least 2 models to compare.")
    st.stop()

compare_df = (
    dataset_df[dataset_df["Model"].isin(selected_models)]
    .sort_values("Score", ascending=False)
    .reset_index(drop=True)
)

render_divider()

# ── Comparison ────────────────────────────────────────
render_step(3, "Comparison")

best = compare_df.iloc[0]
metric_label = (
    "Accuracy"
    if pd.notna(best["Score"]) and best["Problem Type"] == "Classification"
    else "R² Score"
)
render_result_banner(
    f"Best Model &mdash; {metric_label}", f"{best['Algorithm']} ({best['Score']:.4f})"
)

st.dataframe(
    compare_df.drop(columns=["Score"]).round(4),
    hide_index=True,
    width="stretch",
)

cls_df = compare_df[compare_df["Problem Type"] == "Classification"]
reg_df = compare_df[compare_df["Problem Type"] == "Regression"]

chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    if not cls_df.empty:
        fig_acc = px.bar(
            cls_df,
            x="Algorithm",
            y=["Accuracy", "F1 Score"],
            barmode="group",
            template=PLOTLY_TEMPLATE,
            color_discrete_sequence=PLOTLY_COLORS,
            title="Classification Metrics",
        )
        fig_acc.update_layout(height=380, margin=dict(t=40, b=30), yaxis_title="Score")
        st.plotly_chart(fig_acc, width="stretch")
with chart_col2:
    if not reg_df.empty:
        fig_reg = px.bar(
            reg_df,
            x="Algorithm",
            y="R2 Score",
            template=PLOTLY_TEMPLATE,
            color_discrete_sequence=PLOTLY_COLORS,
            title="Regression: R² Score",
        )
        fig_reg.update_layout(height=380, margin=dict(t=40, b=30))
        st.plotly_chart(fig_reg, width="stretch")

        fig_rmse = px.bar(
            reg_df,
            x="Algorithm",
            y="RMSE",
            template=PLOTLY_TEMPLATE,
            color_discrete_sequence=[PLOTLY_COLORS[4]],
            title="Regression: RMSE (lower is better)",
        )
        fig_rmse.update_layout(height=380, margin=dict(t=40, b=30))
        st.plotly_chart(fig_rmse, width="stretch")
