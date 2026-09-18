import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
    roc_curve, auc, roc_auc_score,
)

# --- Classification algorithms ---
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# --- Regression algorithms ---
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    AdaBoostRegressor,
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# --- LSTM (TensorFlow / Keras) ---
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="keras")
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from tensorflow import keras

# --- UI theme ---
from pages.styles import (
    inject_css, render_page_title, render_step, render_divider,
    render_metrics_row, render_sidebar, PLOTLY_TEMPLATE, PLOTLY_COLORS,
)

inject_css()
render_sidebar()

SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved_models")
os.makedirs(SAVE_DIR, exist_ok=True)

LSTM_SENTINEL = "LSTM"

CLASSIFICATION_MODELS = {
    "Logistic Regression": LogisticRegression,
    "Decision Tree": DecisionTreeClassifier,
    "Random Forest": RandomForestClassifier,
    "Gradient Boosting": GradientBoostingClassifier,
    "AdaBoost": AdaBoostClassifier,
    "Support Vector Machine (SVC)": SVC,
    "K-Nearest Neighbors": KNeighborsClassifier,
    "Naive Bayes (Gaussian)": GaussianNB,
    "LSTM (Deep Learning)": LSTM_SENTINEL,
}

REGRESSION_MODELS = {
    "Linear Regression": LinearRegression,
    "Ridge Regression": Ridge,
    "Lasso Regression": Lasso,
    "ElasticNet": ElasticNet,
    "Decision Tree": DecisionTreeRegressor,
    "Random Forest": RandomForestRegressor,
    "Gradient Boosting": GradientBoostingRegressor,
    "AdaBoost": AdaBoostRegressor,
    "Support Vector Machine (SVR)": SVR,
    "K-Nearest Neighbors": KNeighborsRegressor,
    "LSTM (Deep Learning)": LSTM_SENTINEL,
}


# ── helpers ──────────────────────────────────────────
def detect_target_column(df: pd.DataFrame) -> str | None:
    common_names = [
        "target", "label", "class", "output", "y",
        "fault", "fault_id", "category", "result", "diagnosis",
        "status", "failure", "defect",
    ]
    for col in df.columns:
        if col.strip().lower() in common_names:
            return col
    for col in reversed(df.columns.tolist()):
        if df[col].dtype == "object":
            return col
    for col in reversed(df.columns.tolist()):
        if df[col].nunique() < 20:
            return col
    return df.columns[-1]


def detect_problem_type(df: pd.DataFrame, target_col: str) -> str:
    if df[target_col].dtype == "object":
        return "Classification"
    if df[target_col].nunique() < 20:
        return "Classification"
    return "Regression"


def get_classification_subtype(y: pd.Series) -> str:
    return "Binary" if y.nunique() == 2 else "Multiclass"


def balance_test_set(X_test, y_test):
    combined = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
    target_col = y_test.name or combined.columns[-1]
    class_counts = combined[target_col].value_counts()
    min_count = class_counts.min()
    balanced_parts = []
    for cls in class_counts.index:
        cls_subset = combined[combined[target_col] == cls]
        balanced_parts.append(cls_subset.sample(n=min_count, random_state=42))
    balanced = pd.concat(balanced_parts).sample(frac=1, random_state=42).reset_index(drop=True)
    return balanced.drop(columns=[target_col]), balanced[target_col]


def build_lstm_model(n_features, problem_type, n_classes, lstm_units, dense_units, dropout):
    model = keras.Sequential([
        keras.layers.Input(shape=(1, n_features)),
        keras.layers.LSTM(lstm_units, return_sequences=False),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(dense_units, activation="relu"),
        keras.layers.Dropout(dropout),
    ])
    if problem_type == "Classification":
        if n_classes == 2:
            model.add(keras.layers.Dense(1, activation="sigmoid"))
            model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        else:
            model.add(keras.layers.Dense(n_classes, activation="softmax"))
            model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    else:
        model.add(keras.layers.Dense(1))
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


# ── UI ───────────────────────────────────────────────
render_page_title("Model Training", "Upload data, explore, train and evaluate ML models")

# ─── Step 1: Upload ─────────────────────────────────
render_step(1, "Upload Dataset")

upload_tab, sample_tab = st.tabs(["Upload CSV", "Use Sample Dataset"])
with upload_tab:
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
with sample_tab:
    sample_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset", "air-compressor.csv")
    use_sample = st.button("Load air-compressor.csv", type="primary")

df = None
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"Loaded **{uploaded_file.name}** &mdash; {df.shape[0]:,} rows, {df.shape[1]} columns")
elif use_sample or st.session_state.get("use_sample"):
    st.session_state["use_sample"] = True
    if os.path.exists(sample_path):
        df = pd.read_csv(sample_path)
        st.success(f"Loaded **air-compressor.csv** &mdash; {df.shape[0]:,} rows, {df.shape[1]} columns")
    else:
        st.error(f"Sample file not found at {sample_path}")

if df is None:
    st.info("Upload a CSV file or load the sample dataset to begin.")
    st.stop()

render_divider()

# ─── Step 2: Preview & Analysis ─────────────────────
render_step(2, "Data Preview & Analysis")

# Quick stats at top
qs1, qs2, qs3, qs4 = st.columns(4)
qs1.metric("Rows", f"{df.shape[0]:,}")
qs2.metric("Columns", f"{df.shape[1]}")
qs3.metric("Numeric", f"{df.select_dtypes(include='number').shape[1]}")
qs4.metric("Categorical", f"{df.select_dtypes(include='object').shape[1]}")

col_preview, col_stats = st.columns(2)
with col_preview:
    with st.expander("Data Preview (first 10 rows)", expanded=True):
        st.dataframe(df.head(10), width="stretch")
with col_stats:
    with st.expander("Column Information", expanded=True):
        info_df = pd.DataFrame({
            "Column": df.columns,
            "Type": [str(d) for d in df.dtypes],
            "Non-Null": df.notnull().sum().values,
            "Null": df.isnull().sum().values,
            "Unique": df.nunique().values,
        })
        st.dataframe(info_df, width="stretch", hide_index=True)

with st.expander("Statistical Summary"):
    st.dataframe(df.describe(), width="stretch")

# ─── Missing Value Handling ──────────────────────────
total_missing = df.isnull().sum()
cols_with_missing = total_missing[total_missing > 0]

if len(cols_with_missing) > 0:
    st.warning(f"**{len(cols_with_missing)} columns** with missing values detected")
    with st.expander("Missing Value Details & Fix", expanded=True):
        miss_df = pd.DataFrame({
            "Column": cols_with_missing.index,
            "Missing Count": cols_with_missing.values,
            "Missing %": (cols_with_missing.values / len(df) * 100).round(2),
        })
        mc1, mc2 = st.columns([1, 1])
        with mc1:
            st.dataframe(miss_df, width="stretch", hide_index=True)
        with mc2:
            fig_miss = px.bar(
                miss_df, x="Column", y="Missing %",
                text="Missing Count", color="Missing %",
                color_continuous_scale="Reds",
                template=PLOTLY_TEMPLATE,
            )
            fig_miss.update_layout(height=300, margin=dict(t=30, b=30))
            st.plotly_chart(fig_miss, width="stretch")

        strategy = st.selectbox(
            "Fix strategy",
            ["Drop rows with any missing value",
             "Fill with Mean (numeric) / Mode (categorical)",
             "Fill with Median (numeric) / Mode (categorical)",
             "Fill with Zero (numeric) / 'Unknown' (categorical)"],
        )
        if st.button("Apply Fix"):
            before_rows = len(df)
            if strategy == "Drop rows with any missing value":
                df.dropna(inplace=True)
                st.success(f"Dropped {before_rows - len(df)} rows. Remaining: {len(df):,}")
            elif strategy.startswith("Fill with Mean"):
                for col in cols_with_missing.index:
                    if df[col].dtype in ("float64", "int64"):
                        df[col].fillna(df[col].mean(), inplace=True)
                    else:
                        df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown", inplace=True)
                st.success("Filled with Mean / Mode.")
            elif strategy.startswith("Fill with Median"):
                for col in cols_with_missing.index:
                    if df[col].dtype in ("float64", "int64"):
                        df[col].fillna(df[col].median(), inplace=True)
                    else:
                        df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown", inplace=True)
                st.success("Filled with Median / Mode.")
            else:
                for col in cols_with_missing.index:
                    if df[col].dtype in ("float64", "int64"):
                        df[col].fillna(0, inplace=True)
                    else:
                        df[col].fillna("Unknown", inplace=True)
                st.success("Filled with Zero / 'Unknown'.")
            remaining = df.isnull().sum().sum()
            if remaining == 0:
                st.info("All missing values handled.")
            else:
                st.warning(f"Still {remaining} missing values remaining.")
else:
    st.success("No missing values found.")

render_divider()

# ─── Step 3: Column Selection ────────────────────────
render_step(3, "Select Input & Output Columns")

auto_target = detect_target_column(df)
auto_problem = detect_problem_type(df, auto_target) if auto_target else "Classification"

st.info(f"Auto-detected target: **{auto_target}** &bull; Problem type: **{auto_problem}**")

tc1, tc2 = st.columns([1, 2])
with tc1:
    target_col = st.selectbox(
        "Output (Y) Column",
        options=df.columns.tolist(),
        index=df.columns.tolist().index(auto_target) if auto_target in df.columns else len(df.columns) - 1,
    )
with tc2:
    remaining_cols = [c for c in df.columns if c != target_col]
    input_cols = st.multiselect("Input (X) Columns", options=remaining_cols, default=remaining_cols)

if not input_cols:
    st.warning("Please select at least one input column.")
    st.stop()

problem_type = detect_problem_type(df, target_col)

if problem_type == "Classification":
    cls_subtype = get_classification_subtype(df[target_col])
    n_unique_classes = df[target_col].nunique()
    st.markdown(f"**{cls_subtype} Classification** &mdash; {n_unique_classes} classes detected")
else:
    cls_subtype = None
    st.markdown(f"**{problem_type}** problem detected")

with st.expander("Target Distribution", expanded=True):
    if problem_type == "Classification":
        fig_target = px.histogram(df, x=target_col, color=target_col, template=PLOTLY_TEMPLATE,
                                   color_discrete_sequence=PLOTLY_COLORS, title="Class Distribution")
    else:
        fig_target = px.histogram(df, x=target_col, nbins=30, template=PLOTLY_TEMPLATE,
                                   color_discrete_sequence=PLOTLY_COLORS, title="Target Distribution")
    fig_target.update_layout(height=350, margin=dict(t=40, b=30))
    st.plotly_chart(fig_target, width="stretch")

render_divider()

# ─── Step 4: EDA ────────────────────────────────────
render_step(4, "Exploratory Data Analysis")

numeric_input_cols = [c for c in input_cols if df[c].dtype in ("int64", "float64")]
categorical_input_cols = [c for c in input_cols if df[c].dtype == "object"]

eda_tab_corr, eda_tab_dist, eda_tab_box, eda_tab_scatter, eda_tab_cat = st.tabs(
    ["Correlation", "Distributions", "Box Plots", "Scatter", "Categorical"]
)

with eda_tab_corr:
    if len(numeric_input_cols) >= 2:
        corr = df[numeric_input_cols].corr()
        fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                              aspect="auto", zmin=-1, zmax=1, template=PLOTLY_TEMPLATE)
        fig_corr.update_layout(height=550, margin=dict(t=30, b=20))
        st.plotly_chart(fig_corr, width="stretch")
        high_corr_pairs = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                val = corr.iloc[i, j]
                if abs(val) >= 0.8:
                    high_corr_pairs.append({"Feature 1": corr.columns[i], "Feature 2": corr.columns[j], "Correlation": round(val, 4)})
        if high_corr_pairs:
            st.markdown("**Highly correlated pairs (|r| >= 0.8)**")
            st.dataframe(pd.DataFrame(high_corr_pairs), hide_index=True, width="stretch")
    else:
        st.info("Need at least 2 numeric columns.")

with eda_tab_dist:
    if numeric_input_cols:
        dist_col = st.selectbox("Feature", numeric_input_cols, key="eda_dist_col")
        d1, d2 = st.columns(2)
        with d1:
            fig_hist = px.histogram(df, x=dist_col, nbins=40, marginal="rug", template=PLOTLY_TEMPLATE,
                                     color=target_col if problem_type == "Classification" else None,
                                     color_discrete_sequence=PLOTLY_COLORS)
            fig_hist.update_layout(height=380, margin=dict(t=30, b=30))
            st.plotly_chart(fig_hist, width="stretch")
        with d2:
            fig_v = px.violin(df, y=dist_col, box=True, points="outliers", template=PLOTLY_TEMPLATE,
                               color=target_col if problem_type == "Classification" else None,
                               color_discrete_sequence=PLOTLY_COLORS)
            fig_v.update_layout(height=380, margin=dict(t=30, b=30))
            st.plotly_chart(fig_v, width="stretch")
    else:
        st.info("No numeric columns.")

with eda_tab_box:
    if numeric_input_cols:
        box_cols = st.multiselect("Features", numeric_input_cols, default=numeric_input_cols[:6], key="eda_box_cols")
        if box_cols:
            melted = df[box_cols + [target_col]].melt(id_vars=[target_col], var_name="Feature", value_name="Value")
            fig_box = px.box(melted, x="Feature", y="Value", template=PLOTLY_TEMPLATE,
                              color=target_col if problem_type == "Classification" else None,
                              color_discrete_sequence=PLOTLY_COLORS)
            fig_box.update_layout(height=450, margin=dict(t=30, b=30))
            st.plotly_chart(fig_box, width="stretch")

            with st.expander("Outlier Summary (IQR Method)"):
                outlier_info = []
                for col in box_cols:
                    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                    iqr = q3 - q1
                    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    n_out = int(((df[col] < lo) | (df[col] > hi)).sum())
                    outlier_info.append({"Feature": col, "Q1": round(q1, 2), "Q3": round(q3, 2),
                                          "IQR": round(iqr, 2), "Lower": round(lo, 2), "Upper": round(hi, 2),
                                          "Outliers": n_out, "%": round(n_out / len(df) * 100, 2)})
                st.dataframe(pd.DataFrame(outlier_info), hide_index=True, width="stretch")
    else:
        st.info("No numeric columns.")

with eda_tab_scatter:
    if len(numeric_input_cols) >= 2:
        s1, s2 = st.columns(2)
        with s1:
            x_feat = st.selectbox("X-axis", numeric_input_cols, index=0, key="eda_sc_x")
        with s2:
            y_feat = st.selectbox("Y-axis", numeric_input_cols, index=min(1, len(numeric_input_cols) - 1), key="eda_sc_y")
        fig_sc = px.scatter(df, x=x_feat, y=y_feat, opacity=0.6, template=PLOTLY_TEMPLATE,
                             color=target_col if problem_type == "Classification" else None,
                             color_discrete_sequence=PLOTLY_COLORS)
        fig_sc.update_layout(height=420, margin=dict(t=30, b=30))
        st.plotly_chart(fig_sc, width="stretch")
    else:
        st.info("Need at least 2 numeric columns.")

with eda_tab_cat:
    if categorical_input_cols:
        cat_col = st.selectbox("Feature", categorical_input_cols, key="eda_cat_col")
        c1, c2 = st.columns(2)
        with c1:
            fig_bar = px.histogram(df, x=cat_col, barmode="group", template=PLOTLY_TEMPLATE,
                                    color=target_col if problem_type == "Classification" else None,
                                    color_discrete_sequence=PLOTLY_COLORS)
            fig_bar.update_layout(height=380, margin=dict(t=30, b=30))
            st.plotly_chart(fig_bar, width="stretch")
        with c2:
            fig_pie = px.pie(df, names=cat_col, template=PLOTLY_TEMPLATE, color_discrete_sequence=PLOTLY_COLORS)
            fig_pie.update_layout(height=380, margin=dict(t=30, b=30))
            st.plotly_chart(fig_pie, width="stretch")
    else:
        st.info("No categorical input columns.")

render_divider()

# ─── Step 5: Model Selection ────────────────────────
render_step(5, "Select Model")

model_catalog = CLASSIFICATION_MODELS if problem_type == "Classification" else REGRESSION_MODELS
algo_name = st.selectbox("Algorithm", list(model_catalog.keys()))
is_lstm = model_catalog[algo_name] == LSTM_SENTINEL

if is_lstm:
    with st.expander("LSTM Hyperparameters", expanded=True):
        lc1, lc2, lc3, lc4 = st.columns(4)
        with lc1:
            lstm_units = st.number_input("LSTM Units", min_value=8, max_value=256, value=64, step=8)
        with lc2:
            dense_units = st.number_input("Dense Units", min_value=8, max_value=128, value=32, step=8)
        with lc3:
            lstm_epochs = st.number_input("Epochs", min_value=5, max_value=200, value=50, step=5)
        with lc4:
            lstm_batch = st.number_input("Batch Size", min_value=8, max_value=256, value=32, step=8)
        lstm_dropout = st.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.05)

render_divider()

# ─── Step 6: Train ──────────────────────────────────
render_step(6, "Train Model")

if st.button("Train Model", type="primary", use_container_width=True):
    with st.spinner("Training in progress..."):
        X = df[input_cols].copy()
        y = df[target_col].copy()

        # Handle NaN
        x_missing = X.isnull().sum().sum()
        y_missing = y.isnull().sum()
        if x_missing > 0 or y_missing > 0:
            st.warning(f"Auto-fixing {x_missing} input & {y_missing} target missing values.")
            if y_missing > 0:
                valid = y.notnull()
                X, y = X[valid].reset_index(drop=True), y[valid].reset_index(drop=True)
            for col in X.columns:
                if X[col].isnull().any():
                    if X[col].dtype in ("float64", "int64"):
                        X[col].fillna(X[col].median(), inplace=True)
                    else:
                        mode_val = X[col].mode()
                        X[col].fillna(mode_val[0] if not mode_val.empty else "Unknown", inplace=True)

        # Encode
        label_encoders_x = {}
        for col in X.columns:
            if X[col].dtype == "object":
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders_x[col] = le

        label_encoder_y = None
        if problem_type == "Classification" and y.dtype == "object":
            label_encoder_y = LabelEncoder()
            y = pd.Series(label_encoder_y.fit_transform(y.astype(str)), name=target_col)

        n_classes = int(y.nunique()) if problem_type == "Classification" else 0
        is_binary = (problem_type == "Classification" and n_classes == 2)

        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42,
            stratify=y if problem_type == "Classification" else None,
        )
        if problem_type == "Classification":
            X_test, y_test = balance_test_set(X_test, y_test)

        render_metrics_row([
            ("Train Samples", f"{X_train.shape[0]:,}", ""),
            ("Test Samples", f"{X_test.shape[0]:,}", ""),
            ("Features", f"{X_train.shape[1]}", ""),
            ("Classes" if problem_type == "Classification" else "Target Type",
             f"{n_classes}" if problem_type == "Classification" else "Continuous", ""),
        ])

        if problem_type == "Classification":
            test_dist = y_test.value_counts()
            if label_encoder_y:
                test_dist.index = label_encoder_y.inverse_transform(test_dist.index)
            with st.expander("Balanced test set distribution"):
                st.dataframe(test_dist.reset_index().rename(columns={"index": "Class", target_col: "Class", "count": "Count"}), hide_index=True)

        # ── Train ──
        if is_lstm:
            n_features = X_train.shape[1]
            X_train_lstm = X_train.values.reshape(-1, 1, n_features)
            X_test_lstm = X_test.values.reshape(-1, 1, n_features)
            model = build_lstm_model(n_features, problem_type, n_classes, lstm_units, dense_units, lstm_dropout)

            with st.expander("LSTM Architecture"):
                summary_lines = []
                model.summary(print_fn=lambda line: summary_lines.append(line))
                st.code("\n".join(summary_lines))

            progress_bar = st.progress(0, text="Training LSTM...")
            status_text = st.empty()

            class StreamlitCallback(keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    pct = (epoch + 1) / lstm_epochs
                    progress_bar.progress(pct, text=f"Epoch {epoch+1}/{lstm_epochs}")
                    status_text.caption(f"loss: {logs.get('loss',0):.4f} &bull; val_loss: {logs.get('val_loss',0):.4f}")

            history = model.fit(X_train_lstm, y_train.values, epochs=lstm_epochs,
                                 batch_size=lstm_batch, validation_split=0.15, verbose=0,
                                 callbacks=[StreamlitCallback()])
            progress_bar.empty()
            status_text.empty()
            st.success("LSTM training complete!")

            hist_df = pd.DataFrame(history.history)
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(y=hist_df["loss"], name="Train Loss", line=dict(color=PLOTLY_COLORS[0], width=2)))
            fig_loss.add_trace(go.Scatter(y=hist_df["val_loss"], name="Val Loss", line=dict(color=PLOTLY_COLORS[4], width=2)))
            fig_loss.update_layout(template=PLOTLY_TEMPLATE, height=350, margin=dict(t=40, b=30),
                                    title="Training History", xaxis_title="Epoch", yaxis_title="Loss")
            st.plotly_chart(fig_loss, width="stretch")

            raw_pred = model.predict(X_test_lstm, verbose=0)
            if problem_type == "Classification":
                if is_binary:
                    y_pred_proba = raw_pred.ravel()
                    y_pred = (y_pred_proba >= 0.5).astype(int)
                else:
                    y_pred_proba = raw_pred
                    y_pred = np.argmax(raw_pred, axis=1)
            else:
                y_pred = raw_pred.ravel()
                y_pred_proba = None
        else:
            ModelClass = model_catalog[algo_name]
            if algo_name == "Support Vector Machine (SVC)":
                model = ModelClass(probability=True, random_state=42)
            elif algo_name in ("Support Vector Machine (SVR)", "Naive Bayes (Gaussian)"):
                model = ModelClass()
            elif "random_state" in ModelClass().get_params():
                model = ModelClass(random_state=42)
            else:
                model = ModelClass()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = None
            st.success("Training complete!")

        render_divider()

        # ── Step 7: Analytics ──
        render_step(7, "Model Analytics")

        if problem_type == "Classification":
            acc = accuracy_score(y_test, y_pred)
            avg = "binary" if is_binary else "weighted"
            prec = precision_score(y_test, y_pred, average=avg, zero_division=0)
            rec = recall_score(y_test, y_pred, average=avg, zero_division=0)
            f1 = f1_score(y_test, y_pred, average=avg, zero_division=0)

            subtype_label = "Binary" if is_binary else f"Multiclass ({n_classes} classes)"
            st.markdown(f"**{subtype_label} Classification Results**")

            render_metrics_row([
                ("Accuracy", f"{acc:.4f}", "green" if acc > 0.9 else "amber" if acc > 0.7 else "rose"),
                ("Precision", f"{prec:.4f}", "green" if prec > 0.9 else "amber" if prec > 0.7 else "rose"),
                ("Recall", f"{rec:.4f}", "green" if rec > 0.9 else "amber" if rec > 0.7 else "rose"),
                ("F1 Score", f"{f1:.4f}", "green" if f1 > 0.9 else "amber" if f1 > 0.7 else "rose"),
            ])

            labels = label_encoder_y.classes_ if label_encoder_y else sorted(y_test.unique())

            # ROC
            if is_lstm:
                proba_for_roc = y_pred_proba
            elif hasattr(model, "predict_proba"):
                proba_for_roc = model.predict_proba(X_test)
            elif hasattr(model, "decision_function"):
                proba_for_roc = model.decision_function(X_test)
            else:
                proba_for_roc = None

            roc_col, cm_col = st.columns(2)

            with roc_col:
                if proba_for_roc is not None:
                    if is_binary:
                        scores = proba_for_roc[:, 1] if proba_for_roc.ndim == 2 else proba_for_roc
                        fpr, tpr, _ = roc_curve(y_test, scores)
                        roc_auc_val = auc(fpr, tpr)
                        fig_roc = go.Figure()
                        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC = {roc_auc_val:.4f}",
                                                      line=dict(color=PLOTLY_COLORS[0], width=2.5)))
                        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                                                      line=dict(dash="dash", color="#cbd5e1")))
                        fig_roc.update_layout(template=PLOTLY_TEMPLATE, title="ROC Curve", height=400,
                                               margin=dict(t=40, b=30), xaxis_title="FPR", yaxis_title="TPR")
                        st.plotly_chart(fig_roc, width="stretch")
                    elif proba_for_roc.ndim > 1:
                        try:
                            roc_auc_val = roc_auc_score(y_test, proba_for_roc, multi_class="ovr", average="weighted")
                            st.caption(f"Weighted ROC AUC (OvR): **{roc_auc_val:.4f}**")
                        except ValueError:
                            pass
                        fig_roc = go.Figure()
                        y_test_arr = np.array(y_test)
                        for i, label in enumerate(sorted(y_test.unique())):
                            y_bin = (y_test_arr == label).astype(int)
                            if proba_for_roc.shape[1] > i:
                                fpr_i, tpr_i, _ = roc_curve(y_bin, proba_for_roc[:, i])
                                auc_i = auc(fpr_i, tpr_i)
                                disp = str(labels[label]) if label_encoder_y else str(label)
                                fig_roc.add_trace(go.Scatter(x=fpr_i, y=tpr_i, mode="lines",
                                                              name=f"{disp} ({auc_i:.3f})",
                                                              line=dict(color=PLOTLY_COLORS[i % len(PLOTLY_COLORS)], width=2)))
                        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                                                      line=dict(dash="dash", color="#cbd5e1")))
                        fig_roc.update_layout(template=PLOTLY_TEMPLATE, title="ROC Curves (OvR)", height=400,
                                               margin=dict(t=40, b=30), xaxis_title="FPR", yaxis_title="TPR")
                        st.plotly_chart(fig_roc, width="stretch")

            with cm_col:
                cm = confusion_matrix(y_test, y_pred)
                fig_cm = px.imshow(cm, labels=dict(x="Predicted", y="Actual", color="Count"),
                                    x=[str(l) for l in labels], y=[str(l) for l in labels],
                                    text_auto=True, color_continuous_scale="Blues", template=PLOTLY_TEMPLATE)
                fig_cm.update_layout(height=400, margin=dict(t=40, b=30), title="Confusion Matrix")
                st.plotly_chart(fig_cm, width="stretch")

            with st.expander("Classification Report"):
                report = classification_report(y_test, y_pred,
                                                target_names=[str(l) for l in labels] if label_encoder_y else None,
                                                output_dict=True)
                st.dataframe(pd.DataFrame(report).T.round(4), width="stretch")

            if not is_lstm:
                if hasattr(model, "feature_importances_"):
                    with st.expander("Feature Importance"):
                        imp = pd.DataFrame({"Feature": input_cols, "Importance": model.feature_importances_}).sort_values("Importance", ascending=True)
                        fig_imp = px.bar(imp, x="Importance", y="Feature", orientation="h", template=PLOTLY_TEMPLATE,
                                          color_discrete_sequence=[PLOTLY_COLORS[0]])
                        fig_imp.update_layout(height=max(300, len(input_cols) * 25), margin=dict(t=30, b=20))
                        st.plotly_chart(fig_imp, width="stretch")
                elif hasattr(model, "coef_"):
                    with st.expander("Feature Coefficients"):
                        coefs = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
                        imp = pd.DataFrame({"Feature": input_cols[:len(coefs)], "Coefficient": coefs}).sort_values("Coefficient", ascending=True)
                        fig_imp = px.bar(imp, x="Coefficient", y="Feature", orientation="h", template=PLOTLY_TEMPLATE,
                                          color_discrete_sequence=[PLOTLY_COLORS[0]])
                        fig_imp.update_layout(height=max(300, len(input_cols) * 25), margin=dict(t=30, b=20))
                        st.plotly_chart(fig_imp, width="stretch")

        else:
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            render_metrics_row([
                ("MSE", f"{mse:.4f}", ""),
                ("RMSE", f"{rmse:.4f}", ""),
                ("MAE", f"{mae:.4f}", ""),
                ("R\u00b2 Score", f"{r2:.4f}", "green" if r2 > 0.9 else "amber" if r2 > 0.7 else "rose"),
            ])

            reg1, reg2 = st.columns(2)
            with reg1:
                fig_avp = px.scatter(x=y_test, y=y_pred, labels={"x": "Actual", "y": "Predicted"},
                                      template=PLOTLY_TEMPLATE, color_discrete_sequence=[PLOTLY_COLORS[0]])
                fig_avp.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()],
                                              mode="lines", name="Ideal", line=dict(dash="dash", color="#f43f5e")))
                fig_avp.update_layout(title="Actual vs Predicted", height=380, margin=dict(t=40, b=30))
                st.plotly_chart(fig_avp, width="stretch")
            with reg2:
                residuals = y_test.values - y_pred
                fig_res = px.histogram(residuals, nbins=30, template=PLOTLY_TEMPLATE,
                                        color_discrete_sequence=[PLOTLY_COLORS[1]])
                fig_res.update_layout(title="Residual Distribution", height=380, margin=dict(t=40, b=30))
                st.plotly_chart(fig_res, width="stretch")

            if not is_lstm and hasattr(model, "feature_importances_"):
                with st.expander("Feature Importance"):
                    imp = pd.DataFrame({"Feature": input_cols, "Importance": model.feature_importances_}).sort_values("Importance", ascending=True)
                    fig_imp = px.bar(imp, x="Importance", y="Feature", orientation="h", template=PLOTLY_TEMPLATE,
                                      color_discrete_sequence=[PLOTLY_COLORS[0]])
                    fig_imp.update_layout(height=max(300, len(input_cols) * 25), margin=dict(t=30, b=20))
                    st.plotly_chart(fig_imp, width="stretch")

        render_divider()

        # ── Save ──
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{algo_name.replace(' ', '_').replace('(', '').replace(')', '')}_{timestamp}"
        model_dir = os.path.join(SAVE_DIR, model_name)
        os.makedirs(model_dir, exist_ok=True)

        if is_lstm:
            model.save(os.path.join(model_dir, "model.keras"))
        else:
            joblib.dump(model, os.path.join(model_dir, "model.joblib"))
        joblib.dump(scaler, os.path.join(model_dir, "scaler.joblib"))
        if label_encoder_y:
            joblib.dump(label_encoder_y, os.path.join(model_dir, "label_encoder_y.joblib"))
        if label_encoders_x:
            joblib.dump(label_encoders_x, os.path.join(model_dir, "label_encoders_x.joblib"))

        metadata = {
            "algorithm": algo_name,
            "problem_type": problem_type,
            "is_binary": is_binary if problem_type == "Classification" else False,
            "n_classes": n_classes,
            "is_lstm": is_lstm,
            "input_columns": input_cols,
            "target_column": target_col,
            "train_size": int(X_train.shape[0]),
            "test_size": int(X_test.shape[0]),
            "timestamp": timestamp,
            "has_label_encoder_y": label_encoder_y is not None,
            "has_label_encoders_x": len(label_encoders_x) > 0,
        }
        if problem_type == "Classification":
            metadata["accuracy"] = float(acc)
            metadata["f1_score"] = float(f1)
        else:
            metadata["r2_score"] = float(r2)
            metadata["rmse"] = float(rmse)

        with open(os.path.join(model_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        st.success(f"Model saved to **saved_models/{model_name}/**")
        # st.balloons()
