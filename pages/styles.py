"""Shared CSS theme and helper functions for the ML Training Application."""
import streamlit as st

PLOTLY_TEMPLATE = "plotly_white"

PLOTLY_COLORS = [
    "#6366f1", "#8b5cf6", "#a78bfa", "#c4b5fd",
    "#22d3ee", "#2dd4bf", "#34d399", "#4ade80",
    "#facc15", "#fb923c", "#f87171", "#e879f9",
]


def inject_css():
    """Inject custom CSS into the Streamlit app."""
    st.markdown("""
    <style>
    /* ── Global ── */
    .block-container { padding-top: 1.5rem; }

    /* ── Gradient page title ── */
    .page-title {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .page-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    /* ── Section header with step badge ── */
    .step-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-top: 2rem;
        margin-bottom: 0.8rem;
    }
    .step-badge {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        width: 36px;
        height: 36px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 1rem;
        flex-shrink: 0;
    }
    .step-label {
        font-size: 1.35rem;
        font-weight: 700;
        color: #1e293b;
    }

    /* ── Info cards ── */
    .info-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    .info-card h4 {
        margin: 0 0 0.5rem 0;
        color: #475569;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .info-card .value {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1e293b;
    }

    /* ── Metric highlight row ── */
    .metric-row {
        display: flex;
        gap: 16px;
        flex-wrap: wrap;
        margin: 1rem 0;
    }
    .metric-box {
        flex: 1;
        min-width: 140px;
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .metric-box .label {
        font-size: 0.78rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .metric-box .number {
        font-size: 1.6rem;
        font-weight: 800;
        color: #6366f1;
    }
    .metric-box.green .number { color: #10b981; }
    .metric-box.amber .number { color: #f59e0b; }
    .metric-box.rose  .number { color: #f43f5e; }

    /* ── Accent divider ── */
    .accent-divider {
        height: 3px;
        background: linear-gradient(90deg, #6366f1, #a78bfa, transparent);
        border: none;
        border-radius: 2px;
        margin: 1.5rem 0 1rem 0;
    }

    /* ── Result banner ── */
    .result-banner {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        border-radius: 14px;
        padding: 1.5rem 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .result-banner .result-label {
        font-size: 0.85rem;
        opacity: 0.85;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .result-banner .result-value {
        font-size: 2.2rem;
        font-weight: 800;
        margin-top: 0.3rem;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
    }
    .sidebar-brand {
        text-align: center;
        padding: 0.5rem 0 1.5rem 0;
    }
    .sidebar-brand .logo {
        font-size: 2.5rem;
        margin-bottom: 0.3rem;
    }
    .sidebar-brand .name {
        font-size: 1.1rem;
        font-weight: 700;
        color: #6366f1;
    }
    .sidebar-brand .tagline {
        font-size: 0.78rem;
        color: #94a3b8;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
    }

    /* ── Buttons ── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-weight: 600;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)


def render_page_title(title: str, subtitle: str = ""):
    """Render a gradient page title with optional subtitle."""
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def render_step(number: int, label: str):
    """Render a step header with a numbered badge."""
    st.markdown(f"""
    <div class="step-header">
        <div class="step-badge">{number}</div>
        <div class="step-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def render_divider():
    """Render a styled accent divider."""
    st.markdown('<div class="accent-divider"></div>', unsafe_allow_html=True)


def render_metrics_row(metrics: list[tuple[str, str, str]]):
    """Render a row of styled metric boxes.
    metrics: list of (label, value, color) where color is '' | 'green' | 'amber' | 'rose'
    """
    boxes = ""
    for label, value, color in metrics:
        cls = f"metric-box {color}" if color else "metric-box"
        boxes += f"""
        <div class="{cls}">
            <div class="label">{label}</div>
            <div class="number">{value}</div>
        </div>"""
    st.markdown(f'<div class="metric-row">{boxes}</div>', unsafe_allow_html=True)


def render_result_banner(label: str, value: str):
    """Render a large centered result banner."""
    st.markdown(f"""
    <div class="result-banner">
        <div class="result-label">{label}</div>
        <div class="result-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render sidebar branding and information."""
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-brand">
            <div class="logo">&#129302;</div>
            <div class="name">ML AutoTrainer</div>
            <div class="tagline">Train &bull; Evaluate &bull; Deploy</div>
        </div>
        """, unsafe_allow_html=True)
        st.divider()
        st.markdown("**Quick Guide**")
        st.markdown("""
        1. Upload a CSV dataset
        2. Review data & fix missing values
        3. Select input/output columns
        4. Explore data with EDA charts
        5. Pick an algorithm & train
        6. Analyze results & save model
        7. Run inference on new data
        """)
        st.divider()
        st.markdown("**Supported Algorithms**")
        st.markdown("""
        - Logistic / Linear Regression
        - Decision Tree
        - Random Forest
        - Gradient Boosting / AdaBoost
        - SVM (SVC / SVR)
        - K-Nearest Neighbors
        - Naive Bayes
        - LSTM (Deep Learning)
        """)
        st.divider()
        st.caption("Built with Streamlit + scikit-learn + TensorFlow")
