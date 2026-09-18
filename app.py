import streamlit as st

st.set_page_config(page_title="ML Training Application", layout="wide")

train_page = st.Page("pages/train.py", title="Train Model", icon=":material/model_training:")
inference_page = st.Page("pages/inference.py", title="Inference", icon=":material/query_stats:")

pg = st.navigation([train_page, inference_page])
pg.run()
