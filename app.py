import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Steel Plate Fault Classifier", layout="wide")
st.title("Steel Plate Fault Classifier")
st.caption("Deployment smoke test")

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"


@st.cache_resource
def load_metadata():
    with open(MODEL_DIR / "metadata.json") as f:
        return json.load(f)


meta = load_metadata()
st.write("Trained with scikit-learn:", meta["sklearn_version"])

display_name = st.selectbox("Model", sorted(meta["models"].values()))
slug = next(k for k, v in meta["models"].items() if v == display_name)

pipe = joblib.load(MODEL_DIR / f"{slug}.joblib")
st.success(f"Loaded {display_name} - {len(meta['feature_names'])} features, {len(meta['class_names'])} classes")

df = pd.read_csv(ROOT / "test_data.csv")
preds = pipe.predict(df[meta["feature_names"]].head())
st.write("Sample predictions:", list(preds))
