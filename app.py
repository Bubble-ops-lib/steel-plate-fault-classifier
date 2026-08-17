import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)

st.set_page_config(page_title="Steel Plate Fault Classifier", layout="wide")

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
TARGET = "Fault_Type"


@st.cache_resource
def load_metadata():
    with open(MODEL_DIR / "metadata.json") as f:
        return json.load(f)


@st.cache_resource
def load_model(slug):
    return joblib.load(MODEL_DIR / f"{slug}.joblib")


meta = load_metadata()
FEATURES = meta["feature_names"]

st.title("Steel Plate Fault Classifier")
st.caption(
    "UCI Steel Plates Faults - 27 surface features, 7 fault types. "
    "Five classifiers trained on 1552 plates, evaluated on a held-out 389."
)

st.sidebar.header("1. Test data")
uploaded = st.sidebar.file_uploader("Upload CSV", type="csv")

st.sidebar.header("2. Model")
display_name = st.sidebar.selectbox("Classifier", sorted(meta["models"].values()))
slug = next(k for k, v in meta["models"].items() if v == display_name)

st.sidebar.header("3. Averaging")
average = st.sidebar.radio(
    "Multi-class averaging", ["macro", "weighted"], index=0,
    help="macro treats every fault type equally; weighted favours common classes.",
)

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.success(f"Using uploaded file - {df.shape[0]} rows, {df.shape[1]} columns")
else:
    df = pd.read_csv(ROOT / "test_data.csv")
    st.info(f"Using bundled test_data.csv - {df.shape[0]} rows. Upload your own in the sidebar.")

missing = [c for c in FEATURES if c not in df.columns]
if missing:
    st.error(f"CSV is missing {len(missing)} required feature column(s), e.g. {missing[:4]}")
    st.stop()

X = df[FEATURES]
has_labels = TARGET in df.columns

pipe = load_model(slug)
y_pred = pipe.predict(X)

st.subheader(f"Predictions - {display_name}")
preview = pd.DataFrame({"Predicted": y_pred})
if has_labels:
    preview["Actual"] = df[TARGET].values
    preview["Correct"] = preview["Predicted"] == preview["Actual"]
st.dataframe(preview.head(20), width="stretch")

if not has_labels:
    st.warning(f"No '{TARGET}' column found, so metrics cannot be computed.")
    st.stop()

y_true = df[TARGET]


def compute_metrics(p, X, y, average):
    pred = p.predict(X)
    m = {
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, average=average, zero_division=0),
        "Recall": recall_score(y, pred, average=average, zero_division=0),
        "F1": f1_score(y, pred, average=average, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }
    try:
        m["AUC"] = roc_auc_score(y, p.predict_proba(X), multi_class="ovr",
                                 average=average, labels=p.classes_)
    except Exception:
        m["AUC"] = float("nan")
    return {k: m[k] for k in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]}


st.subheader(f"Evaluation metrics - {average} averaging")
scores = compute_metrics(pipe, X, y_true, average)
for col, (label, value) in zip(st.columns(6), scores.items()):
    col.metric(label, f"{value:.4f}")

left, right = st.columns(2)

with left:
    st.subheader("Confusion matrix")
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, ax=ax, xticks_rotation=45, colorbar=False, cmap="Blues"
    )
    ax.set_title(display_name)
    plt.tight_layout()
    st.pyplot(fig)

with right:
    st.subheader("Classification report")
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(3), width="stretch")

st.subheader("All five models on this data")
comparison = pd.DataFrame(
    {name: compute_metrics(load_model(s), X, y_true, average)
     for s, name in meta["models"].items()}
).T.round(4)
st.dataframe(comparison, width="stretch")
st.caption("Same test data, same metrics - the comparison table from the README, computed live.")
