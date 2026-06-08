# ── nvcb_streamlit.py ────────────────────────────────────────────────────────
# Run with:  streamlit run nvcb_streamlit.py
# Requires:  nvcb_gbm_model.pkl  |  nvcb_scaler.pkl  |  nvcb_meta.json
# ─────────────────────────────────────────────────────────────────────────────

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import json
import joblib
import streamlit as st

st.set_page_config(
    page_title="NVCB Prediction Tool",
    page_icon="🩺",
    layout="wide",
)

st.markdown("""
<style>
    .stApp { background-color: #f0f4f8; }
    h1 { color: #1a365d; font-family: Georgia, serif; }
    h3 { color: #2d3748; }
    .bleeder-box {
        background-color: #fff5f5;
        border-left: 6px solid #e53e3e;
        padding: 20px; border-radius: 8px; margin: 10px 0;
    }
    .nobleed-box {
        background-color: #f0fff4;
        border-left: 6px solid #38a169;
        padding: 20px; border-radius: 8px; margin: 10px 0;
    }
    .info-box {
        background-color: #ebf8ff;
        border-left: 4px solid #3182ce;
        padding: 12px 16px; border-radius: 6px;
        font-size: 0.9em; color: #2c5282;
    }
    .metric-box {
        background: white; padding: 16px; border-radius: 8px;
        text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


# ── STEP 8a : Load saved model, scaler, and metadata ─────────────────────────
# The model is loaded ONCE and cached.
# It is NEVER re-trained here — it is exactly the model validated in the notebook.

@st.cache_resource(show_spinner="Loading NVCB model …")
def load_model():
    model  = joblib.load("nvcb_gbm_model.pkl")
    scaler = joblib.load("nvcb_scaler.pkl")
    with open("nvcb_meta.json") as f:
        meta = json.load(f)
    return model, scaler, meta

try:
    gbm, scaler, meta = load_model()
    threshold    = meta["threshold"]
    all_features = meta["all_features"]
    top9         = meta["top9_features"]   # 9 features shown in UI (DIC excluded)
    medians      = meta["feature_medians"]
except FileNotFoundError:
    st.error(
        "Model files not found! Make sure these files are in the same folder:\n"
        "• nvcb_gbm_model.pkl\n• nvcb_scaler.pkl\n• nvcb_meta.json"
    )
    st.stop()


# Header

st.title("🩺 NVCB Prediction Tool")
st.markdown("**Non-Variceal Coagulopathic Bleeding in Acute Decompensated Cirrhosis**")
st.markdown(
    f"*ILBS · GBM Model · AUC {meta.get('model_auc', '—')} · "
    f"Sensitivity {meta.get('model_sensitivity', '—')} · "
    f"Threshold {threshold:.3f} (tuned on validation set)*"
)
st.divider()

with st.sidebar:
    st.header("Model Info")
    st.markdown(f"""
    **Model:** Gradient Boosting (GBM)

    **Why GBM?**
    - Highest AUC
    - Best sensitivity for bleeders
    - Validated on held-out test set before saving

    **Threshold:** `{threshold:.3f}`
    *(tuned on validation set for ≥80% sensitivity)*

    **Input:** Top-9 most important features
    """)
    st.divider()
    st.markdown("""
    <div class='info-box'>
    This tool is for clinical decision support only.
    All predictions must be reviewed by a qualified physician.
    </div>
    """, unsafe_allow_html=True)


# Model performance summary

st.subheader("Model Performance (Test Set — validated before deployment)")
c1, c2, c3, c4 = st.columns(4)
for col, label, value in zip(
    [c1, c2, c3, c4],
    ["AUC", "Sensitivity", "Specificity", "Accuracy"],
    [
        str(meta.get("model_auc",         "—")),
        str(meta.get("model_sensitivity", "—")),
        str(meta.get("model_specificity", "—")),
        "See results file",
    ],
):
    with col:
        st.markdown(f"""
        <div class='metric-box'>
            <h2 style='margin:0; color:#1a365d'>{value}</h2>
            <p style='margin:0; color:#718096'>{label}</p>
        </div>
        """, unsafe_allow_html=True)

st.divider()


# ── STEP 8b : Patient input — top-10 features only ───────────────────────────
# We show only the top-10 most important features to the clinician.
# For all other features, we use the TRAINING SET MEDIAN as a safe default
# (not 0, which would be an unrealistic/extreme value for most features).

st.subheader("📋 Enter Patient Values  (Top-9 Features)")
st.markdown(
    f"Fill in the **top {len(top9)} most important features** identified by the GBM model. "

)

# Build dynamic input widgets based on top-9 feature list
# Labels and units for known features
FEATURE_META = {
    "EXTEMA10"          : ("EXTEMA10  — EXTEM amplitude at 10 min (mm)",       0.0,  100.0,   45.0,  0.1),
    "EXTEMCFT"          : ("EXTEMCFT  — EXTEM clot formation time (sec)",       0.0, 1000.0,  120.0,  1.0),
    "EXTEMMCF"          : ("EXTEMMCF  — EXTEM max clot firmness (mm)",          0.0,  100.0,   55.0,  0.1),
    "EXTEMMCFt"         : ("EXTEMMCFt — EXTEM max clot firmness time (sec)",    0.0, 3000.0,    7.0,  0.1),
    "EXTEMCT"           : ("EXTEMCT   — EXTEM clotting time (sec)",             0.0, 1000.0,   80.0,  1.0),
    "EXTEMAUC"          : ("EXTEMAUC  — EXTEM area under curve (AU)",                0.0,20000.0, 5000.0, 10.0),
    "EXTEMACF"          : ("EXTEMACF  — EXTEM amplitude at clot formation (mm)",     0.0,  100.0,   30.0,  0.1),
    "EXTEMMAXV"         : ("EXTEMMAXV — EXTEM max velocity (mm/sec)",                    0.0,  500.0,   10.0,  0.1),
    "FIBTEMLI30"        : ("FIBTEMLI30 — FIBTEM lysis index at 30 min (%)",     0.0,  100.0,    5.0,  0.1),
    "FIBTEMA10"         : ("FIBTEMA10 — FIBTEM amplitude at 10 min (mm)",       0.0,   80.0,   12.0,  0.1),
    "FIBTEMMCF"         : ("FIBTEMMCF — FIBTEM max clot firmness (mm)",         0.0,   80.0,    6.0,  0.1),
    "FIBTEMAUC"         : ("FIBTEMAUC — FIBTEM area under curve (AU)",               0.0,10000.0,  800.0, 10.0),
    "Hb"                : ("Hb        — Haemoglobin (g/dL)",                    0.0,   20.0,    7.5,  0.1),
    "PLT"               : ("PLT       — Platelet count (×10³/µL)",              0.0,  800.0,   60.0,  1.0),
    "INR"               : ("INR       — International Normalised Ratio",         0.5,   15.0,    2.1,  0.1),
    "Albumin"           : ("Albumin   — Serum albumin (g/dL)",                  0.0,    6.0,    2.4,  0.1),
    "Creatinine"        : ("Creatinine — Serum creatinine (mg/dL)",             0.0,   20.0,    1.8,  0.1),
    "Bilirubin"         : ("Bilirubin — Total bilirubin (mg/dL)",               0.0,   80.0,    5.2,  0.1),
    "Fibrinogen"        : ("Fibrinogen — Plasma fibrinogen (mg/dL)",            0.0, 1000.0,  120.0,  1.0),
    "TLC"               : ("TLC       — Total leucocyte count (×10³/µL)",       0.0,  100.0,    9.5,  0.1),
    "Urea"              : ("Urea      — Blood urea nitrogen (mg/dL)",           0.0,  300.0,   45.0,  1.0),
    "MELD"              : ("MELD      — MELD score",                            6.0,   40.0,   22.0,  1.0),
    "CTP"               : ("CTP       — Child-Turcotte-Pugh score",             5.0,   15.0,   10.0,  1.0),
}
# Binary features (checkbox) — DIC is excluded from UI (handled via training-median)
BINARY_FEATURES = {
    "AKI"               : "AKI — Acute Kidney Injury present",
    "PriorDecompensation": "Prior Decompensation (1=Decompensated, 0=Compensated)",
    "PortalHypertension": "Portal Hypertension present",
    "Sepsis"            : "Sepsis present",
    "DM2"               : "Diabetes Mellitus (Type 2)",
}

col1, col2 = st.columns(2)
entered_values = {}
half = len(top9) // 2

for i, feat in enumerate(top9):
    target_col = col1 if i < half else col2
    with target_col:
        if feat in BINARY_FEATURES:
            # Binary feature → checkbox
            val = st.checkbox(BINARY_FEATURES[feat], value=False)
            entered_values[feat] = int(val)
        elif feat in FEATURE_META:
            label, mn, mx, default, step = FEATURE_META[feat]
            val = st.number_input(label, min_value=mn, max_value=mx,
                                  value=default, step=step)
            entered_values[feat] = val
        else:
            # Feature not in our label dictionary — show a generic input
            val = st.number_input(f"{feat}", value=float(medians.get(feat, 0.0)))
            entered_values[feat] = val

st.divider()

predict_clicked = st.button("Predict NVCB Risk", type="primary", use_container_width=True)

if predict_clicked:

    # ── STEP 8c : Build full feature vector ──────────────────────────────────
    # Start from TRAINING MEDIANS (safe realistic defaults for all features).
    # Then override with what the clinician actually entered (top-10).
    # This avoids the bug where non-entered features were set to 0.

    patient_dict = medians.copy()            # safe baseline = training medians
    patient_dict.update(entered_values)      # override with clinician values

    patient_df = pd.DataFrame([patient_dict])[all_features]
    patient_sc = scaler.transform(patient_df)

    prob       = gbm.predict_proba(patient_sc)[0, 1]
    is_bleeder = prob >= threshold

    st.subheader("GBM Prediction Result")
    st.markdown(f"**Probability of NVCB (bleeder):** `{prob:.3f}`  |  "
                f"**Threshold:** `{threshold:.3f}`")

    if is_bleeder:
        st.markdown("""
        <div class='bleeder-box'>
            <h2 style='color:#e53e3e; margin:0'>🔴 NVCB Likely</h2>
            <p style='margin:8px 0 0 0'>
                The GBM model predicts this patient is at <b>high risk</b> of
                Non-Variceal Coagulopathic Bleeding. Consider clinical intervention.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='nobleed-box'>
            <h2 style='color:#38a169; margin:0'>🟢 NVCB Unlikely</h2>
            <p style='margin:8px 0 0 0'>
                The GBM model predicts this patient is at <b>low risk</b> of
                Non-Variceal Coagulopathic Bleeding.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Show entered values
    st.subheader("Values Entered (Top-9 Features)")
    summary_rows = []
    for feat, val in entered_values.items():
        unit = ""
        if feat in FEATURE_META:
            unit = FEATURE_META[feat][0].split("(")[-1].replace(")", "").strip() if "(" in FEATURE_META[feat][0] else ""
        elif feat in BINARY_FEATURES:
            unit = "1=Yes / 0=No"
        summary_rows.append({"Feature": feat, "Value": val, "Unit": unit})

    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

st.divider()
st.markdown("""
<small>
 <b>Disclaimer:</b> This tool is for research and clinical decision support only.
It does not replace clinical judgement. All predictions must be reviewed by a qualified physician.<br><br>
Developed at ILBS · GBM Model · Validated on held-out test set · n = 6,992 patients
</small>
""", unsafe_allow_html=True)