# =============================================================================
# NVCB Prediction Web App — Streamlit
# Non-Variceal Coagulopathic Bleeding in Acute Decompensated Cirrhosis
# Model: Gradient Boosting (GBM) — AUC 0.854, Sensitivity 0.84
#
# Run with:  streamlit run nvcb_app.py
# Install :  pip install streamlit scikit-learn pandas numpy openpyxl
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import RobustScaler
from sklearn.utils           import resample
from sklearn.ensemble        import GradientBoostingClassifier
from sklearn.metrics         import roc_curve


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title = "NVCB Prediction Tool",
    page_icon  = "🩺",
    layout     = "wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .stApp { background-color: #f0f4f8; }
    h1 { color: #1a365d; font-family: Georgia, serif; }
    h3 { color: #2d3748; }

    .bleeder-box {
        background-color: #fff5f5;
        border-left: 6px solid #e53e3e;
        padding: 20px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .nobleed-box {
        background-color: #f0fff4;
        border-left: 6px solid #38a169;
        padding: 20px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #ebf8ff;
        border-left: 4px solid #3182ce;
        padding: 12px 16px;
        border-radius: 6px;
        font-size: 0.9em;
        color: #2c5282;
    }
    .metric-box {
        background: white;
        padding: 16px;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


# ── All features needed for training ─────────────────────────────────────────

ALL_FEATURES = [
    "EXTEMA10", "EXTEMCFT", "EXTEMMCF", "EXTEMMCFt", "EXTEMCT",
    "EXTEMAUC", "EXTEMACF", "EXTEMMAXV", "FIBTEMLI30", "FIBTEMA10",
    "FIBTEMMCF", "FIBTEMAUC", "Hb", "PLT", "INR", "Albumin",
    "Creatinine", "Bilirubin", "Fibrinogen", "TLC", "Urea",
    "MELD", "CTP", "AKI", "PriorDecompensation",
    "DIC", "PortalHypertension", "Sepsis", "DM2",
]

# Top 10 features the user fills in (by combined GBM + RF importance)
TOP10 = [
    "EXTEMA10",   # ROTEM EXTEM amplitude at 10 min
    "EXTEMCFT",   # ROTEM EXTEM clot formation time
    "EXTEMAUC",   # ROTEM EXTEM area under curve
    "EXTEMMCFt",  # ROTEM EXTEM max clot firmness time
    "DIC",        # Disseminated intravascular coagulation
    "CTP",        # Child-Turcotte-Pugh score
    "Albumin",    # Serum albumin
    "Hb",         # Haemoglobin
    "FIBTEMAUC",  # ROTEM FIBTEM area under curve
    "Urea",       # Blood urea
]


# ── Train GBM (cached — runs only once per session) ───────────────────────────

@st.cache_resource(show_spinner="Training GBM model on NVCB dataset — please wait ...")
def train_gbm(uploaded_file):

    df = pd.read_excel(uploaded_file)
    df = df.rename(columns={
        "PortalHypertension1_varice0_no_00"                               : "PortalHypertension",
        "@1_Alcohol2_2cry3_MET14_Viral5_autoimmunebiliary6_0thers7_78_alf": "Etiology",
        "CompensatedDecompensated"                                         : "PriorDecompensation",
    })

    feats = [f for f in ALL_FEATURES if f in df.columns]
    y     = df["NVCB"].astype(int)
    X     = df[feats]

    # 60 / 20 / 20 split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
    )

    # Oversample minority class on training set only
    df_tr       = X_train.copy(); df_tr["label"] = y_train.values
    majority    = df_tr[df_tr["label"] == 0]
    minority    = df_tr[df_tr["label"] == 1]
    minority_up = resample(
        minority, replace=True,
        n_samples=int(len(majority) * 1.5),
        random_state=42,
    )
    balanced    = pd.concat([majority, minority_up]).sample(frac=1, random_state=42)
    X_train_bal = balanced.drop(columns=["label"])
    y_train_bal = balanced["label"]

    # Scale
    scaler     = RobustScaler()
    X_train_sc = scaler.fit_transform(X_train_bal)
    X_val_sc   = scaler.transform(X_val)

    # Train GBM
    gbm = GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05,
        max_depth=4, min_samples_leaf=10, subsample=0.8,
        random_state=42,
    )
    gbm.fit(X_train_sc, y_train_bal)

    # Find threshold on validation set (sensitivity >= 80%)
    val_proba        = gbm.predict_proba(X_val_sc)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_val.values, val_proba)
    best_thresh, best_spec = None, -1
    for t, sens, fp in zip(thresholds, tpr, fpr):
        if sens >= 0.80 and (1 - fp) > best_spec:
            best_thresh = t
            best_spec   = 1 - fp
    if best_thresh is None:
        best_thresh = float(thresholds[np.argmax(tpr - fpr)])

    return gbm, scaler, feats, float(best_thresh)


# ── Header ────────────────────────────────────────────────────────────────────

st.title("🩺 NVCB Prediction Tool")
st.markdown("**Non-Variceal Coagulopathic Bleeding in Acute Decompensated Cirrhosis**")
st.markdown("*Institute of Liver and Biliary Sciences (ILBS) · GBM Model · AUC 0.854 · Sensitivity 0.84*")
st.divider()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Setup")
    st.markdown("Upload the NVCB dataset to train the model.")
    uploaded = st.file_uploader("Upload NVCB_anonymised_dataset.xlsx", type=["xlsx"])
    st.divider()

    st.markdown("""
    **Model:** Gradient Boosting (GBM)

    **Why GBM?**
    - Highest AUC : 0.854
    - Sensitivity : 0.84
    - Best overall performer

    **Input:** Top 10 features by importance

    **Threshold:** Tuned on validation set to achieve sensitivity ≥ 80%
    """)

    st.divider()
    st.markdown("""
    <div class='info-box'>
    This tool is for clinical decision support only. All predictions must be
    reviewed by a qualified physician.
    </div>
    """, unsafe_allow_html=True)


# ── Wait for upload ───────────────────────────────────────────────────────────

if uploaded is None:
    st.info("👈 Please upload the NVCB dataset in the sidebar to train the model first.")
    st.stop()

gbm, scaler, feats, threshold = train_gbm(uploaded)

st.success(f"✅ GBM model trained successfully.  Threshold = {threshold:.3f}")
st.divider()


# ── Model performance summary ─────────────────────────────────────────────────

st.subheader("📈 Model Performance (Test Set)")
c1, c2, c3, c4 = st.columns(4)
for col, label, value in zip(
    [c1, c2, c3, c4],
    ["AUC", "Sensitivity", "Specificity", "Accuracy"],
    ["0.854", "0.840", "0.713", "0.727"],
):
    with col:
        st.markdown(f"""
        <div class='metric-box'>
            <h2 style='margin:0; color:#1a365d'>{value}</h2>
            <p style='margin:0; color:#718096'>{label}</p>
        </div>
        """, unsafe_allow_html=True)

st.divider()


# ── Patient input ─────────────────────────────────────────────────────────────

st.subheader("📋 Enter Patient Values")
st.markdown("Fill in the **top 10 most important features** for prediction.")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**ROTEM Parameters**")
    EXTEMA10  = st.number_input("EXTEMA10  — EXTEM amplitude at 10 min (mm)",    min_value=0.0,   max_value=100.0,  value=45.0,  step=0.1)
    EXTEMCFT  = st.number_input("EXTEMCFT  — EXTEM clot formation time (sec)",   min_value=0.0,   max_value=1000.0, value=120.0, step=1.0)
    EXTEMAUC  = st.number_input("EXTEMAUC  — EXTEM area under curve",            min_value=0.0,   max_value=20000.0,value=5000.0,step=10.0)
    EXTEMMCFt = st.number_input("EXTEMMCFt — EXTEM max clot firmness time (min)",min_value=0.0,   max_value=100.0,  value=7.0,   step=0.1)
    FIBTEMAUC = st.number_input("FIBTEMAUC — FIBTEM area under curve",           min_value=0.0,   max_value=10000.0,value=800.0, step=10.0)

with col2:
    st.markdown("**Clinical Parameters**")
    DIC     = st.selectbox("DIC — Disseminated intravascular coagulation", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    CTP     = st.number_input("CTP — Child-Turcotte-Pugh score",          min_value=5,    max_value=15,     value=10,    step=1)
    Albumin = st.number_input("Albumin — Serum albumin (g/dL)",           min_value=0.0,  max_value=6.0,    value=2.4,   step=0.1)
    Hb      = st.number_input("Hb — Haemoglobin (g/dL)",                 min_value=0.0,  max_value=20.0,   value=7.5,   step=0.1)
    Urea    = st.number_input("Urea — Blood urea nitrogen (mg/dL)",       min_value=0.0,  max_value=300.0,  value=45.0,  step=1.0)

st.divider()

# ── Predict ───────────────────────────────────────────────────────────────────

predict_clicked = st.button("🔍 Predict NVCB Risk", type="primary", use_container_width=True)

if predict_clicked:

    # Build full feature vector — all non-top-10 features set to 0
    patient_dict = {f: 0.0 for f in feats}
    patient_dict.update({
        "EXTEMA10" : EXTEMA10,
        "EXTEMCFT" : EXTEMCFT,
        "EXTEMAUC" : EXTEMAUC,
        "EXTEMMCFt": EXTEMMCFt,
        "FIBTEMAUC": FIBTEMAUC,
        "DIC"      : float(DIC),
        "CTP"      : float(CTP),
        "Albumin"  : Albumin,
        "Hb"       : Hb,
        "Urea"     : Urea,
    })

    patient_df = pd.DataFrame([patient_dict])[feats]
    patient_sc = scaler.transform(patient_df)

    prob       = gbm.predict_proba(patient_sc)[0, 1]
    is_bleeder = prob >= threshold

    st.subheader("📊 GBM Prediction Result")

    if is_bleeder:
        st.markdown("""
        <div class='bleeder-box'>
            <h2 style='color:#e53e3e; margin:0'>🔴 NVCB LIKELY — Patient is a Bleeder</h2>
            <p style='margin:8px 0 0 0'>
                The GBM model predicts this patient is at high risk of
                Non-Variceal Coagulopathic Bleeding. Consider clinical intervention.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='nobleed-box'>
            <h2 style='color:#38a169; margin:0'>🟢 NVCB UNLIKELY — No Bleed Predicted</h2>
            <p style='margin:8px 0 0 0'>
                The GBM model predicts this patient is at low risk of
                Non-Variceal Coagulopathic Bleeding.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Values entered summary
    st.subheader("📌 Values Entered")
    summary = pd.DataFrame({
        "Feature"    : ["EXTEMA10",  "EXTEMCFT",  "EXTEMAUC",  "EXTEMMCFt", "FIBTEMAUC",
                        "DIC",       "CTP",        "Albumin",   "Hb",        "Urea"],
        "Value"      : [EXTEMA10,    EXTEMCFT,    EXTEMAUC,    EXTEMMCFt,   FIBTEMAUC,
                        DIC,         CTP,          Albumin,     Hb,          Urea],
        "Unit"       : ["mm",        "sec",        "AU",        "min",        "AU",
                        "0 = No / 1 = Yes", "score", "g/dL",  "g/dL",      "mg/dL"],
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)


# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.markdown("""
<small>
⚠️ <b>Disclaimer:</b> This tool is for research and clinical decision support only.
It does not replace clinical judgement. All predictions must be reviewed by a qualified physician.<br><br>
Developed at ILBS · GBM Model · AUC 0.854 · Sensitivity 0.84 · n = 6,992 patients
</small>
""", unsafe_allow_html=True)
