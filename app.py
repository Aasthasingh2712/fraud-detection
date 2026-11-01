import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from tensorflow.keras.models import load_model

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Radar — Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

# ==============================
# GLOBAL STYLE (Stripe Radar inspired)
# ==============================
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"], .stMarkdown, .stText, p, span, div {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
:root {
    --radar-purple: #635BFF;
    --radar-purple-dark: #4B44D6;
    --radar-bg: #0A0E27;
    --radar-danger: #DF1B41;
    --radar-success: #0A8A5F;
    --radar-warning: #FF9F1C;
    --radar-card: #FFFFFF;
    --radar-border: #E3E8EF;
    --radar-text: #1A1F36;
    --radar-muted: #697386;
}
.stApp {
    background: #F6F8FA;
}
/* Hide default streamlit chrome */
#MainMenu, footer, header {visibility: hidden;}
/* Main container spacing */
.main .block-container {
    padding: 2.5rem 3rem 3rem 3rem;
    max-width: 1200px;
}
/* Top banner */
.radar-header {
    background: linear-gradient(135deg, #0A0E27 0%, #1A1F5C 50%, #635BFF 100%);
    border-radius: 16px;
    padding: 2.2rem 2.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 10px 30px rgba(99, 91, 255, 0.25);
}
.radar-header h1 {
    color: #FFFFFF !important;
    font-size: 2rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.5px;
}
.radar-header p {
    color: #C9CCFB;
    font-size: 0.95rem;
    margin-top: 0.5rem;
    margin-bottom: 0;
    font-weight: 400;
}
.radar-badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    color: #fff;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-bottom: 0.8rem;
    letter-spacing: 0.3px;
}
/* Headings */
h2, h3 {
    color: var(--radar-text) !important;
    font-weight: 700 !important;
    letter-spacing: -0.3px;
}
/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0A0E27;
}
section[data-testid="stSidebar"] * {
    color: #E3E8EF !important;
}
section[data-testid="stSidebar"] .stSelectbox label {
    color: #9CA3D4 !important;
    font-weight: 600;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.1);
}
/* Cards */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--radar-card);
    border-radius: 12px;
    border: 1px solid var(--radar-border);
}
/* Metrics */
div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid var(--radar-border);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
div[data-testid="stMetric"] label {
    color: var(--radar-muted) !important;
    font-weight: 600;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.4px;
}
div[data-testid="stMetricValue"] {
    color: var(--radar-text) !important;
    font-weight: 800 !important;
    font-size: 1.8rem !important;
}
/* Buttons */
.stButton>button {
    background: var(--radar-purple);
    color: white;
    font-weight: 600;
    font-size: 0.9rem;
    border-radius: 8px;
    border: none;
    padding: 0.6rem 1.8rem;
    transition: all 0.15s ease;
    box-shadow: 0 2px 8px rgba(99, 91, 255, 0.3);
}
.stButton>button:hover {
    background: var(--radar-purple-dark);
    box-shadow: 0 4px 14px rgba(99, 91, 255, 0.4);
    transform: translateY(-1px);
}
/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: #EEF0F5;
    padding: 6px;
    border-radius: 10px;
}
.stTabs [data-baseweb="tab"] {
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--radar-muted);
    border-radius: 8px;
    padding: 8px 18px;
}
.stTabs [aria-selected="true"] {
    background: #FFFFFF !important;
    color: var(--radar-purple) !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
}
/* Alerts (fraud / normal results) */
div[data-testid="stAlert"] {
    border-radius: 10px;
    font-weight: 600;
    padding: 1rem 1.2rem;
}
/* Dataframes */
div[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid var(--radar-border);
}
/* Number inputs */
.stNumberInput input {
    border-radius: 8px;
    border: 1px solid var(--radar-border);
}
/* File uploader */
div[data-testid="stFileUploaderDropzone"] {
    background: #FAFBFC;
    border: 1.5px dashed #C9CDDB;
    border-radius: 10px;
}
/* Risk score pill */
.risk-pill {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.3px;
}
.risk-high { background: #FDE7EA; color: var(--radar-danger); }
.risk-low { background: #E3F5EE; color: var(--radar-success); }
span[data-testid="stIconMaterial"], .material-icons, .material-symbols-rounded {
    font-family: "Material Symbols Rounded", "Material Icons" !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    background-color: #635BFF !important;
}
.stTabs [data-baseweb="tab-border"] {
    background-color: transparent !important;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# LOAD MODELS AND SCALERS
# ==============================
@st.cache_resource
def load_all_models():
    scaler_amount = joblib.load('models/scaler_amount.pkl')
    scaler_time = joblib.load('models/scaler_time.pkl')
    iso_forest = joblib.load('models/isolation_forest_final.pkl')
    rf_model = joblib.load('models/random_forest_final.pkl')
    autoencoder = load_model('models/autoencoder_final.keras')
    with open('models/autoencoder_threshold.json', 'r') as f:
        ae_threshold = json.load(f)['threshold']
    return scaler_amount, scaler_time, iso_forest, rf_model, autoencoder, ae_threshold

scaler_amount, scaler_time, iso_forest, rf_model, autoencoder, ae_threshold = load_all_models()

FEATURE_COLUMNS = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

# ==============================
# PREDICTION FUNCTIONS
# ==============================
def predict_isolation_forest(X):
    pred = iso_forest.predict(X)
    pred = np.where(pred == -1, 1, 0)
    score = -iso_forest.decision_function(X)
    return pred, score

def predict_autoencoder(X):
    X_pred = autoencoder.predict(X, verbose=0)
    mse = np.mean(np.power(X - X_pred, 2), axis=1)
    pred = np.where(mse > ae_threshold, 1, 0)
    return pred, mse

def predict_random_forest(X):
    pred = rf_model.predict(X)
    score = rf_model.predict_proba(X)[:, 1]
    return pred, score

def run_prediction(model_choice, X):
    if model_choice == "Isolation Forest (Anomaly Detection)":
        return predict_isolation_forest(X)
    elif model_choice == "Autoencoder (Anomaly Detection)":
        return predict_autoencoder(X)
    else:
        return predict_random_forest(X)

# ==============================
# SIDEBAR
# ==============================
with st.sidebar:
    st.markdown("### 🛡️ Radar")
    st.caption("Fraud Detection Engine")
    st.markdown("---")
    model_choice = st.selectbox(
        "Detection Model",
        ["Isolation Forest (Anomaly Detection)", "Autoencoder (Anomaly Detection)", "Random Forest (Supervised)"]
    )
    st.markdown("---")
    st.markdown("**How it works**")
    st.caption(
        "Anomaly detection models (Isolation Forest, Autoencoder) learn what "
        "normal transactions look like and flag anything unusual — useful for "
        "catching new fraud patterns."
    )
    st.caption(
        "Random Forest is trained directly on labeled fraud examples using "
        "SMOTE, giving higher precision on known fraud types."
    )

# ==============================
# HEADER BANNER
# ==============================
st.markdown("""
    <div class="radar-header">
        <span class="radar-badge">● LIVE MODEL</span>
        <h1>Credit Card Fraud Detection</h1>
        <p>Real-time transaction risk scoring using anomaly detection and supervised machine learning.</p>
    </div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Transaction Check", "Bulk Upload", "Model Performance"])

# ==============================
# TAB 1: SINGLE TRANSACTION CHECK
# ==============================
with tab1:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("#### Transaction Details")

        samples = pd.read_csv('data/sample_transactions.csv')
        st.session_state.setdefault('time_in', 0.0)
        st.session_state.setdefault('amount_in', 100.0)
        for i in range(1, 29):
            st.session_state.setdefault(f'v{i}', 0.0)

        def fill_sample(cls):
            row = samples[samples['Class'] == cls].sample(1).iloc[0]
            st.session_state['time_in'] = float(row['Time'])
            st.session_state['amount_in'] = float(row['Amount'])
            for i in range(1, 29):
                st.session_state[f'v{i}'] = float(row[f'V{i}'])

        b1, b2 = st.columns(2)
        b1.button("Load sample: Normal", on_click=fill_sample, args=(0,))
        b2.button("Load sample: Fraud", on_click=fill_sample, args=(1,))

        time_val = st.number_input("Time (seconds since first transaction)", key='time_in')
        amount_val = st.number_input("Amount ($)", min_value=0.0, key='amount_in')

        with st.expander("V1–V28 features (PCA components, required by the model)", expanded=False):
            v_values = []
            cols = st.columns(4)
            for i in range(1, 29):
                with cols[(i - 1) % 4]:
                    v = st.number_input(f"V{i}", key=f"v{i}", format="%.4f")
                    v_values.append(v)

        predict_clicked = st.button("Run Risk Check", type="primary")

    with right:
        st.markdown("#### Result")
        if predict_clicked:
            scaled_amount = scaler_amount.transform([[amount_val]])[0][0]
            scaled_time = scaler_time.transform([[time_val]])[0][0]

            row = [scaled_time] + v_values + [scaled_amount]
            X_input = pd.DataFrame([row], columns=FEATURE_COLUMNS)

            pred, score = run_prediction(model_choice, X_input)

            if pred[0] == 1:
                st.markdown('<span class="risk-pill risk-high">⚠ HIGH RISK — FLAGGED</span>', unsafe_allow_html=True)
                st.error(f"This transaction looks fraudulent according to **{model_choice}**.")
            else:
                st.markdown('<span class="risk-pill risk-low">✓ LOW RISK — CLEARED</span>', unsafe_allow_html=True)
                st.success(f"This transaction looks normal according to **{model_choice}**.")

            st.metric("Anomaly / Fraud Score", f"{score[0]:.4f}")
        else:
            st.info("Load a sample or enter details, then click **Run Risk Check**.")

# ==============================
# TAB 2: BULK CSV UPLOAD
# ==============================
with tab2:
    st.markdown("#### Upload Transactions")
    st.caption("CSV must contain columns: Time, V1–V28, Amount (Class column is optional, for evaluation).")

    uploaded_file = st.file_uploader("Drop CSV file here", type="csv", label_visibility="collapsed")

    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)

        has_class = 'Class' in df_upload.columns
        X_upload = df_upload[FEATURE_COLUMNS].copy()
        X_upload['Amount'] = scaler_amount.transform(X_upload[['Amount']])
        X_upload['Time'] = scaler_time.transform(X_upload[['Time']])

        pred, score = run_prediction(model_choice, X_upload)

        results_df = df_upload.copy()
        results_df['Prediction'] = np.where(pred == 1, 'Fraud', 'Normal')
        results_df['Score'] = score

        fraud_count = int(pred.sum())
        total_count = len(pred)

        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Transactions", f"{total_count:,}")
        m2.metric("Flagged as Fraud", f"{fraud_count:,}")
        m3.metric("Fraud Rate", f"{(fraud_count/total_count)*100:.2f}%")
        st.markdown("<br>", unsafe_allow_html=True)

        summary_cols = ['Time', 'Amount', 'Prediction', 'Score'] + (['Class'] if has_class else [])
        flagged_df = results_df[results_df['Prediction'] == 'Fraud']

        st.markdown("#### Flagged Transactions")
        st.dataframe(flagged_df[summary_cols].sort_values('Score', ascending=False), use_container_width=True, height=300, hide_index=True)

        with st.expander(f"View all predictions ({total_count:,} rows)"):
            st.dataframe(results_df[summary_cols], use_container_width=True, height=300, hide_index=True)

        csv_download = results_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Full Results (CSV)", csv_download, "fraud_predictions.csv", "text/csv")

        if has_class:
            from sklearn.metrics import classification_report, average_precision_score
            st.markdown("#### Evaluation (true labels provided)")
            report = classification_report(df_upload['Class'], pred, labels=[0, 1], output_dict=True, zero_division=0)
            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Precision", f"{report['1']['precision']:.2f}")
            e2.metric("Recall", f"{report['1']['recall']:.2f}")
            e3.metric("F1-Score", f"{report['1']['f1-score']:.2f}")
            e4.metric("PR-AUC", f"{average_precision_score(df_upload['Class'], score):.3f}")

# ==============================
# TAB 3: MODEL COMPARISON
# ==============================
with tab3:
    st.markdown("#### Model Performance Comparison")
    st.caption("Evaluated on a held-out test set (30% split, stratified).")

    comparison_df = pd.DataFrame({
        'Model': ['Isolation Forest', 'Isolation Forest (Tuned)', 'LOF', 'Autoencoder', 'Random Forest + SMOTE'],
        'Type': ['Anomaly Detection', 'Anomaly Detection', 'Anomaly Detection', 'Anomaly Detection (Deep Learning)', 'Supervised Learning'],
        'Recall': [0.27, 0.57, 0.00, 0.80, 0.75],
        'Precision': [0.26, 0.18, 0.00, 0.12, 0.91],
        'PR-AUC': [0.133, 0.194, 0.002, 0.518, 0.811]
    })

    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.bar_chart(comparison_df.set_index('Model')['PR-AUC'], color="#635BFF")

    st.markdown("""
    <div style="background:#FFFFFF; border:1px solid #E3E8EF; border-radius:12px; padding:1.5rem 1.8rem; margin-top:1rem;">
    <b>Key takeaway:</b> Anomaly detection methods (Isolation Forest, Autoencoder) don't need fraud
    labels to train and can catch previously unseen fraud patterns, making them valuable as a
    first line of defense. Supervised learning (Random Forest + SMOTE) achieves much higher
    accuracy on known fraud patterns because it learns directly from labeled examples. In
    production, banks typically combine both approaches for layered protection.
    </div>
    """, unsafe_allow_html=True)
