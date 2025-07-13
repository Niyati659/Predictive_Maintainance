import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from sklearn.preprocessing import StandardScaler

# Load trained model
model = joblib.load("xgb_model.pkl")

# Load and clean column names from file
with open("columns.txt", "r") as f:
    columns_raw = [line.strip() for line in f.readlines()]

# Extract valid feature names (remove comments, quotes, etc.)
columns = []
for line in columns_raw:
    match = re.match(r"['\"]?(\w+)", line)
    if match:
        columns.append(match.group(1))

# Define required features used during training (in exact order)
required_features = [
    'BPR', 'NRc', 'NRf', 'Nc', 'Nf', 'Nf_dmd',
    'P15', 'P2', 'P30', 'PCNfR_dmd', 'Ps30', 'T2',
    'T24', 'T30', 'T50', 'W31', 'W32', 'epr',
    'farB', 'htBleed', 'op_setting_1', 'op_setting_2',
    'op_setting_3', 'phi'
]


# Load scaler parameters if available
try:
    mean = np.load("scaler_mean.npy")
    scale = np.load("scaler_scale.npy")
    scaler = StandardScaler()
    scaler.mean_ = mean
    scaler.scale_ = scale
    use_scaler = True
except FileNotFoundError:
    use_scaler = False

# Streamlit UI setup
st.set_page_config(page_title="🔧 Predictive Maintenance App", layout="centered")
st.title("🔧 Predictive Maintenance — RUL Prediction")
st.markdown("Enter sensor values below to predict Remaining Useful Life (RUL).")

# Input fields
inputs = {}
for col in columns:
    inputs[col] = st.number_input(f"Enter value for `{col}`", value=0.0)

# Prediction logic
if st.button("🔍 Predict RUL"):
    input_df = pd.DataFrame([inputs])

    # Keep only required features in correct order
    input_df = input_df[required_features]

    # Scale if scaler is available
    if use_scaler:
        input_df = scaler.transform(input_df)

    # Predict RUL
    predicted_rul = model.predict(input_df)[0]

    # Show result
    st.success(f"✅ Predicted RUL: **{predicted_rul:.2f} cycles**")

    # Alert system
    if predicted_rul < 20:
        st.error("⚠️ Maintenance Alert: Immediate maintenance required!")
    elif predicted_rul < 50:
        st.warning("⚠️ Schedule maintenance within the next 10–15 cycles.")
    else:
        st.info("✅ System healthy. No immediate maintenance required.")
