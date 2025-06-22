import pandas as pd
import numpy as np
import joblib
from data_handling import prepare_features_and_target

# Load the trained model
model = joblib.load("xgb_model.pkl")

# List of required sensor/setting names (as per your final feature set)
required_features = [
    'op_setting_1', 'op_setting_2', 'op_setting_3',
    'T24', 'T30', 'T50', 'P15', 'P30',
    'Nf', 'Nc', 'Ps30', 'phi', 'NRf', 'NRc',
    'BPR', 'htBleed', 'W31', 'W32', 'T2',
    'P2', 'epr', 'farB', 'Nf_dmd', 'PCNfR_dmd'
]

# Take user input for each sensor
print(" Enter the sensor and operational setting values below:")
user_input = {}
for feature in required_features:
    while True:
        try:
            value = float(input(f"{feature}: "))
            user_input[feature] = value
            break
        except ValueError:
            print("Please enter a valid number.")

# Create a DataFrame from user input
input_df = pd.DataFrame([user_input])

# Prepare features (scaling inside function if needed)
X_input, _, _ = prepare_features_and_target(input_df, scale=True, include_target=False)


# Predict RUL
predicted_rul = model.predict(X_input)[0]
print(f"Predicted Remaining Useful Life (RUL): {predicted_rul:.2f} cycles")
 # Maintenance Scheduler Logic
 
if predicted_rul < 20:
    print(" Maintenance Alert: Immediate maintenance required!")
elif predicted_rul < 50:
    print(f" Schedule maintenance within the next 10–15 cycles.")
else:
    print(" System healthy. No immediate maintenance required.")
