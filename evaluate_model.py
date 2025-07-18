import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Load trained XGBoost model
model = joblib.load("xgb_model.pkl")

# Get feature columns from training (saved earlier)
feature_cols = model.get_booster().feature_names

# Function to plot predictions
def plot_predictions(true, predicted, dataset_id):
    plt.figure(figsize=(8, 5))
    plt.plot(true, label="True RUL", marker='o')
    plt.plot(predicted, label="Predicted RUL", marker='x')
    plt.title(f"Predicted vs True RUL – {dataset_id}")
    plt.xlabel("Engine Index")
    plt.ylabel("Remaining Useful Life (RUL)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"rul_plot_{dataset_id}.png")  # Saves the plot as image
    plt.show()


# Loop through each test set
for file_id in ['FD001', 'FD002', 'FD003', 'FD004']:
    print(f" Evaluating on test dataset {file_id}")

    # Load test and RUL files
    test_df = pd.read_csv(f"test_{file_id}.csv")
    rul_df = pd.read_csv(f"RUL_{file_id}.csv")

    # Use last cycle of each engine
    last_cycles = test_df.groupby("unit_number").last().reset_index()

    # Drop non-feature columns
    test_features = last_cycles.drop(columns=["unit_number", "time_in_cycles"])

    # Add missing columns with 0s
    for col in feature_cols:
        if col not in test_features.columns:
            test_features[col] = 0

    # Reorder columns to match training
    test_features = test_features[feature_cols]

    # Predict
    predictions = model.predict(test_features)

    # Evaluate
    true_rul = rul_df["RUL"].values
    mae = mean_absolute_error(true_rul, predictions)
    rmse = np.sqrt(mean_squared_error(true_rul, predictions))

    print(f"MAE: {mae:.2f}")
    print(f" RMSE: {rmse:.2f}")
    plot_predictions(true_rul, predictions, file_id)



