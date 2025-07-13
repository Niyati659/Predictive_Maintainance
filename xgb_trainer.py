import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import joblib

from data_handling import computeRUL, prepare_features_and_target

# Load and merge all training datasets
datasets = []
for file_id in ['FD001', 'FD002', 'FD003', 'FD004']:
    df = pd.read_csv(f"train_{file_id}.csv")
    df = computeRUL(df)
    datasets.append(df)

full_df = pd.concat(datasets, ignore_index=True)

# Prepare features and labels
X, y, _ = prepare_features_and_target(full_df, scale=False)

# Save feature/column names
with open("columns.txt", "w") as f:
    for col in X.columns:
        f.write(f"{col}\n")
print("✅ Column names saved to 'columns.txt'")

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
print(" Training XGBoost model")
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb.fit(X_train, y_train)

# Evaluation
val_preds = xgb.predict(X_val)
mae = mean_absolute_error(y_val, val_preds)
rmse = np.sqrt(mean_squared_error(y_val, val_preds))

print(" Validation Performance:")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")

# Save the model
joblib.dump(xgb, "xgb_model.pkl")
print(" XGBoost model saved as 'xgb_model.pkl'")