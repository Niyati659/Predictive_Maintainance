import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from data_handling import computeRUL, prepare_features_and_target

# Load & combine all 4 training datasets 
datasets = []
for file_id in ['FD001', 'FD002', 'FD003', 'FD004']:
    df = pd.read_csv(f"train_{file_id}.csv")
    df = computeRUL(df)
    df["dataset_id"] = file_id  # Optional: for tracking origin
    datasets.append(df)

# Combine all datasets
full_df = pd.concat(datasets, ignore_index=True)

# Prepare data (RandomForest & XGBoost - no scaling)
X, y, _ = prepare_features_and_target(full_df, scale=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
print("\n🟡 Training Random Forest...")
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

# Train XGBoost
print("\n🟠 Training XGBoost...")
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)

# Prepare scaled data for SVM
X_scaled, y_scaled, scaler = prepare_features_and_target(full_df, scale=True)

# Fix: Remove any NaNs after scaling
X_scaled = X_scaled.dropna()
y_scaled = y_scaled[X_scaled.index] 
X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Train SVM
print("\n🔵 Training SVM...")
svm = SVR(kernel='rbf')
svm.fit(X_train_svm, y_train_svm)
svm_preds = svm.predict(X_test_svm)

# Evaluation function
def evaluate(model_name, y_true, y_pred):
    print(f"\n✅ {model_name} Results")
    print("MAE :", round(mean_absolute_error(y_true, y_pred), 2))
    print("RMSE:", round(np.sqrt(mean_squared_error(y_true, y_pred)), 2))

# Evaluate all models
evaluate("Random Forest", y_test, rf_preds)
evaluate("XGBoost", y_test, xgb_preds)
evaluate("SVM", y_test_svm, svm_preds)
