import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from data_handling import computeRUL, prepare_features_and_target

# Load data 
df = pd.read_csv("train_FD001.csv")
df = computeRUL(df)

# Train/test split 
from sklearn.model_selection import train_test_split
X, y, _ = prepare_features_and_target(df, scale=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest 
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

# XGBoost 
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)

# SVM (requires scaled features)
X_scaled, y_scaled, scaler = prepare_features_and_target(df, scale=True)
X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

svm = SVR(kernel='rbf')
svm.fit(X_train_svm, y_train_svm)
svm_preds = svm.predict(X_test_svm)

#  Evaluation
def evaluate(model_name, y_true, y_pred):
    print(f"\n{model_name}")
    print("MAE :", round(mean_absolute_error(y_true, y_pred), 2))
    print("RMSE:", round(mean_squared_error(y_true, y_pred, squared=False), 2))

evaluate("Random Forest", y_test, rf_preds)
evaluate("XGBoost", y_test, xgb_preds)
evaluate("SVM", y_test_svm, svm_preds)
