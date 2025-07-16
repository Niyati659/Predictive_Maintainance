# Predictive Maintenance using NASA Turbofan Engine Dataset

## 🚀 Project Summary

This project predicts the **Remaining Useful Life (RUL)** of aircraft engines using the NASA CMAPSS dataset. The goal is to enable proactive maintenance before failure occurs, reducing downtime and improving safety.

We used sensor data from 27 sensors across 4 different operating conditions. Multiple machine learning models were tested — including **SVM**, **Random Forest**, and **XGBoost**. XGBoost gave the best performance and was selected as the final model.

---

## 📁 Dataset

- Source: [NASA CMAPSS Dataset](https://www.nasa.gov/cmapps/)
- Files: `train_FD00X.txt`, `test_FD00X.txt`, `RUL_FD00X.txt`
- Features:
  - 3 operational settings
  - 21–24 sensor measurements (some removed during cleaning)
- Label:
  - Remaining Useful Life (RUL) — calculated from engine cycle count

---

## 🧹 Preprocessing

- Removed constant and highly correlated sensors
- Normalized data using `StandardScaler` (for SVM)
- Computed RUL for each engine
- Combined datasets `FD001–FD004` for generalization

---

## 🤖 Modeling

- Trained three regressors:
  - Support Vector Machine (SVM)
  - Random Forest Regressor
  - ✅ XGBoost Regressor (best performing)

- Evaluation Metrics:
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)

- Trained model saved using `joblib`

---

## 🧪 Inference

- User inputs sensor values via command-line
- Model predicts RUL
- Maintenance alert triggered based on threshold:
  - RUL < 20: 🚨 Immediate maintenance
  - RUL < 50: ⏳ Schedule soon
  - Else: ✅ Healthy

---

## 🌐 Deployed Web App

[🔗 Click here to access the Streamlit app](https://predictivemaintainance-u8iykyt5eepetw4fyjoafv.streamlit.app/)

---


