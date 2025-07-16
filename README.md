# 🛠️ Predictive Maintenance - NASA Turbofan Engine RUL Prediction

 **Live Demo**: [Streamlit App](https://predictivemaintainance-u8iykyt5eepetw4fyjoafv.streamlit.app/)

##  Overview

This project leverages **machine learning** to predict the **Remaining Useful Life (RUL)** of aircraft engines using the **NASA Turbofan Engine Degradation Simulation Dataset**. The goal is to anticipate failures **before** they occur and optimize maintenance schedules to improve safety and reduce downtime.

## 📊 Dataset Description

- **Source**: [NASA C-MAPSS Dataset](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository)
- **Objective**: Predict RUL based on time-series sensor data from multiple engines under varying operating conditions and fault modes.
- **Features**:
  - 3 Operational Settings
  - 27 Sensor Measurements
  - Engine ID and Time Step (Cycle)
- **Target**: Remaining Useful Life (RUL) per engine per cycle

## 📂 Dataset Format

Each row in the dataset represents one time step for one engine. The columns include:

| Column     | Description               |
|------------|---------------------------|
| id         | Engine unit number        |
| cycle      | Time step in cycles       |
| op_setting | Operational conditions    |
| s1 - s27   | Sensor measurements       |

>  Uninformative sensors with constant values were removed during preprocessing.

---

## 🧪 Project Workflow

1. **Data Preprocessing**
   - StandardScaler
   - Engine-wise cycle grouping
   - Computation of actual RUL


2. **Modeling**
   - **Linear Regression** (Baseline)
   - **Support Vector Machine (SVM) Regressor**
   - **Random Forest Regressor**
   - **XGBoost Regressor**

> 🏆 **XGBoost** performed best in terms of accuracy and generalization, and hence was chosen as the **final model for deployment**.

3. **Evaluation Metrics**
   - 📉 Root Mean Squared Error (RMSE)
   - 🔁 Mean Absolute Error (MAE)

---

## 🧠 Model Performance Summary

| Model              | RMSE ↓ | MAE Score ↑ | Final Deployment |
|-------------------|--------|------------|------------------|
| Linear Regression | High   | Low        | ❌               |
| SVM Regressor     | Medium | Medium     | ❌               |
| Random Forest     | Good   | Good       | ❌               |
| **XGBoost**       | ✅ Best | ✅ Highest | ✅ **Used**      |

---

## 🚀 How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/Niyati659/Predictive_Maintainance.git
cd Predictive_Maintainance

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run Streamlit app
streamlit run app.py
