# Financial Fraud Detection Model using Machine Learning

A robust machine learning pipeline developed to identify fraudulent transactions in highly imbalanced financial datasets. This system leverages state-of-the-art tree-based algorithms alongside robust data encoding and preprocessing to ensure optimal precision and recall under extreme class imbalance.

## 📌 Project Overview
Online transactional systems process millions of events daily, a small fraction of which are fraudulent. Detecting these anomalies requires high precision to protect consumers without triggering overwhelming false alarms. 

This repository implements an end-to-end data science process—from ingestion and Exploratory Data Analysis (EDA) to feature encoding, class-imbalance analysis, modeling, and performance evaluation.

---

## 💾 Dataset Source
The transactional data evaluated in this project can be accessed at the following link:
* **Dataset Link:** [https://drive.google.com/file/d/133E0TDrfIjnhwRoGTw9OEozwBXUL38D8/view]

---

## 📊 Dataset Insights & Schema
The model evaluates transactional data featuring **5,535,323 total observations** across 10 initial properties:

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `step` | Integer | Maps a unit of time in the real world (1 step = 1 hour). |
| `type` | Object | Transactional format (`PAYMENT`, `TRANSFER`, `CASH_OUT`, `DEBIT`, `CASH_IN`). |
| `amount` | Float | Absolute volume of the transaction in local currency. |
| `nameOrig` | Object | Customer ID initiating the transaction. |
| `oldbalanceOrg` | Float | Initial balance before the transaction. |
| `newbalanceOrig` | Float | Destination balance post-transaction. |
| `nameDest` | Object | Recipient ID of the transaction. |
| `oldbalanceDest` | Float | Initial recipient balance before transaction. |
| `newbalanceDest` | Float | Destination recipient balance post-transaction. |
| `isFraud` | Float (Target) | Flag identifying whether an action is fraudulent (`1.0`) or valid (`0.0`). |

### Key Statistics & Imbalance:
* **Valid Transactions (`0.0`):** 5,531,082 cases
* **Fraudulent Transactions (`1.0`):** 4,241 cases
* **Imbalance Ratio:** ~0.0766% (Highly imbalanced context)

---

## 🛠️ Pipeline Flow & Engineering
1. **Exploratory Data Analysis (EDA):** Visualized feature distributions using Seaborn (`countplot`, `barplot`, `distplot`) and analyzed correlations utilizing factorized heatmaps to spot target associations.
2. **Categorical Feature Encoding:** Extracted structural behavior from the `type` column using structural dummy encoding (`pd.get_dummies`) with multi-collinearity prevention (`drop_first=True`).
3. **Data Splitting:** Dropped non-predictive metadata string IDs (`nameOrig`, `nameDest`) and isolated the processed features into a stratified **70/30 train-test split**.
4. **Ensemble Modeling:** Trained and evaluated three comparative classifiers under identical circumstances:
   * **Logistic Regression** (baseline)
   * **Random Forest Classifier** (7 estimators, entropy criterion)
   * **XGBoost Classifier** (`XGBClassifier`)

---

## 📈 Model Performance Metrics

Models were scored using Area Under the Receiver Operating Characteristic Curve (**ROC-AUC**) to reliably measure separation capabilities independently of the 0.07% class density base-rate:

| Algorithm Class | Training ROC-AUC | Validation ROC-AUC | Status |
| :--- | :---: | :---: | :---: |
| Logistic Regression | 0.9540 | 0.9494 | Stable Baseline |
| **XGBoost Classifier** | **0.9861** | **0.9820** | **Best Performer** |
| Random Forest Classifier | 0.9998 | 0.9473 | Overfitting Flagged |

### Detailed Evaluation of the Champion Model (XGBoost):
* **ROC-AUC Score:** 0.9820
* **Global Accuracy:** 99.98%
* **Precision (Alert Reliability):** 93.24% *(Only ~6.7% of fraud flags are false alarms)*
* **Recall (Fraud Capture Rate):** 77.34% *(Successfully captures more than 3/4 of overall hidden theft)*
* **F1-Score:** 0.8455

---

## 🛠️ Technical Prerequisites & Libraries
The code requires a Python 3.x environment along with the following standard engineering stack packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
