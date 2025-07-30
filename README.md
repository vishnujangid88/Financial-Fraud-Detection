# 🛡️ Financial Fraud Detection with GPU-Accelerated XGBoost

A powerful, interpretable, and GPU-accelerated machine learning model built to proactively detect fraudulent financial transactions. This project processes a large-scale dataset to identify patterns in financial fraud with high precision and recall.

---

## 📚 Table of Contents

- [🌟 Project Overview](#-project-overview)
- [✨ Key Features](#-key-features)
- [💾 Dataset](#-dataset)
- [🛠️ Tech Stack & Libraries](#️-tech-stack--libraries)
- [🚀 Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [📊 Model Performance](#-model-performance)
- [🔍 Feature Importance](#-feature-importance)
- [💡 Actionable Insights](#-actionable-insights)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)

---

## 🌟 Project Overview

This project tackles the challenge of detecting fraudulent financial transactions using advanced machine learning techniques. Built with XGBoost and GPU support, it not only delivers exceptional accuracy but also identifies meaningful patterns that can guide prevention strategies.

---

## ✨ Key Features

- ✅ High Accuracy (ROC-AUC: **0.9995**)
- 🔍 High Recall – Captures **99%** of fraudulent transactions
- ⚡ GPU acceleration with `xgboost` using `device='cuda'`
- 🧪 Advanced feature engineering for detecting balance inconsistencies
- ⚖️ SMOTE applied to address severe class imbalance
- 📒 Complete end-to-end pipeline in `Fraud_Detection_Model.ipynb`

---

## 💾 Dataset

- **Source**: [Kaggle - PaySim Synthetic Financial Dataset]([https://www.kaggle.com/datasets/ntnu-testimon/paysim1](https://drive.usercontent.google.com/download?id=1VNpyNkGxHdskfdTNRSjjyNa5qC9u0JyV&export=download&authuser=0))
- **Rows**: 6,362,620
- **Columns**: 11
- **Target Column**: `isFraud`

---

## 🛠️ Tech Stack & Libraries

- Python 3.9+
- Jupyter Notebook

**Required Libraries** (also in `requirements.txt`):

```txt
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
imblearn
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- pip
- (Optional) CUDA-enabled GPU for fast training

### Installation

```bash
git clone https://github.com/your-username/fraud-detection-model.git
cd fraud-detection-model

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Place the `Fraud.csv` dataset in the root directory.

Launch:

```bash
jupyter notebook
```

Open and run `Fraud_Detection_Model.ipynb`.

---

## 📊 Model Performance

**Test Set**: 1,916,877 transactions  
**Fraud Cases**: 2,468

### Evaluation Metrics

| Metric        | Value  | Description                                  |
|---------------|--------|----------------------------------------------|
| ROC-AUC       | 0.9995 | Near-perfect fraud/non-fraud separation       |
| Recall        | 0.99   | 99% of fraudulent cases were detected         |
| Precision     | 0.77   | 77% of fraud predictions were correct         |
| F1-Score      | 0.87   | Balanced precision and recall                 |

### Confusion Matrix

|                   | Predicted Not Fraud | Predicted Fraud |
|-------------------|---------------------|-----------------|
| **Actual Not Fraud** | 1,913,687         | 722             |
| **Actual Fraud**     | 21                | 2,447           |

- ✅ **True Positives**: 2,447  
- ❌ **False Negatives**: 21

---

### ROC and PR Curves

<p align="center">
  <img src="images/roc_curve.png" width="45%" alt="ROC Curve">
  <img src="images/pr_curve.png" width="45%" alt="Precision-Recall Curve">
</p>

> **Note**: Save the ROC and Precision-Recall plots from the notebook as `roc_curve.png` and `pr_curve.png` in an `images/` folder.

---

## 🔍 Feature Importance

Top predictors of fraud:

- **`newbalanceOrig`** – Final sender balance
- **`errorBalanceOrig`** – Custom anomaly feature
- **`amount`** – Transaction amount
- **`type`** – Transaction type (e.g., TRANSFER, CASH_OUT)

> Engineered features like `errorBalanceOrig` were especially effective at surfacing fraud patterns.

---

## 💡 Actionable Insights

- **Real-Time Scoring**: Integrate the model to flag transactions during execution.
- **Fraud Rule Automation**: Flag any transaction with non-zero accounting errors.
- **Transaction-Type Filters**: Apply additional scrutiny to high-risk transaction types (e.g., TRANSFER, CASH_OUT).

---

## 🤝 Contributing

We welcome community contributions! Here’s how:

```bash
# Fork the repo and create your branch
git checkout -b feature/YourFeature

# Make changes and commit
git commit -m "Added a new feature"

# Push and open a Pull Request
git push origin feature/YourFeature
```

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

> Built with ❤️ to empower financial systems against fraud using fast, interpretable AI.
