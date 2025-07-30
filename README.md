# Financial Fraud Detection with GPU-Accelerated XGBoost 🚀

A high-performance machine learning model to proactively detect fraudulent financial transactions using a large-scale dataset.  
This repository contains the complete workflow for building a robust fraud detection system—from data exploration and feature engineering to training a highly accurate XGBoost model on a GPU and interpreting its results for actionable business insights.

---

## 📋 Table of Contents

- [🌟 Project Overview](#-project-overview)
- [✨ Key Features](#-key-features)
- [💾 Dataset](#-dataset)
- [🛠️ Tech Stack & Libraries](#️-tech-stack--libraries)
- [🚀 Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [📊 Model Performance](#-model-performance)
  - [Evaluation Metrics](#evaluation-metrics)
  - [Confusion Matrix](#confusion-matrix)
  - [ROC and Precision-Recall Curves](#roc-and-precision-recall-curves)
- [🔑 Key Predictive Factors](#-key-predictive-factors)
- [💡 Actionable Insights](#-actionable-insights)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)

---

## 🌟 Project Overview

The goal of this project is to address the critical business need for detecting fraudulent financial transactions. Using a synthetic dataset of over **6.3 million** transactions, we develop a machine learning model capable of identifying fraudulent activity with high **precision and recall**.  

This project emphasizes not only **model accuracy** but also **interpretability**, providing key insights into the drivers of fraud that can inform proactive business strategies.

---

## ✨ Key Features

- **High Accuracy**: Achieves a 99.95% ROC-AUC score.
- **High Recall**: Identifies 99% of all fraudulent transactions.
- **GPU Acceleration**: Utilizes `device='cuda'` in XGBoost for fast model training.
- **Advanced Feature Engineering**: Custom features capture subtle accounting inconsistencies.
- **Imbalance Handling**: SMOTE is used to tackle severe class imbalance.
- **End-to-End Workflow**: Covers EDA, modeling, evaluation, and insights in `Fraud_Detection.ipynb`.

---

## 💾 Dataset

- **Name**: Synthetic Financial Dataset for Fraud Detection  
- **Rows**: 6,362,620  
- **Columns**: 11  
- **Source**: [Kaggle - PaySim Simulation](https://www.kaggle.com/datasets/ntnu-testimon/paysim1)

➡️ **[Download Fraud.csv](https://www.kaggle.com/datasets/ntnu-testimon/paysim1)**

---

## 🛠️ Tech Stack & Libraries

- **Python 3.9+**
- **Jupyter Notebook**

**Core Libraries**:
- `pandas`, `numpy` – data handling
- `scikit-learn` – preprocessing & evaluation
- `xgboost` – classification model
- `imblearn` – SMOTE for imbalance handling
- `matplotlib`, `seaborn` – visualization

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- pip
- CUDA-enabled GPU (optional but recommended)

### Installation

```bash
git clone https://github.com/your-username/fraud-detection-model.git
cd fraud-detection-model

