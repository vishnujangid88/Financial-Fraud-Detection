Financial Fraud Detection with GPU-Accelerated XGBoost 🚀
A high-performance machine learning model to proactively detect fraudulent financial transactions using a large-scale dataset.
This repository contains the complete workflow for building a robust fraud detection system, from data exploration and feature engineering to training a highly accurate XGBoost model on a GPU and interpreting its results for actionable business insights.

📋 Table of Contents
Project Overview

Key Features

Dataset

Tech Stack & Libraries

Getting Started

Prerequisites

Installation

Model Performance

Evaluation Metrics

Confusion Matrix

ROC and Precision-Recall Curves

Key Predictive Factors

Actionable Insights

Contributing

License

🌟 Project Overview
The goal of this project is to address the critical business need for detecting fraudulent financial transactions. Using a synthetic dataset of over 6.3 million transactions, we develop a machine learning model capable of identifying fraudulent activity with high precision and recall. The project emphasizes not only model accuracy but also interpretability, providing key insights into the drivers of fraud that can inform preventative business strategies.

✨ Key Features
High Accuracy: Achieves an exceptional 99.95% ROC-AUC score on the test set.

High Recall: Successfully identifies 99% of all fraudulent transactions, minimizing financial loss.

GPU Acceleration: Leverages XGBoost's device='cuda' for significantly faster training on large datasets.

Advanced Feature Engineering: Creates custom features to capture subtle accounting anomalies indicative of fraud.

Imbalance Handling: Effectively manages the highly imbalanced dataset using the SMOTE technique.

End-to-End Workflow: Provides a complete Jupyter Notebook (Fraud_Detection.ipynb) covering the entire process from EDA to model evaluation.

💾 Dataset
The model is trained on the "Synthetic Financial Dataset For Fraud Detection" which mimics real-world transaction data.

Rows: 6,362,620

Columns: 11

Source: Kaggle (originally generated using PaySim)

You can download the dataset directly using the following link:

➡️ Download Fraud.csv

🛠️ Tech Stack & Libraries
This project relies on the following technologies and Python libraries:

Python 3.9+

Jupyter Notebook

Core Libraries:

pandas for data manipulation

numpy for numerical operations

scikit-learn for preprocessing and evaluation

xgboost for the core classification model

imblearn for handling class imbalance (SMOTE)

matplotlib & seaborn for data visualization

🚀 Getting Started
Follow these instructions to set up and run the project on your local machine.

Prerequisites
Python 3.9 or higher

pip package manager

A CUDA-enabled GPU is recommended for fast training, but the model will run on a CPU as well.

Installation
Clone the repository:

git clone https://github.com/your-username/fraud-detection-model.git
cd fraud-detection-model

Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install the required packages:

pip install -r requirements.txt

(Note: A requirements.txt file should be created containing the libraries listed in the tech stack.)

Download the dataset:

Download Fraud.csv from the link provided above.

Place the Fraud.csv file in the root directory of the project.

Launch Jupyter Notebook:

jupyter notebook

Open the Fraud_Detection.ipynb file and run the cells sequentially.

📊 Model Performance
The model was evaluated on a test set containing 1,916,877 transactions, of which 2,468 were fraudulent.

Evaluation Metrics
Metric

Score

Interpretation

ROC-AUC Score

0.9995

Near-perfect ability to distinguish between fraud and non-fraud.

Fraud Recall

0.99

Excellent. The model successfully caught 99% of all fraudulent transactions.

Fraud Precision

0.77

Good. When the model predicts fraud, it is correct 77% of the time.

Fraud F1-Score

0.87

Strong harmonic mean of precision and recall.

Confusion Matrix
The confusion matrix shows the trade-off between correctly identifying fraud (True Positives) and incorrectly flagging legitimate transactions (False Positives).

              Predicted Not Fraud   Predicted Fraud
Actual Not Fraud      1,913,687             722
Actual Fraud                 21           2,447

True Positives (Caught Fraud): 2,447

False Negatives (Missed Fraud): 21 (A very low number, which is excellent)

ROC and Precision-Recall Curves
<p align="center">
<img src="https://i.imgur.com/your-roc-curve-image.png" width="48%" alt="ROC Curve">
<img src="https://i.imgur.com/your-pr-curve-image.png" width="48%" alt="Precision-Recall Curve">
</p>
(Note: You would replace the placeholder image URLs with actual images of your plots generated from the notebook.)

🔑 Key Predictive Factors
The model's feature importance analysis revealed the top factors that predict fraudulent activity:

newbalanceOrig (Originator's New Balance): The final balance of the sender's account was the most influential feature.

errorBalanceOrig (Engineered Feature): Our custom-built feature to detect accounting errors (oldBalance + amount - newBalance) was the second most powerful predictor. This confirms that balance inconsistencies are a major red flag.

Transaction type: The type of transaction (e.g., PAYMENT, CASH_OUT, TRANSFER) was highly indicative of fraud patterns.

amount: The transaction amount itself was also a significant factor.

These factors make strong intuitive sense, as they align with common fraud tactics like account draining and exploiting system accounting logic.

💡 Actionable Insights
Based on the model's findings, a company can implement the following preventative measures:

Deploy Real-Time Scoring: Integrate the model to score transactions in real-time and automatically flag or block those with a high fraud probability.

Strengthen Business Rules: Implement hard-coded rules to flag any transaction where the balance does not reconcile (i.e., where errorBalanceOrig is non-zero).

Apply Contextual Scrutiny: Place stricter limits or require secondary authentication for TRANSFER and CASH_OUT transactions that originate from accounts with unusual balance patterns.

🤝 Contributing
Contributions are welcome! If you have suggestions for improving this model or want to add new features, please feel free to:

Fork the repository.

Create a new feature branch (git checkout -b feature/AmazingFeature).

Commit your changes (git commit -m 'Add some AmazingFeature').

Push to the branch (git push origin feature/AmazingFeature).

Open a Pull Request.

📜 License
This project is licensed under the MIT License. See the LICENSE file for details.
