Fraud Detection with Logistic Regression and SMOTE
This repository contains a comprehensive fraud detection project using logistic regression to classify transactions as genuine or fraudulent. The project addresses challenges like class imbalance using SMOTE, employs cross-validation for reliable model evaluation, and includes extensive visualizations for better insights.

Features of the Project
Data Exploration:
Visualize the class distribution of transactions.
Explore the summary statistics of key features like Time and Amount.

Data Preprocessing:
Normalize the Time and Amount columns using StandardScaler.
Handle class imbalance with SMOTE (Synthetic Minority Oversampling Technique).

Model Training and Evaluation:
Build a Logistic Regression model.
Implement cross-validation for robust model evaluation.
Evaluate the model with metrics like Precision, Recall, F1-Score, and AUC-PR.

Generate visualizations such as:
Precision-Recall curve.
ROC-AUC curve.

Reusable Code:
Organized and modular code for ease of understanding and extension.

Usage Instructions
Prerequisites
Ensure you have the following installed:
Python 3.x

Required libraries listed in requirements.txt:
numpy
pandas
seaborn
matplotlib
scikit-learn
imbalanced-learn

Install them with:
pip install -r requirements.txt

Steps to Use the Code
Clone this repository:
git clone https://github.com/yourusername/fraud-detection.git
cd fraud-detection

Download the dataset:
Obtain the dataset file creditcard.csv https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Place the file in the root folder of the repository.

Run the code:
python fraudDetection.py

Visualizations Included
Class distribution bar chart.
Precision-Recall and ROC-AUC curves for performance evaluation.

Key Learnings
This project demonstrates:
Handling imbalanced datasets with techniques like SMOTE.
The importance of cross-validation in ensuring model reliability.
The value of visualizations for interpreting data and results.
An end-to-end pipeline for fraud detection using machine learning.

Acknowledgments
Special thanks to the creators of the Credit Card Fraud Detection Dataset.
