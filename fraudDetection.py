import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_curve, auc, f1_score, roc_curve, roc_auc_score
from imblearn.over_sampling import SMOTE

data = pd.read_csv("creditcard.csv", header=0, sep=",")
class_counts = data['Class'].value_counts()

# Class imbalance
plt.figure(figsize=(8, 5))
sns.barplot(x=class_counts.index, y=class_counts.values, palette="viridis")
plt.title("Class Distribution (0 = Genuine, 1 = Fraud)")
plt.ylabel("Number of Transactions")
plt.xlabel("Class")
plt.show()

print("Summary Statistics for 'Time' and 'Amount':")
print(data[['Time', 'Amount']].describe())

# Normalizing Time and Amount
scaler = StandardScaler()
data[['Time', 'Amount']] = scaler.fit_transform(data[['Time', 'Amount']])

# Splitting features and target
X = data.drop('Class', axis=1)
y = data['Class']

# Oversampling using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Model Selection - Logistic Regression
logistic_model = LogisticRegression(random_state=42, max_iter=1000)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(logistic_model, X_train, y_train, cv=cv, scoring='f1')
print(f"Cross-Validation F1 Scores: {cv_scores}")
print(f"Mean F1 Score: {np.mean(cv_scores):.4f}")

logistic_model.fit(X_train, y_train)
logistic_preds = logistic_model.predict(X_test)
logistic_probs = logistic_model.predict_proba(X_test)[:, 1]

print("Logistic Regression Classification Report:")
print(classification_report(y_test, logistic_preds))

# Precision-Recall Curve and AUC
logistic_precision, logistic_recall, _ = precision_recall_curve(y_test, logistic_probs)
logistic_auprc = auc(logistic_recall, logistic_precision)

plt.figure(figsize=(10, 6))
plt.plot(logistic_recall, logistic_precision, label=f'Logistic Regression (AUPRC = {logistic_auprc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

# ROC Curve and AUC
fpr, tpr, _ = roc_curve(y_test, logistic_probs)
roc_auc = roc_auc_score(y_test, logistic_probs)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

feature_importance = pd.Series(logistic_model.coef_[0], index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
feature_importance.head(10).plot(kind='bar', color='teal')
plt.title("Top 10 Feature Importances (Logistic Regression)")
plt.ylabel("Coefficient Value")
plt.show()
