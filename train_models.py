"""Training pipeline for the Wisconsin Diagnostic Breast Cancer dataset.
Run in the BITS Virtual Lab to generate model files and evaluation artifacts.
"""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / 'model'
MODEL_DIR.mkdir(exist_ok=True)

source = load_breast_cancer(as_frame=True)
df = source.frame
X = df.drop(columns='target')
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

estimators = {
    'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'KNN': make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    'Naive Bayes': GaussianNB(),
    'SVM': make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
}

rows = []
filenames = {}
for label, estimator in estimators.items():
    estimator.fit(X_train, y_train)
    predicted = estimator.predict(X_test)
    probability = estimator.predict_proba(X_test)[:, 1]
    key = label.lower().replace(' ', '_')
    filename = f'{key}.joblib'
    joblib.dump(estimator, MODEL_DIR / filename)
    filenames[label] = filename
    rows.append({
        'Model': label,
        'Accuracy': accuracy_score(y_test, predicted),
        'AUC': roc_auc_score(y_test, probability),
        'Precision': precision_score(y_test, predicted, zero_division=0),
        'Recall': recall_score(y_test, predicted, zero_division=0),
        'F1': f1_score(y_test, predicted, zero_division=0),
        'MCC': matthews_corrcoef(y_test, predicted),
    })

pd.DataFrame(rows).to_csv(ROOT / 'model_comparison.csv', index=False)
test_csv = X_test.copy()
test_csv['target'] = y_test
test_csv.to_csv(ROOT / 'test_data.csv', index=False)
with (ROOT / 'model_manifest.json').open('w', encoding='utf-8') as f:
    json.dump({'models': filenames, 'features': list(X.columns),
               'dataset': 'Wisconsin Diagnostic Breast Cancer'}, f, indent=2)
print(pd.DataFrame(rows).round(4).to_string(index=False))
print('Saved trained models, model_comparison.csv, test_data.csv and model_manifest.json')
