# Machine Learning Assignment 2 — Working Starter

**Important:** This is a learning scaffold. The assignment explicitly requires original work and allows AI only for learning support, not direct copy-paste submission. Understand, adapt, test and document your own decisions before submitting.

## Problem statement
Compare five binary classifiers using the Wisconsin Diagnostic Breast Cancer dataset and expose a test CSV evaluation UI in Streamlit. Predictions are made using already-trained models, without training on uploaded test data.

## Dataset
Scikit-learn's copy of the Wisconsin Diagnostic Breast Cancer dataset: 569 records, 30 numeric predictors, and one binary target column. The assignment asks for a dataset chosen from a public repository (Kaggle or UCI); cite and verify the **UCI Breast Cancer Wisconsin (Diagnostic)** source in your final report.

## Models and metrics
Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes, SVM. Accuracy, ROC AUC, Precision, Recall, F1 and MCC are calculated on the same 20% stratified held-out test split (random_state=42). Scaler is fit on the training data within the Logistic Regression, KNN and SVM pipelines.

## Run in BITS Virtual Lab
1. Upload this folder's files into a single working folder in the lab, or unzip them there.
2. Install dependencies: `python -m pip install -r requirements.txt` (prefer a separate environment if lab's TensorFlow dependency conflicts with Streamlit).
3. Run `python train_models.py` in a Jupyter Terminal. This saves five fitted `.joblib` models in `model/`, a `model_manifest.json`, `test_data.csv`, and `model_comparison.csv`.
4. In a terminal, run `streamlit run app.py --server.address 0.0.0.0`. Use the lab's forwarded-port/browser support if available. A public deployment requires a GitHub repo and Streamlit Community Cloud.
5. Upload `test_data.csv` through the app, choose models and inspect results. The app does not train on the uploaded CSV.
6. Capture at least one BITS Virtual Lab screenshot during actual execution. Do not claim local execution as lab execution.

## Model comparison
**Replace with your results from the actual lab run.** Include a table for Accuracy, AUC, Precision, Recall, F1 and MCC for all five algorithms, plus commentary explaining why the best model was selected. Do not fabricate numbers.

## GitHub and Streamlit deployment
Upload `app.py`, `train_models.py`, `requirements.txt`, `README.md`, `test_data.csv`, `model_manifest.json`, and the `model/` directory to a repository. The pretrained models must exist in the repository for the Streamlit app to load without retraining. Deploy `app.py` on Streamlit Community Cloud from that repo. Test the URL and CSV upload before submitting.

## Final PDF checklist
1. GitHub repo URL with full source and saved models.
2. Live Streamlit app URL.
3. Screenshot demonstrating the BITS Virtual Lab run.
4. README material: problem, UCI dataset description/source, metrics table, interpretation and best model.
5. Make sure your course submission is **submitted**, not left as a draft, before the stated deadline.

## What to personalize
Describe the dataset source, your own motivation, hyperparameter choices, model findings, limitations and real-world relevance. Check prediction labels (0=malignant, 1=benign), explain class imbalance and positive-class convention, and record the package versions you actually ran.
