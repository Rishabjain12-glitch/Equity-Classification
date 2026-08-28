# Equity Classification

A machine-learning project that evaluates long-term equity classifications using fundamental financial metrics rather than short-term price movement.

## What this repository contains

- Raw and calculated metrics for an Adani Green Energy case study
- A comparison pipeline for 10 classifiers
- Five-fold stratified cross-validation
- Accuracy, precision, recall, F1, ROC-AUC, and cross-validation summaries
- Reproducible Python dependencies in `requirements.txt`

## Models compared

Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, linear and RBF SVMs, Naive Bayes, Linear Discriminant Analysis, K-Nearest Neighbours, and a multilayer perceptron.

## Run locally

```bash
git clone https://github.com/Rishabjain12-glitch/Equity-Classification.git
cd Equity-Classification
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/complete_analysis.py
```

On Windows, activate the environment with `.venv\\Scripts\\activate`.

## Repository structure

```text
.
├── src/                         # Model-training scripts
├── ADANI_GREEN_METRICS*         # Case-study inputs and calculated metrics
├── ml_model_metrics.csv         # Model evaluation output
├── requirements.txt
└── README.md
```

## Scope

This is an applied-learning project for comparing modelling approaches in fundamental equity analysis. It is not financial advice.
