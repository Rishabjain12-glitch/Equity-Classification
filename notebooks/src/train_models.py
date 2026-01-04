"""
Model Training and Evaluation Script
Train 10 ML models on processed financial data and generate comparison reports
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

def train_models(X_train_scaled, X_test_scaled, y_train, y_test):
    """Train 10 ML models and evaluate performance."""
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM (Linear)': SVC(kernel='linear', random_state=42, probability=True),
        'SVM (RBF)': SVC(kernel='rbf', random_state=42, probability=True),
        'Naive Bayes': GaussianNB(),
        'LDA': LinearDiscriminantAnalysis(),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(64, 32), random_state=42, max_iter=1000)
    }
    
    results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else np.nan
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        roc_auc = roc_auc_score(y_test, y_pred_proba) if not np.isnan(y_pred_proba).all() else np.nan
        
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc,
            'CV Mean': cv_scores.mean(),
            'CV Std': cv_scores.std()
        })
        
        print(f'{name:25} | Accuracy: {accuracy:.4f} | F1: {f1:.4f} | CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')
    
    return pd.DataFrame(results).sort_values('Accuracy', ascending=False), models

if __name__ == '__main__':
    print('Loading processed financial data...')
    try:
        df = pd.read_csv('data/processed_financial_data.csv')
        print(f'Data loaded: {df.shape}')
        
        # Select features (exclude identifiers and target)
        feature_cols = [col for col in df.columns if col not in ['Company', 'Sector', 'Year', 'Quarter', 'InvestmentQuality']]
        X = df[feature_cols].copy()
        y = df['InvestmentQuality'].copy()
        
        # Encode target
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        print(f'Features: {len(feature_cols)}')
        print(f'Target classes: {le.classes_}')
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f'\nTraining set: {X_train_scaled.shape[0]} samples')
        print(f'Test set: {X_test_scaled.shape[0]} samples')
        
        # Train models
        print('\nTraining 10 ML models...')
        results_df, trained_models = train_models(X_train_scaled, X_test_scaled, y_train, y_test)
        
        # Save results
        results_df.to_csv('reports/model_comparison.csv', index=False)
        print(f'\nModel comparison saved to reports/model_comparison.csv')
        print(f'\nTop 3 Models:')
        print(results_df.head(3).to_string(index=False))
        
    except FileNotFoundError:
        print('Error: processed_financial_data.csv not found. Run complete_analysis.py first.')
