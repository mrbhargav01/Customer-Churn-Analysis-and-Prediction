# -*- coding: utf-8 -*-
"""
Customer Churn Analysis and Prediction
Robust, environment-friendly version that avoids crashing when optional
visualization libraries (matplotlib / seaborn) are not available.

Features:
- Graceful fallback when plotting libraries are missing (no ModuleNotFoundError)
- Fixed file-path quoting/syntax issues
- Safer handling of the 'Churn' column (maps Yes/No -> 1/0 when present)
- Small synthetic test added (run with --run-tests)
- CLI: pass a file path with --file
"""

import warnings
import os
import sys
import argparse
import tempfile

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix,
                             classification_report)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings('ignore')

# Try importing plotting libraries but don't fail if they aren't available
PLOTTING_AVAILABLE = True
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.style.use('fivethirtyeight')
    sns.set_palette("Set2")
    plt.rcParams['figure.figsize'] = (10, 6)
except Exception as e:
    PLOTTING_AVAILABLE = False
    print("Warning: Matplotlib/Seaborn not available. Plots will be skipped.", file=sys.stderr)


# -----------------------------
# Task 1: Data Preparation
# -----------------------------

def load_and_prepare_data(file_path):
    """Load and prepare the dataset.

    Returns:
        X_train, X_test, y_train, y_test, df, scaler
        or (None,)*6 on failure.
    """
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found. Please check the file path.")
        return None, None, None, None, None, None

    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        print(f"Error reading CSV: {exc}")
        return None, None, None, None, None, None

    print(f"Dataset loaded: shape={df.shape}")

    # Basic preview
    print(df.head(3).to_string(index=False))

    # Replace blank-like strings with NaN
    df.replace(['', ' '], np.nan, inplace=True)

    # Convert TotalCharges to numeric (some datasets have spaces)
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Fill missing values: categorical -> mode, numeric -> median
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype == 'object':
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col].fillna(mode_val[0], inplace=True)
                else:
                    df[col].fillna('Unknown', inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)

    # Drop customerID if present
    for candidate in ['customerID', 'customer_id', 'CustomerID']:
        if candidate in df.columns:
            df.drop(candidate, axis=1, inplace=True)
            break

    # Ensure Churn column exists
    if 'Churn' not in df.columns:
        print("Error: 'Churn' column not found in dataset.")
        return None, None, None, None, None, None

    # Map Churn to 0/1 if it's Yes/No
    if df['Churn'].dtype == 'object':
        mapping = {'No': 0, 'Yes': 1}
        unique_vals = set(df['Churn'].dropna().unique())
        if unique_vals.issubset(set(mapping.keys())):
            df['Churn'] = df['Churn'].map(mapping).astype(int)
        else:
            # Fallback to LabelEncoder
            le = LabelEncoder()
            df['Churn'] = le.fit_transform(df['Churn']).astype(int)

    # Convert other object cols: label-encode binary, one-hot the rest
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Binary columns (excluding Churn)
    le = LabelEncoder()
    binary_cols = [c for c in categorical_cols if df[c].nunique() == 2]
    for col in binary_cols:
        df[col] = le.fit_transform(df[col])

    # Remaining non-numeric columns -> one-hot
    remaining_object_cols = df.select_dtypes(include=['object']).columns.tolist()
    if remaining_object_cols:
        df = pd.get_dummies(df, columns=remaining_object_cols, drop_first=True)

    # Separate X and y
    X = df.drop('Churn', axis=1)
    y = df['Churn'].astype(int)

    # Final safety: ensure all X columns are numeric
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        # Convert any remaining to numeric where possible, otherwise drop
        for c in non_numeric:
            try:
                X[c] = pd.to_numeric(X[c], errors='coerce').fillna(0)
            except Exception:
                X.drop(c, axis=1, inplace=True)

    # Train-test split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
    except Exception as exc:
        print(f"Error during train_test_split: {exc}")
        return None, None, None, None, None, None

    # Scale numeric features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"Prepared data: X_train={X_train.shape}, X_test={X_test.shape}")
    return X_train, X_test, y_train, y_test, df, scaler


# -----------------------------
# EDA: If plotting libs are missing, print textual summaries instead
# -----------------------------

def perform_eda(df):
    """Simple EDA — uses plots when available, otherwise prints stats."""
    if df is None:
        print("No dataframe provided for EDA.")
        return

    # Ensure Churn in human-readable form
    if 'Churn' in df.columns and set(df['Churn'].unique()).issubset({0, 1}):
        churn_labels = df['Churn'].map({0: 'No', 1: 'Yes'})
    else:
        churn_labels = df['Churn']

    churn_counts = churn_labels.value_counts()
    churn_rate = churn_counts.get('Yes', 0) / len(df) * 100 if 'Yes' in churn_counts else (
        churn_counts.get(1, 0) / len(df) * 100 if 1 in churn_counts else 0)

    print(f"Overall churn rate: {churn_rate:.2f}%")

    if not PLOTTING_AVAILABLE:
        print("Plotting is unavailable in this environment — skipping visual EDA.")
        # Print a few useful summaries
        if 'tenure' in df.columns:
            print("Tenure: ", df['tenure'].describe().to_string())
        if 'MonthlyCharges' in df.columns:
            print("MonthlyCharges: ", df['MonthlyCharges'].describe().to_string())
        return

    # If plotting available, do some quick visuals
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(8, 6))
        churn_counts = churn_labels.value_counts()
        plt.pie(churn_counts, labels=churn_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title('Overall Churn Rate')
        plt.show()

        if 'gender' in df.columns:
            plt.figure(figsize=(10, 4))
            sns.countplot(x='gender', hue='Churn', data=df)
            plt.title('Churn by Gender')
            plt.show()

        if 'tenure' in df.columns:
            plt.figure(figsize=(10, 4))
            sns.histplot(data=df, x='tenure', hue='Churn', kde=True, element='step', stat='density', common_norm=False)
            plt.title('Tenure Distribution by Churn')
            plt.show()
    except Exception as exc:
        print(f"Plotting failed: {exc}")


# -----------------------------
# A minimal segmentation function that is safe when plotting is unavailable
# -----------------------------

def customer_segmentation(df):
    if df is None:
        print("No dataframe provided for segmentation.")
        return pd.DataFrame()

    if 'tenure' not in df.columns or 'MonthlyCharges' not in df.columns:
        print("Required columns for segmentation ('tenure' or 'MonthlyCharges') are missing.")
        return pd.DataFrame()

    df_segment = df.copy()
    # Map churn back to labels for readability
    df_segment['ChurnLabel'] = df_segment['Churn'].map({0: 'No', 1: 'Yes'}) if set(df_segment['Churn'].unique()).issubset({0, 1}) else df_segment['Churn']

    df_segment['TenureSegment'] = pd.cut(df_segment['tenure'], bins=[-1, 12, 24, 48, 72, np.inf],
                                         labels=['0-1yr', '1-2yr', '2-4yr', '4-6yr', '6+yr'])

    df_segment['ChargeSegment'] = pd.cut(df_segment['MonthlyCharges'], bins=[-1, 35, 70, 90, np.inf],
                                         labels=['Low', 'Medium', 'High', 'Very High'])

    # Return high-value customers for later recommendation
    high_value = df_segment[(df_segment['ChargeSegment'].isin(['High', 'Very High'])) &
                             (df_segment['TenureSegment'].isin(['4-6yr', '6+yr']))]
    if high_value.empty:
        print("No high-value customers found based on the simple thresholds.")
    else:
        print(f"Found {len(high_value)} high-value customers (simple heuristic).")

    return high_value


# -----------------------------
# Modeling: training and basic evaluation (kept optional because it can be heavy)
# -----------------------------

def build_and_evaluate_models(X_train, X_test, y_train, y_test, do_tuning=False):
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # Some classifiers may not implement predict_proba
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
        except Exception:
            y_proba = np.zeros_like(y_pred, dtype=float)

        results[name] = {
            'model': model,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba) if y_proba.sum() > 0 else 0.0,
            'y_pred': y_pred,
            'y_pred_proba': y_proba
        }

        print(f"{name}: acc={results[name]['accuracy']:.4f}, f1={results[name]['f1']:.4f}")

    # Optionally, hyperparameter tuning for Random Forest
    if do_tuning:
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='f1', n_jobs=-1)
        grid.fit(X_train, y_train)
        best = grid.best_estimator_
        print(f"Tuned RF best params: {grid.best_params_}")
        # Evaluate tuned model
        y_pred = best.predict(X_test)
        try:
            y_proba = best.predict_proba(X_test)[:, 1]
        except Exception:
            y_proba = np.zeros_like(y_pred, dtype=float)
        results['Tuned Random Forest'] = {
            'model': best,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba) if y_proba.sum() > 0 else 0.0,
            'y_pred': y_pred,
            'y_pred_proba': y_proba
        }

    return results


def evaluate_and_interpret_model(results, X_test, y_test, df):
    if not results:
        print("No results to evaluate.")
        return pd.DataFrame()

    # Choose best by F1
    best_name = max(results.keys(), key=lambda k: results[k]['f1'])
    best = results[best_name]['model']
    y_pred = results[best_name]['y_pred']
    y_proba = results[best_name]['y_pred_proba']

    print(f"Best model by F1: {best_name}")
    print(classification_report(y_test, y_pred, zero_division=0))

    if PLOTTING_AVAILABLE:
        try:
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d')
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.show()

            fpr, tpr, _ = roc_curve(y_test, y_proba)
            plt.plot(fpr, tpr, label=f'{best_name} (AUC={roc_auc_score(y_test, y_proba):.3f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.legend()
            plt.show()
        except Exception as exc:
            print(f"Plotting evaluation failed: {exc}")

    # Feature importance if available
    feature_importance_df = pd.DataFrame()
    feature_names = df.drop('Churn', axis=1).columns.tolist()
    if hasattr(best, 'feature_importances_'):
        fi = best.feature_importances_
        feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': fi}).sort_values('importance', ascending=False)
    elif hasattr(best, 'coef_'):
        coef = best.coef_[0]
        feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': coef}).sort_values('importance', ascending=False)
    else:
        print('No feature importance available for this model type.')

    return feature_importance_df


def provide_business_recommendations(feature_importance_df, high_value_customers, df):
    print("\n" + "=" * 60)
    print("BUSINESS RECOMMENDATIONS")
    print("=" * 60)

    if not feature_importance_df.empty:
        top = feature_importance_df.head(5)['feature'].tolist()
        print("Top features influencing churn:", top)
    else:
        print("Feature importance not available.")

    if high_value_customers is not None and not high_value_customers.empty:
        at_risk = high_value_customers[high_value_customers['Churn'] == 1]
        print(f"High-value customers at risk: {len(at_risk)}")
    else:
        print("No high-value customers identified.")

    print("Sample recommendations:\n - Incentivize longer contracts\n - Promote automatic payments\n - Target early-tenure customers with onboarding/support")


# -----------------------------
# Small synthetic test case for the preprocessing pipeline
# -----------------------------

def run_synthetic_test():
    """Create a tiny synthetic Telco-like CSV and ensure preprocessing works."""
    print("Running synthetic test...")
    data = {
        'customerID': [f'C{i:03d}' for i in range(10)],
        'gender': ['Female', 'Male'] * 5,
        'SeniorCitizen': [0, 1] * 5,
        'Partner': ['Yes', 'No'] * 5,
        'Dependents': ['No', 'Yes'] * 5,
        'tenure': list(range(1, 11)),
        'PhoneService': ['Yes', 'No'] * 5,
        'MonthlyCharges': [20.0 + i for i in range(10)],
        'TotalCharges': [20.0 + i for i in range(10)],
        'Churn': ['No', 'Yes'] * 5
    }
    df_test = pd.DataFrame(data)
    tmp = os.path.join(tempfile.gettempdir(), 'test_telco.csv')
    df_test.to_csv(tmp, index=False)

    X_train, X_test, y_train, y_test, df, scaler = load_and_prepare_data(tmp)
    assert df is not None, "Preprocessing returned None df"
    assert X_train is not None and X_test is not None, "Train/test data not returned"
    assert hasattr(scaler, 'transform'), "Scaler missing"
    print("Synthetic test passed.")


# -----------------------------
# CLI / main
# -----------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description='Customer Churn Analysis (robust)')
    parser.add_argument('--file', '-f', default='Telco_Customer_Churn_Dataset.csv', help='Path to CSV file')
    parser.add_argument('--run-tests', action='store_true', help='Run synthetic tests and exit')
    parser.add_argument('--no-model', action='store_true', help='Skip model training (just preprocess + EDA)')
    parser.add_argument('--tune', action='store_true', help='Enable hyperparameter tuning (may be slow)')
    args = parser.parse_args(argv)

    if args.run_tests:
        run_synthetic_test()
        return

    X_train, X_test, y_train, y_test, df, scaler = load_and_prepare_data(args.file)
    if df is None:
        print("Failed to load data — exiting.")
        return

    perform_eda(df)
    high_value_customers = customer_segmentation(df)

    if not args.no_model:
        results = build_and_evaluate_models(X_train, X_test, y_train, y_test, do_tuning=args.tune)
        fi = evaluate_and_interpret_model(results, X_test, y_test, df)
    else:
        results = {}
        fi = pd.DataFrame()

    provide_business_recommendations(fi, high_value_customers, df)


if __name__ == '__main__':
    main()