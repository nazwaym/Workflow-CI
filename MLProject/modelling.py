"""
Modelling - Workflow CI Pipeline
================================
Script untuk melatih model ML dengan Hyperparameter Tuning.
Dijalankan melalui MLflow Project dalam GitHub Actions CI.

Nama: Nazwa Yulianti Munjana
"""

import pandas as pd
import numpy as np
import os
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

import warnings
warnings.filterwarnings('ignore')


def main():
    print("=" * 60)
    print("Workflow CI - Model Training Pipeline")
    print("Nama: Nazwa Yulianti Munjana")
    print("=" * 60)

    # 1. Setup MLflow (Local tracking - default)
    mlflow.set_experiment("Titanic_CI_Training")
    mlflow.sklearn.autolog(disable=True)

    # 2. Load Data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "titanic_preprocessing", "titanic_preprocessed.csv")
    df = pd.read_csv(data_path)
    print(f"[INFO] Dataset dimuat: {df.shape}")

    # 3. Split Data
    X = df.drop(columns=["survived"])
    y = df["survived"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[INFO] Train: {X_train.shape}, Test: {X_test.shape}")

    # 4. Hyperparameter Tuning dengan GridSearchCV
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5]
    }

    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring='accuracy',
        verbose=1
    )

    print("\n[INFO] Melakukan Hyperparameter Tuning (GridSearchCV)...")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"[INFO] Best Parameters: {best_params}")
    print(f"[INFO] Best CV Score: {grid_search.best_score_:.4f}")

    # 5. MLflow Manual Logging
    with mlflow.start_run(run_name="RandomForest_CI") as run:
        run_id = run.info.run_id
        print(f"[INFO] MLflow Run ID: {run_id}")

        # --- Log Parameters (Manual) ---
        for param_name, param_value in best_params.items():
            mlflow.log_param(param_name, param_value)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("random_state", 42)

        # --- Evaluasi Model ---
        y_pred = best_model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
        }

        # --- Log Metrics (Manual) ---
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        print("\n--- Model Evaluation ---")
        for k, v in metrics.items():
            print(f"  {k:<12}: {v:.4f}")

        # --- Log Model (wajib untuk mlflow models build-docker) ---
        mlflow.sklearn.log_model(best_model, "model")
        print("[INFO] Model di-log ke MLflow")

        # --- Artefak Tambahan: Confusion Matrix ---
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Not Survived', 'Survived'],
                    yticklabels=['Not Survived', 'Survived'], ax=ax)
        ax.set_title('Confusion Matrix - RandomForest CI', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()

        cm_path = os.path.join(script_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=100)
        plt.close()
        mlflow.log_artifact(cm_path)
        print("[INFO] Artefak: confusion_matrix.png di-log")

        # --- Artefak Tambahan: metric_info.json ---
        metric_info = {
            "metrics": metrics,
            "best_params": {k: str(v) for k, v in best_params.items()},
            "best_cv_score": float(grid_search.best_score_)
        }
        info_path = os.path.join(script_dir, "metric_info.json")
        with open(info_path, 'w') as f:
            json.dump(metric_info, f, indent=4)
        mlflow.log_artifact(info_path)
        print("[INFO] Artefak: metric_info.json di-log")

        # --- Simpan Run ID ke file untuk CI pipeline ---
        run_id_path = os.path.join(script_dir, "run_id.txt")
        with open(run_id_path, 'w') as f:
            f.write(run_id)
        print(f"[INFO] Run ID disimpan ke run_id.txt: {run_id}")

    print("\n" + "=" * 60)
    print("SELESAI - Model training pipeline berhasil!")
    print("=" * 60)


if __name__ == "__main__":
    main()
