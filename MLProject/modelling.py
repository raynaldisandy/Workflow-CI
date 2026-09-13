"""
modelling.py - untuk MLflow Project (Kriteria 3)
Nama: Raynaldi Sandy
"""

import argparse
import warnings

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")


def main(n_estimators, max_depth):
    # Load data
    train = pd.read_csv(
        "telco_churn_preprocessing/telco_churn_train_preprocessing.csv"
    )
    test = pd.read_csv(
        "telco_churn_preprocessing/telco_churn_test_preprocessing.csv"
    )

    # Pisahkan fitur dan target
    X_train = train.drop(columns=["Churn"])
    y_train = train["Churn"]

    X_test = test.drop(columns=["Churn"])
    y_test = test["Churn"]

    print(f"Train shape: {X_train.shape}")
    print(f"Test shape : {X_test.shape}")

    # Mulai MLflow run
    with mlflow.start_run() as run:

        # Model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth if max_depth > 0 else None,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        # Training
        model.fit(X_train, y_train)

        # Prediksi
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba)

        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        # Simpan MLflow Run ID untuk proses Docker CI
        with open("run_id.txt", "w") as f:
            f.write(run.info.run_id)

        # Output hasil
        print(f"MLflow Run ID: {run.info.run_id}")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1-Score : {f1:.4f}")
        print(f"ROC-AUC  : {roc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n_estimators",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--max_depth",
        type=int,
        default=8,
    )

    args = parser.parse_args()

    main(
        args.n_estimators,
        args.max_depth,
    )