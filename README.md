# Workflow-CI - Kriteria 3

Repository ini berisi MLflow Project untuk melakukan re-training model
secara otomatis menggunakan GitHub Actions serta membangun dan
mempublikasikan Docker image ke Docker Hub.

## Struktur Project

```text
Workflow-CI/
├── .github/
│   └── workflows/
│       └── ci.yml
├── MLProject/
│   ├── modelling.py
│   ├── MLproject
│   ├── conda.yaml
│   ├── Dockerfile
│   └── telco_churn_preprocessing/
│       ├── preprocessor.joblib
│       ├── telco_churn_train_preprocessing.csv
│       └── telco_churn_test_preprocessing.csv
└── README.md
```

## Menjalankan MLflow Project Secara Lokal

Masuk ke folder `MLProject`:

```bash
cd MLProject
```

Kemudian jalankan:

```bash
mlflow run . --env-manager=local -P n_estimators=200 -P max_depth=8
```

Model menggunakan Random Forest Classifier dengan metrik evaluasi
Accuracy, Precision, Recall, F1-Score, dan ROC-AUC.

## GitHub Actions

Workflow CI berjalan ketika terdapat push atau pull request ke branch
`main` atau `master`, serta dapat dijalankan secara manual menggunakan
`workflow_dispatch`.

Tahapan CI meliputi:

1. Checkout repository.
2. Setup Python 3.13.
3. Install dependency.
4. Menjalankan MLflow Project untuk training model.
5. Menyimpan MLflow artifacts sebagai GitHub Actions artifact.
6. Login ke Docker Hub.
7. Membangun Docker image menggunakan `mlflow models build-docker`.
8. Push Docker image ke Docker Hub.

## Docker Hub

Docker image dipublikasikan ke repository:

**raynaldi/telco-churn-mlflow**

Tag yang digunakan:

```text
latest
```

Docker Hub:
https://hub.docker.com/r/raynaldi/telco-churn-mlflow

## Docker Image

Image Docker dapat dijalankan menggunakan:

```bash
docker run -p 8080:8080 raynaldi/telco-churn-mlflow:latest
```

Model menggunakan MLflow Model Serving pada port `8080`.

Endpoint prediksi:

```text
POST /invocations
```
