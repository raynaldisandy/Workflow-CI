# Workflow-CI - Kriteria 3

Repository ini berisi MLflow Project untuk melakukan re-training model secara otomatis menggunakan GitHub Actions.

## Struktur

```
Workflow-CI/
├── .github/workflows/ci.yml
├── MLProject/
│   ├── modelling.py
│   ├── conda.yaml
│   ├── MLproject
│   └── telco_churn_preprocessing/
└── README.md
```

## Cara Menjalankan Lokal

```bash
cd MLProject
mlflow run . -P n_estimators=200 -P max_depth=8 --env-manager=local
```

## GitHub Actions

Setiap push ke branch `main` akan otomatis menjalankan training model dan menyimpan artifact.

## Docker (Advance)

Untuk membangun Docker image:

```bash
cd MLProject
mlflow models build-docker -m "runs:/<run_id>/model" -n "raynaldisandy/telco-churn-model"
```

Tautan Docker Hub: (isi setelah push image)
https://hub.docker.com/r/raynaldisandy/telco-churn-model
