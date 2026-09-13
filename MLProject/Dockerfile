FROM python:3.13.5-slim

WORKDIR /opt/mlflow

RUN pip install --no-cache-dir mlflow==3.16.0 scikit-learn==1.9.1 skops==0.14.0 pandas==3.0.5 numpy==2.5.3

COPY model_docker/model /opt/ml/model

EXPOSE 8080

CMD ["mlflow", "models", "serve", "-m", "/opt/ml/model", "--host", "0.0.0.0", "--port", "8080", "--no-conda"]