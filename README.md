# Telco Customer Churn MLOps

End-to-end production-style machine learning project for predicting telecom customer churn.

## Stack
- Python, scikit-learn, pandas
- DVC for data/pipeline versioning
- MLflow for experiment tracking and model registry
- Flask REST API for inference
- Prometheus + Grafana for production monitoring
- GitHub Actions for CI
- Docker for packaging

## Architecture
```text
IBM Telco Churn Dataset
        |
        v
      DVC
        |
        v
 Preprocess -> Train -> Evaluate
        |         |
        |       MLflow
        |         |
        +---------+----> Registered Model
                         |
                         v
                    Flask REST API
                         |
             +-----------+-----------+
             |                       |
        Prometheus                Drift check
             |                       |
             +-----------+-----------+
                         v
                       Grafana
```

Dataset: IBM Telco Customer Churn sample, 7,043 customers and 21 raw columns. Source URL is documented in `src/data/download.py`.

## Local setup
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python src/data/download.py
python src/pipeline.py
pytest -q
python app.py
```

## MLflow
Set `MLFLOW_TRACKING_URI` to your MLflow server. For local development:
```bash
mlflow server --host 127.0.0.1 --port 5000
```
Then:
```bash
set MLFLOW_TRACKING_URI=http://127.0.0.1:5000
# Linux/macOS: export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
python src/train.py
```

For a remote MLflow server, use its private/VPN URL rather than exposing it unnecessarily. MLflow's current docs recommend a database-backed backend store for Model Registry usage.

## DVC
```bash
dvc init
dvc add data/raw/Telco-Customer-Churn.csv
git add data/raw/Telco-Customer-Churn.csv.dvc .gitignore
# Configure a remote of your choice, then:
dvc push
```

## API
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No","tenure":12,"PhoneService":"Yes","MultipleLines":"No","InternetService":"DSL","OnlineSecurity":"No","OnlineBackup":"Yes","DeviceProtection":"No","TechSupport":"No","StreamingTV":"No","StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":70.35,"TotalCharges":844.2}'
```

Metrics: `GET /metrics`.

## Grafana
Configure Prometheus to scrape:
```yaml
scrape_configs:
  - job_name: telco-churn-api
    static_configs:
      - targets: ['host.docker.internal:8000']
```
If Flask is on the EC2 host and Prometheus is also on Docker, replace the target with the reachable host/container address. Create Grafana panels from Prometheus metrics such as `api_requests_total`, `api_request_latency_seconds`, `api_prediction_total`, `api_prediction_errors_total`, and `data_drift_score`.
