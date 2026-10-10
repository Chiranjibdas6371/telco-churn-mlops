import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import app

SAMPLE = {
    "gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No","tenure":12,
    "PhoneService":"Yes","MultipleLines":"No","InternetService":"DSL","OnlineSecurity":"No",
    "OnlineBackup":"Yes","DeviceProtection":"No","TechSupport":"No","StreamingTV":"No",
    "StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes",
    "PaymentMethod":"Electronic check","MonthlyCharges":70.35,"TotalCharges":844.2
}

def test_health():
    c = app.test_client()
    r = c.get('/health')
    assert r.status_code == 200

def test_predict_shape():
    c = app.test_client()
    r = c.post('/predict', json=SAMPLE)
    assert r.status_code in (200, 503)
