# Customer Churn Prediction & Retention Analytics

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

An enterprise-ready machine learning service for identifying customer attrition risk, quantifying feature impact, and operationalizing real-time retention triggers.

## Quickstart
```bash
git clone https://github.com/GenoJ83/customer-churn-prediction.git
cd customer-churn-prediction
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python src/models/train.py
uvicorn src.api.app:app --reload
```

## Evaluation Results
- **ROC-AUC**: 0.842
- **PR-AUC**: 0.718
- **Inference Latency**: 12ms / request
