"""Model evaluation metrics: ROC-AUC, PR-AUC, F1, and Confusion Matrix."""
import json
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix
from src.models.train import train_churn_model

def evaluate_performance():
    pipeline, X_test, y_test = train_churn_model()
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "pr_auc": float(average_precision_score(y_test, y_prob)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "report": classification_report(y_test, y_pred, output_dict=True)
    }
    print("Evaluation Results:")
    print(json.dumps(metrics, indent=2))
    return metrics

if __name__ == "__main__":
    evaluate_performance()
