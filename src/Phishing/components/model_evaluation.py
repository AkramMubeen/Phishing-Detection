import os
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.Phishing.utils.utils import load_object
import json

class ModelEvaluation:
    def __init__(self):
        pass

    def eval_metrics(self, actual, pred):
        accuracy = accuracy_score(actual, pred)
        precision = precision_score(actual, pred)
        recall = recall_score(actual, pred)
        f1 = f1_score(actual, pred)
        roc_auc = roc_auc_score(actual, pred)
        return accuracy, precision, recall, f1, roc_auc

    def initiate_model_evaluation(self, train_array, test_array):
        try:
            X_test, y_test = (test_array[:, :-1], test_array[:, -1])

            model_path = os.path.join("artifacts", "model.pkl")
            model = load_object(model_path)

            mlflow.set_registry_uri("https://dagshub.com/AkramMubeen/Phishing-Detection.mlflow")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            print(tracking_url_type_store)

            with mlflow.start_run():
                predicted_qualities = model.predict(X_test)
                accuracy, precision, recall, f1, roc_auc = self.eval_metrics(y_test, predicted_qualities)

                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("roc_auc", roc_auc)

                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(model, "model", registered_model_name="ml_model")
                else:
                    mlflow.sklearn.log_model(model, "model")
            
            # Serialize metrics to JSON file
            metrics_dict = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "roc_auc": roc_auc
            }
            with open('scores.json', 'w') as json_file:
                json.dump(metrics_dict, json_file)

        except Exception as e:
            print(f"An error occurred: {e}")
            raise e
