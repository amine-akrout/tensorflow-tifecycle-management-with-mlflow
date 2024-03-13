"""
Select the best model from the MLflow experiment and save it to a directory""
"""

import os

import mlflow


# Function to find the best model
def get_best_model(experiment_name):
    """Get the best model from the MLflow experiment"""
    runs = mlflow.search_runs(experiment_names=[experiment_name])
    best_run = runs.sort_values("metrics.val_recall", ascending=False).iloc[0]
    print(f"best model: ", best_run["tags.mlflow.runName"])
    return best_run


def save_model(best_run, model_path="../tf_serving/tmp/best_model"):
    """Save the best model to a directory"""
    # Model artifacts
    model_uri = best_run["artifact_uri"] + "/model"
    os.makedirs(model_path, exist_ok=True)
    mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=model_path)


def select_best_model():
    """Select the best model from the MLflow experiment and save it to a directory"""
    # MLflow server details
    mlflow.set_tracking_uri("http://localhost:5000")
    experiment_name = "Reviews_Classification"
    best_run = get_best_model(experiment_name)
    save_model(best_run)


if __name__ == "__main__":
    select_best_model()
