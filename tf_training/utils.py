'''Trains and evaluate a simple MLP
on the Reuters newswire topic classification task.
'''
import os
import warnings
import mlflow.tensorflow
from dotenv import find_dotenv, load_dotenv
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split

load_dotenv(find_dotenv())
experiment_name = 'Reviews_Classification'
mlflow.set_experiment(experiment_name)
mlflow.tensorflow.autolog()


debug = True
def debug_print(*args, **kwargs):
    if debug == True:
        print(os.path.expandvars('${MLFLOW_TRACKING_URI}'))
        print(os.path.expandvars('${MLFLOW_ARTIFACT_URI}'))
        print(os.path.expandvars('${AWS_ACCESS_KEY_ID}'))
        print(os.path.expandvars('${AWS_SECRET_ACCESS_KEY}'))
        print(os.path.expandvars('${MLFLOW_S3_ENDPOINT_URL}'))
        experiment_id = mlflow.set_experiment(experiment_name)

        experiment = mlflow.get_experiment(experiment_id)

        print("Name: {}".format(experiment.name))
        print("Experiment_id: {}".format(experiment.experiment_id))
        print("Artifact Location: {}".format(experiment.artifact_location))
        print("Tags: {}".format(experiment.tags))
        print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))

        mr_uri = mlflow.get_registry_uri()
        print("Current model registry uri: {}".format(mr_uri))

        # Get the current tracking uri
        tracking_uri = mlflow.get_tracking_uri()
        print("Current tracking uri: {}".format(tracking_uri))

        def print_auto_logged_info(r):
            tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
            artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
            print("run_id: {}".format(r.info.run_id))
            print("artifacts: {}".format(artifacts))
            print("params: {}".format(r.data.params))
            print("metrics: {}".format(r.data.metrics))
            print("tags: {}".format(tags))