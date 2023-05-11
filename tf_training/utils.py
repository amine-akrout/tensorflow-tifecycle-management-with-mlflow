'''Trains and evaluate a simple MLP
on the Reuters newswire topic classification task.
'''
import os
import datetime
import tensorflow as tf
import mlflow.tensorflow
from dotenv import find_dotenv, load_dotenv
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
# from tensorflow.python.keras.utils import pad_sequences
import pandas as pd
load_dotenv(find_dotenv())
experiment_name = 'Reviews_Classification'
mlflow.set_experiment(experiment_name)
mlflow.tensorflow.autolog()

def import_data():
    """
    Import data from csv file

    Returns:
        tuple: Tuple of training and test data
    """
    data = pd.read_csv("./data/tripadvisor_hotel_reviews.csv")
    groups = []
    for rating in data["Rating"]:
        if rating in [1, 2, 3]:
            groups.append(0)
        else:
            groups.append(1)
    data["sentiment"] = groups
    return data

def split_data(data):
    """
    Split data into training and test data

    Args:
        data (pd.DataFrame): Data to split

    Returns:
        tuple: Tuple of training and test data
    """
    X_train, X_test, y_train, y_test = train_test_split(
        data["Review"],
        data["sentiment"],
        test_size=0.2,
        random_state=123,
        stratify=data["sentiment"],
    )
    return X_train, X_test, y_train, y_test

def preprocess_text(X_train, X_test, vocab_size, oov_tok, padding_type, max_length):
    """
    Preprocess text data for training

    Args:
        X_train (list): Training data
        X_test (list): Test data
        vocab_size (int): Size of vocabulary
        oov_tok (str): Out of vocabulary token
        padding_type (str): Padding type
        max_length (int): Maximum length of sequence

    Returns:
        tuple: Tuple of preprocessed training and test data
    """
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(X_train)
    train_sequences = tokenizer.texts_to_sequences(X_train)
    train_padded = tf.keras.utils.pad_sequences(
        train_sequences, padding=padding_type, maxlen=max_length
    )
    test_sequences = tokenizer.texts_to_sequences(X_test)
    test_padded = tf.keras.utils.pad_sequences(
        test_sequences, padding=padding_type, maxlen=max_length
        )
    return train_padded, test_padded


def train_model(
    model_name, model, train_padded, train_labels, test_padded, test_labels, num_epochs
):
    """
    Train DL model and log metrics to MLflow

    Args:
        model_name (str): Name of model
        model (tf.keras.Sequential): CNN model
        train_padded (list): Preprocessed training data
        train_labels (list): Training labels
        test_padded (list): Preprocessed test data
        test_labels (list): Test labels
        num_epochs (int): Number of epochs
    """
    log_dir = "logs/fit/{}-".format(model_name) + datetime.datetime.now().strftime(
        "%d%m%y-%H_%M_%S"
    )
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=3)
    with mlflow.start_run(run_name=model_name) as run:
        _ = model.fit(
            train_padded,
            train_labels,
            epochs=num_epochs,
            validation_data=(test_padded, test_labels),
            verbose=1,
            callbacks=[tensorboard, es],
        )
        model.save("./tmp/{}".format(model_name))
        mlflow.tensorflow.log_model(model, artifact_path="saved_model")
        # shutil.rmtree("./tmp")
    tf.keras.backend.clear_session()

# debug = True
# def debug_print(*args, **kwargs):
#     if debug == True:
#         print(os.path.expandvars('${MLFLOW_TRACKING_URI}'))
#         print(os.path.expandvars('${MLFLOW_ARTIFACT_URI}'))
#         print(os.path.expandvars('${AWS_ACCESS_KEY_ID}'))
#         print(os.path.expandvars('${AWS_SECRET_ACCESS_KEY}'))
#         print(os.path.expandvars('${MLFLOW_S3_ENDPOINT_URL}'))
#         experiment_id = mlflow.set_experiment(experiment_name)

#         experiment = mlflow.get_experiment(experiment_id)

#         print("Name: {}".format(experiment.name))
#         print("Experiment_id: {}".format(experiment.experiment_id))
#         print("Artifact Location: {}".format(experiment.artifact_location))
#         print("Tags: {}".format(experiment.tags))
#         print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))

#         mr_uri = mlflow.get_registry_uri()
#         print("Current model registry uri: {}".format(mr_uri))

#         # Get the current tracking uri
#         tracking_uri = mlflow.get_tracking_uri()
#         print("Current tracking uri: {}".format(tracking_uri))

#         def print_auto_logged_info(r):
#             tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
#             artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
#             print("run_id: {}".format(r.info.run_id))
#             print("artifacts: {}".format(artifacts))
#             print("params: {}".format(r.data.params))
#             print("metrics: {}".format(r.data.metrics))
#             print("tags: {}".format(tags))