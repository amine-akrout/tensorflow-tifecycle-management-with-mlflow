"""
Trains and evaluate a swivel Model.
"""

import datetime
import warnings

import mlflow
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from utils import import_data, split_data


warnings.filterwarnings("ignore")

# pylint: disable=E0401, C0103, E1120


# Create Model using gnews-swivel-20dim

def create_model(hub_layer, metrics):
    """
    Create a swivel model for sentiment analysis

    Args:
        hub_layer (tf.keras.layers): Hub layer
        metrics (list): List of metrics

    Returns:
        tf.keras.Model: swivel model
    """
    model = tf.keras.Sequential()
    model.add(hub_layer)
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(32, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(16, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=metrics)
    return model


def train_model(model, X_train, y_train, X_test, y_test, num_epoch):
    """
    Train a swivel model

    Args:
        model (tf.keras.Model): Model to train
        X_train (list): Training data
        y_train (list): Training labels
        X_test (list): Test data
        y_test (list): Test labels
        num_epoch (int): Number of epochs

    Returns:
        tf.keras.Model: Trained model
    """
    model_name = "swivel"
    log_dir = "logs/fit/{}-".format(model_name) + datetime.datetime.now().strftime(
        "%d%m%y-%H_%M_%S"
    )
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", verbose=1, patience=3
    )
    with mlflow.start_run(run_name=model_name) as run:
        _ = model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=num_epoch,
            callbacks=[tensorboard, es],
        )
        model.save("./tmp/{}".format(model_name))
        mlflow.tensorflow.log_model(model, artifact_path="saved_model")
    tf.keras.backend.clear_session()


def main():
    """
    Main function
    """
    num_epoch = 1
    metrics = [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ]
    data = import_data()
    X_train, X_test, y_train, y_test = split_data(data)
    hub_layer = hub.KerasLayer(
        "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim-with-oov/1",
        output_shape=[20],
        input_shape=[],
        dtype=tf.string,
        trainable=True,
    )
    model = create_model(hub_layer, metrics)
    train_model(model, X_train, y_train, X_test, y_test, num_epoch)


if __name__ == "__main__":
    main()
