"""
Trains and evaluate a bert Model.
"""

import datetime
import warnings

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from utils import import_data, split_data

import mlflow

warnings.filterwarnings("ignore")


# define model


def create_model(bert_preprocessor, bert_encoder, metrics):
    """
    Create a BERT model for sentiment analysis

    Args:
        bert_preprocessor (tf.keras.layers): Preprocessor layer
        bert_encoder (tf.keras.layers): Encoder layer
        metrics (list): List of metrics

    Returns:
        tf.keras.Model: BERT model
    """
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="Inputs")
    preprocessed_text = bert_preprocessor(text_input)
    embedded = bert_encoder(preprocessed_text)
    dropout = tf.keras.layers.Dropout(0.1, name="Dropout")(embedded["pooled_output"])
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="Dense")(dropout)

    model = tf.keras.Model(inputs=[text_input], outputs=[outputs])
    # check the summary of the model
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=metrics)
    return model


# fit the model


def train_model(model, x_train, y_train, x_test, y_test, num_epochs):
    """
    Train the model and log metrics to mlflow

    Args:
        model (tf.keras.Model): Model to train
        x_train (list): Training data
        y_train (list): Training labels
        x_test (list): Test data
        y_test (list): Test labels
        num_epochs (int): Number of epochs
    """
    # pylint: disable=E1120
    model_name = "Bert"
    log_dir = "logs/fit/{}-".format(model_name) + datetime.datetime.now().strftime(
        "%d%m%y-%H_%M_%S"
    )
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", verbose=1, patience=3
    )
    with mlflow.start_run(run_name=model_name):
        _ = model.fit(
            x_train,
            y_train,
            validation_data=(x_test, y_test),
            epochs=num_epochs,
            callbacks=[tensorboard, early_stop],
        )
        model.save("./tmp/{}".format(model_name))
        mlflow.tensorflow.log_model(model, artifact_path="saved_model")
    tf.keras.backend.clear_session()


def run_bert():
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
    x_train, x_test, y_train, y_test = split_data(data)
    # Create Bert Model
    # downloading preprocessing files and model
    bert_preprocessor = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
    )
    bert_encoder = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/2"
    )
    model = create_model(bert_preprocessor, bert_encoder, metrics)
    train_model(model, x_train, y_train, x_test, y_test, num_epoch)


if __name__ == "__main__":
    run_bert()
