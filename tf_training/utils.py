"""Trains and evaluate a simple MLP
on the Reuters newswire topic classification task.
"""

import datetime

import mlflow.tensorflow
import pandas as pd
import tensorflow as tf
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import train_test_split

load_dotenv(find_dotenv())
EXPERIMENT = "Reviews_Classification"
mlflow.set_experiment(EXPERIMENT)
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
    x_train, x_test, y_train, y_test = train_test_split(
        data["Review"],
        data["sentiment"],
        test_size=0.2,
        random_state=123,
        stratify=data["sentiment"],
    )
    return x_train, x_test, y_train, y_test


def preprocess_text(x_train, x_test, vocab_size, oov_tok, padding_type, max_length):
    """
    Preprocess text data for training

    Args:
        x_train (list): Training data
        x_test (list): Test data
        vocab_size (int): Size of vocabulary
        oov_tok (str): Out of vocabulary token
        padding_type (str): Padding type
        max_length (int): Maximum length of sequence

    Returns:
        tuple: Tuple of preprocessed training and test data
    """
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=vocab_size, oov_token=oov_tok
    )
    tokenizer.fit_on_texts(x_train)
    train_sequences = tokenizer.texts_to_sequences(x_train)
    train_padded = tf.keras.utils.pad_sequences(
        train_sequences, padding=padding_type, maxlen=max_length
    )
    test_sequences = tokenizer.texts_to_sequences(x_test)
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
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", verbose=1, patience=3
    )
    with mlflow.start_run(run_name=model_name):
        _ = model.fit(
            train_padded,
            train_labels,
            epochs=num_epochs,
            validation_data=(test_padded, test_labels),
            verbose=1,
            callbacks=[tensorboard, early_stop],
        )
        model.save("./tmp/{}".format(model_name))
        mlflow.tensorflow.log_model(model, artifact_path="saved_model")
    tf.keras.backend.clear_session()
