"""
Trains and evaluate a Baseline Model.
"""

import datetime
import warnings

import numpy as np
import pandas as pd
import mlflow

# pylint: disable=E0401, C0103, E1120
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

warnings.filterwarnings("ignore")


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

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(X_train)
    train_sequences = tokenizer.texts_to_sequences(X_train)
    train_padded = pad_sequences(
        train_sequences, padding=padding_type, maxlen=max_length
    )
    test_sequences = tokenizer.texts_to_sequences(X_test)
    test_padded = pad_sequences(test_sequences, padding=padding_type, maxlen=max_length)
    return train_padded, test_padded


# Create CNN model


def create_model(vocab_size, embedding_dim, max_length, metrics):
    """
    Create CNN model

    Args:
        vocab_size (int): Size of vocabulary
        embedding_dim (int): Embedding dimension
        max_length (int): Maximum length of sequence
        metrics (list): List of metrics to evaluate model

    Returns:
        tf.keras.Sequential: CNN model
    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(
                vocab_size, embedding_dim, input_length=max_length
            ),
            tf.keras.layers.Conv1D(128, 5, activation="relu"),
            tf.keras.layers.GlobalMaxPooling1D(),
            tf.keras.layers.Dense(24, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=metrics)
    return model


def train_model(
    model, train_padded, train_labels, test_padded, test_labels, num_epochs
):
    """
    Train CNN model and log metrics to MLflow

    Args:
        model (tf.keras.Sequential): CNN model
        train_padded (list): Preprocessed training data
        train_labels (list): Training labels
        test_padded (list): Preprocessed test data
        test_labels (list): Test labels
        num_epochs (int): Number of epochs
    """
    model_name = "CNN"
    log_dir = "logs/fit/{}-".format(model_name) + datetime.datetime.now().strftime(
        "%d%m%y-%H_%M_%S"
    )
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)
    es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=3)
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


def main():
    """Main function"""
    vocab_size = 1000
    embedding_dim = 16
    max_len = 120
    padding_type = "post"
    oov_tok = "<oov>"
    num_epoch = 2
    metrics = [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ]
    X_train, X_test, y_train, y_test = import_data()
    train_padded, test_padded = preprocess_text(
        X_train, X_test, vocab_size, oov_tok, padding_type, max_len
    )
    train_padded = np.array(train_padded)
    train_labels = np.array(y_train)
    test_padded = np.array(test_padded)
    test_labels = np.array(y_test)
    model = create_model(vocab_size, embedding_dim, max_len, metrics)
    train_model(model, train_padded, train_labels, test_padded, test_labels, num_epoch)


if __name__ == "__main__":
    main()
