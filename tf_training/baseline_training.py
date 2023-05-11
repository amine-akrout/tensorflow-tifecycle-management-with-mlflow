"""
Trains and evaluate a Baseline Model.
"""
# pylint: disable=E0401, C0103, E1120
import datetime

import mlflow
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# import shutil

# Create Baseline Model

tf.keras.preprocessing.text.sequence.pad_sequences()

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

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(X_train)
    train_sequences = tokenizer.texts_to_sequences(X_train)
    train_padded = tf.keras.preprocessing.text.sequence.pad_sequences(
        train_sequences, padding=padding_type, maxlen=max_length
    )
    test_sequences = tokenizer.texts_to_sequences(X_test)
    test_padded = tf.keras.preprocessing.text.sequence.pad_sequences(
        test_sequences, padding=padding_type, maxlen=max_length
        )
    return train_padded, test_padded


def create_model(input_dim, output_dim, input_length, metrics):
    """
    Create a simple MLP model

    Args:
        input_dim (int): Input dimension
        output_dim (int): Output dimension
        input_length (int): Input length
        metrics (list): List of metrics

    Returns:
        tf.keras.Model: Compiled model
    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(
                input_dim=input_dim, output_dim=output_dim, input_length=input_length
            ),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(24, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=metrics)
    return model


def train_model(model, train_padded, test_padded, y_train, y_test, num_epochs):
    """
    Train the model and log metrics to MLflow tracking server

    Args:
        model (tf.keras.Model): Compiled model
        train_padded (numpy.ndarray): Training data
        test_padded (numpy.ndarray): Test data
        y_train (numpy.ndarray): Training labels
        y_test (numpy.ndarray): Test labels
        num_epochs (int): Number of epochs

    Returns:
        None
    """
    model_name = "baseline"
    log_dir = "logs/fit/{}-".format(model_name) + datetime.datetime.now().strftime(
        "%d%m%y-%H_%M_%S"
    )
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=3)
    with mlflow.start_run(run_name=model_name) as run:
        _ = model.fit(
            train_padded,
            y_train,
            epochs=num_epochs,
            validation_data=(test_padded, y_test),
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
    num_epoch = 20
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
    model = create_model(vocab_size, embedding_dim, max_len, metrics)
    train_model(model, train_padded, test_padded, y_train, y_test, num_epoch)


if __name__ == "__main__":
    main()
