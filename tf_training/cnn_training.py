"""
Trains and evaluate a Baseline Model.
"""

import warnings

import numpy as np
import tensorflow as tf
from utils import import_data, preprocess_text, split_data, train_model

warnings.filterwarnings("ignore")


# Create CNN model


def create_cnn_model(vocab_size, embedding_dim, max_length, metrics):
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


def run_cnn():
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
    data = import_data()
    x_train, x_test, y_train, y_test = split_data(data)
    train_padded, test_padded = preprocess_text(
        x_train, x_test, vocab_size, oov_tok, padding_type, max_len
    )
    train_padded = np.array(train_padded)
    train_labels = np.array(y_train)
    test_padded = np.array(test_padded)
    test_labels = np.array(y_test)
    model = create_cnn_model(vocab_size, embedding_dim, max_len, metrics)
    train_model(
        "cnn", model, train_padded, train_labels, test_padded, test_labels, num_epoch
    )


if __name__ == "__main__":
    run_cnn()
