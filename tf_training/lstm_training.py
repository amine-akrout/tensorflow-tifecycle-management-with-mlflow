"""
Trains and evaluate a LSTM Model
"""

import warnings

import numpy as np
import tensorflow as tf
from utils import import_data, preprocess_text, split_data, train_model

warnings.filterwarnings("ignore")


# Create RNN model
def create_lstm_model(input_dim, output_dim, input_length, metrics):
    """
    Create RNN model for training and evaluation of sentiment analysis task

    Args:
        input_dim (int): Input dimension
        output_dim (int): Output dimension
        input_length (int): Input length
        metrics (list): List of metrics

    Returns:
        tf.keras.Sequential: RNN model
    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(
                input_dim=input_dim, output_dim=output_dim, input_length=input_length
            ),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
            tf.keras.layers.Dense(24, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=metrics)
    return model


def run_lstm():
    """
    Main function
    """
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
    test_padded = np.array(test_padded)
    model = create_lstm_model(vocab_size, embedding_dim, max_len, metrics)
    train_model(
        "baseline", model, train_padded, y_train, test_padded, y_test, num_epoch
    )


if __name__ == "__main__":
    run_lstm()
