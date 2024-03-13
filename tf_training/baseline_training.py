"""
Trains and evaluate a Baseline Model.
"""

import warnings

import tensorflow as tf
from utils import import_data, split_data, preprocess_text, train_model

warnings.filterwarnings("ignore")

# Create Baseline Model


def create_mlp_model(input_dim, output_dim, input_length, metrics):
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


def run_baseline():
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
    data = import_data()
    x_train, x_test, y_train, y_test = split_data(data)
    train_padded, test_padded = preprocess_text(
        x_train, x_test, vocab_size, oov_tok, padding_type, max_len
    )
    model = create_mlp_model(vocab_size, embedding_dim, max_len, metrics)
    # train_model("baseline", model, train_padded, test_padded, y_train, y_test, num_epoch)
    train_model(
        "baseline", model, train_padded, y_train, test_padded, y_test, num_epoch
    )


if __name__ == "__main__":
    run_baseline()
