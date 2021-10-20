#!/usr/bin/env python
# coding: utf-8

"""
Trains and evaluate a Baseline Model.
"""

from utils import *
import tensorflow as tf
import numpy as np
import datetime

X_train, X_test, y_train, y_test = import_data()
train_padded, test_padded = tok_pad(X_train=X_train, X_test=X_test, vocab_size=VOCAB_SIZE, oov_tok=OOV_TOK, padding_type=PADDING_TYPE, max_length=MAX_LEN)

# Create CNN model
##
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN),
    tf.keras.layers.Conv1D(128, 5, activation='relu'),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=METRICS)
model.summary()
num_epochs = NUM_EPOCH

train_padded = np.array(train_padded)
train_labels = np.array(y_train)
test_padded = np.array(test_padded)
test_labels = np.array(y_test)

if __name__=='__main__':
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%d%m%y-%H_%M_%S")
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(train_padded, train_labels, epochs=num_epochs, validation_data=(test_padded, test_labels), verbose=1, callbacks=[tensorboard])
    tf.keras.backend.clear_session()