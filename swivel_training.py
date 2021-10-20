#!/usr/bin/env python
# coding: utf-8

"""
Trains and evaluate a swivel Model.
"""

from utils import *
import tensorflow as tf
import datetime
import tensorflow_hub as hub
import tensorflow_text as text

X_train, X_test, y_train, y_test = import_data()
# Create Model using gnews-swivel-20dim
##
hub_layer = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim-with-oov/1", output_shape=[20],input_shape=[], dtype=tf.string, trainable=True)

model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=METRICS)
num_epoch = NUM_EPOCH +10
if __name__ == '__main__':
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%d%m%y-%H_%M_%S")
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=num_epoch, callbacks=[tensorboard])