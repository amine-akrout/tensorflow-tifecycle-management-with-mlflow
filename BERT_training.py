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

## Create Bert Model
# downloading preprocessing files and model
bert_preprocessor = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3')
bert_encoder = hub.KerasLayer('https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/2')
#bert_encoder = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4')
# https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1

# define model
text_input = tf.keras.layers.Input(shape = (), dtype = tf.string, name = 'Inputs')
preprocessed_text = bert_preprocessor(text_input)
embedded = bert_encoder(preprocessed_text)
dropout = tf.keras.layers.Dropout(0.1, name = 'Dropout')(embedded['pooled_output'])
outputs = tf.keras.layers.Dense(1, activation = 'sigmoid', name = 'Dense')(dropout)

model = tf.keras.Model(inputs = [text_input], outputs = [outputs])
# check the summary of the model
model.summary()
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=METRICS)

# fit the model

num_epochs=NUM_EPOCH + 10
if __name__=='__main__':
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%d%m%y-%H_%M_%S")
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(X_train, y_train,validation_data=(X_test, y_test), epochs = num_epochs, callbacks=[tensorboard],)
    tf.keras.backend.clear_session()
