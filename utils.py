#!/usr/bin/env python
# coding: utf-8

'''Trains and evaluate a simple MLP
on the Reuters newswire topic classification task.
'''
# Import Libaries
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
from sklearn.model_selection import train_test_split

### mlflow configs
import mlflow
import mlflow.tensorflow
from mlflow import pyfunc
# Setup Experiment Tracker
#registry_uri = 'sqlite:///mlflow.db'
registry_uri = os.path.expandvars('mysql+pymysql://${MYSQL_USER}:${MYSQL_PASSWORD}@localhost:3306/${MYSQL_DATABASE}')
mlflow.tracking.set_registry_uri(registry_uri)

tracking_uri = 'http://localhost:5000'
mlflow.tracking.set_tracking_uri(tracking_uri)

experiment_name = 'Reviews_Classification'
mlflow.set_experiment(experiment_name)

mlflow.tensorflow.autolog()



# parms
VOCAB_SIZE = 1000
EMBEDDING_DIM = 16
MAX_LEN = 120
TRUNC_TYPE = 'post'
PADDING_TYPE = 'post'
OOV_TOK = "<OOV>"
NUM_EPOCH = 10

# Import Data

def import_data():
    data = pd.read_csv('./data/tripadvisor_hotel_reviews.csv')
    groups = []
    for rating in data['Rating']:
        if rating in [1, 2, 3]:
            groups.append(0)
        else:
            groups.append(1)
    data['sentiment'] = groups
    X_train, X_test, y_train, y_test = train_test_split(data['Review'], data['sentiment'], test_size=0.2,
                                                        random_state=123, stratify=data['sentiment'])
    return X_train, X_test, y_train, y_test



# Create Tokenizer and fit to train sentences
def tok_pad(X_train, X_test, vocab_size, oov_tok, padding_type, max_length):
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(X_train)
    train_sequences = tokenizer.texts_to_sequences(X_train)
    train_padded = pad_sequences(train_sequences, padding=padding_type, maxlen=max_length)
    test_sequences = tokenizer.texts_to_sequences(X_test)
    test_padded = pad_sequences(test_sequences, padding=padding_type, maxlen=max_length)
    return train_padded, test_padded


METRICS = [tf.keras.metrics.BinaryAccuracy(name='accuracy'),
           tf.keras.metrics.Precision(name='precision'),
           tf.keras.metrics.Recall(name='recall'),
           tf.keras.metrics.AUC(name='auc')
]