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

import mlflow.tensorflow

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import warnings

from tensorflow.keras.callbacks import EarlyStopping



experiment_name = 'Reviews_Classification'
mlflow.set_experiment(experiment_name)
mlflow.tensorflow.autolog()


# parms for RNN model
VOCAB_SIZE = 1000
EMBEDDING_DIM = 16
MAX_LEN = 120
TRUNC_TYPE = 'post'
PADDING_TYPE = 'post'
OOV_TOK = "<OOV>"
NUM_EPOCH = 20

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



debug = True

if debug == True:
    print(os.path.expandvars('${MLFLOW_TRACKING_URI}'))
    print(os.path.expandvars('${MLFLOW_ARTIFACT_URI}'))
    print(os.path.expandvars('${AWS_ACCESS_KEY_ID}'))
    print(os.path.expandvars('${AWS_SECRET_ACCESS_KEY}'))
    print(os.path.expandvars('${MLFLOW_S3_ENDPOINT_URL}'))

    experiment_id = mlflow.set_experiment(experiment_name)

    experiment = mlflow.get_experiment(experiment_id)

    print("Name: {}".format(experiment.name))
    print("Experiment_id: {}".format(experiment.experiment_id))
    print("Artifact Location: {}".format(experiment.artifact_location))
    print("Tags: {}".format(experiment.tags))
    print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))

    mr_uri = mlflow.get_registry_uri()
    print("Current model registry uri: {}".format(mr_uri))

    # Get the current tracking uri
    tracking_uri = mlflow.get_tracking_uri()
    print("Current tracking uri: {}".format(tracking_uri))

    from mlflow.tracking import MlflowClient


    def print_auto_logged_info(r):
        tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
        artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
        print("run_id: {}".format(r.info.run_id))
        print("artifacts: {}".format(artifacts))
        print("params: {}".format(r.data.params))
        print("metrics: {}".format(r.data.metrics))
        print("tags: {}".format(tags))