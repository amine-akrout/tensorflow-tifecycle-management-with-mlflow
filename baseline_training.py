#!/usr/bin/env python
# coding: utf-8

"""
Trains and evaluate a Baseline Model.
"""
from utils import *
import tensorflow as tf
import datetime
from dotenv import load_dotenv, find_dotenv
import shutil


load_dotenv(find_dotenv())
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

X_train, X_test, y_train, y_test = import_data()
train_padded, test_padded = tok_pad(X_train=X_train, X_test=X_test, vocab_size=VOCAB_SIZE, oov_tok=OOV_TOK, padding_type=PADDING_TYPE, max_length=MAX_LEN)
# Create Baseline Model
##
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=METRICS)
model.summary()



##################
from mlflow.tracking import MlflowClient

def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print("run_id: {}".format(r.info.run_id))
    print("artifacts: {}".format(artifacts))
    print("params: {}".format(r.data.params))
    print("metrics: {}".format(r.data.metrics))
    print("tags: {}".format(tags))
########################################


num_epochs = NUM_EPOCH-9
if __name__=='__main__':
    from urllib.parse import urlparse
    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
    print(mlflow.get_tracking_uri())
    model_name = 'baseline'
    log_dir = "logs/fit/{}-".format(model_name) + datetime.datetime.now().strftime("%d%m%y-%H_%M_%S")
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    with mlflow.start_run() as run:
        history = model.fit(train_padded, y_train, epochs=num_epochs, validation_data=(test_padded, y_test), verbose=1, callbacks=[tensorboard])
        model.save("./tmp")

        mlflow.tensorflow.log_model(tf_saved_model_dir='./tmp',
                                    tf_meta_graph_tags='serve',
                                    tf_signature_def_key='serving_default',
                                    artifact_path='saved_model',
                                    registered_model_name='Baseline Model')

        shutil.rmtree("./tmp")

    tf.keras.backend.clear_session()

    print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))