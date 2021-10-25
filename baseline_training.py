#!/usr/bin/env python
# coding: utf-8

"""
Trains and evaluate a Baseline Model.
"""
from utils import *
import tensorflow as tf
import datetime
import shutil

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


###
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
                                    registered_model_name=model_name)

        shutil.rmtree("./tmp")

    tf.keras.backend.clear_session()

    print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))