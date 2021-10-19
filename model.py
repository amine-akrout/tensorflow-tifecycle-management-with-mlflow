# Import Libaries
##
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_hub as hub
import tensorflow_text as text


from sklearn.model_selection import train_test_split

import mlflow
import mlflow.tensorflow
from mlflow import pyfunc

import datetime

# parms
##
vocab_size = 1000
embedding_dim = 16
max_length = 120
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"

# Import Data
##
data = pd.read_csv('./data/tripadvisor_hotel_reviews.csv')
print(data.head())

# group ratings pos vs neg
##
groups = []
for rating in data['Rating']:
    if rating in [1,2,3]:
        groups.append(0)
    else:
        groups.append(1)
data['sentiment'] = groups


# split data to train/test
##
X_train, X_test, y_train, y_test = train_test_split(data['Review'], data['sentiment'], test_size=0.2, random_state=67, stratify=data['sentiment'])
print(len(X_train))
print(len(X_test))
print(len(y_train))
print(len(y_test))

# Create Tokenizer and fit to train sentences
##
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(X_train)
train_padded = pad_sequences(train_sequences, padding=padding_type, maxlen=max_length)

print(len(train_sequences[0]))
print(len(train_padded[0]))

print(len(train_sequences[1]))
print(len(train_padded[1]))

print(len(train_sequences[10]))
print(len(train_padded[10]))

# Aplly Tokenizer and padding on test sentences
##
validation_sequences = tokenizer.texts_to_sequences(X_test)
validation_padded = pad_sequences(validation_sequences, padding=padding_type, maxlen=max_length)

print(len(validation_sequences))
print(validation_padded.shape)

# Create Baseline Model
##
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

##
num_epochs = 20
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%d%m%y-%H_%M_%S")
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
history = model.fit(train_padded, y_train, epochs=num_epochs, validation_data=(validation_padded, y_test), verbose=1, callbacks=[tensorboard])

##
tf.keras.backend.clear_session()

# Create CNN model
##
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Conv1D(128, 5, activation='relu'),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
num_epochs = 20

train_padded = np.array(train_padded)
training_labels = np.array(y_train)
testing_padded = np.array(validation_padded)
testing_labels = np.array(y_test)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%d%m%y-%H_%M_%S")
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(train_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels), verbose=1, callbacks=[tensorboard])


##
tf.keras.backend.clear_session()

# Create RNN model
##
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

num_epochs = 20
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%d%m%y-%H_%M_%S")
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(train_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels), verbose=1, callbacks=[tensorboard])

##
tf.keras.backend.clear_session()

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
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

# fit the model
##
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%d%m%y-%H_%M_%S")
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(X_train, y_train,validation_data=(X_test, y_test), epochs = 1, callbacks=[tensorboard], batch_size=32)

##
tf.keras.backend.clear_session()


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

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

# fit the model
##
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%d%m%y-%H_%M_%S")
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(X_train, y_train,validation_data=(X_test, y_test), epochs = 20, callbacks=[tensorboard])

##

