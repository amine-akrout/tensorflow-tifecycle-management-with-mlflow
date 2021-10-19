# Import Libaries
##
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split

import mlflow
import mlflow.tensorflow
from mlflow import pyfunc

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
num_epochs = 30
history = model.fit(train_padded, y_train, epochs=num_epochs, validation_data=(validation_padded, y_test), verbose=1)

##

