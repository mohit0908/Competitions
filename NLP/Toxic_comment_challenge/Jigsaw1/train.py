# import necessary libraries

import pandas as pd
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import model
import config

max_features = config.max_features
maxlen = config.maxlen
batch_size = config.batch_size
epochs = config.epochs

# Read training file
train = pd.read_csv('datasets/train.csv')

prediction_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[prediction_classes]

train_sentences = train.comment_text

tokenizer = Tokenizer(num_words = max_features)
tokenizer.fit_on_texts(list(train_sentences))
list_tokenizer_train = tokenizer.texts_to_sequences(train_sentences)

# pad sequences to a fixed length input.
X_train = pad_sequences(list_tokenizer_train, maxlen = maxlen)

# Calling model creation
model = model.create_model()

# Running model for 10 epochs 
model.fit(X_train, y, batch_size=batch_size, epochs=epochs, validation_split=0.1)

# Save model weights
model.save_weights('trained_model/keras_model_weights.h5')
print('Model weights loaded')