import pandas as pd

import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


import utils
import config
import model as md

max_features = config.max_features
maxlen = config.maxlen

train = pd.read_csv('datasets/train.csv')
test = pd.read_csv('datasets/test.csv')


prediction_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[prediction_classes]

train_sentences = train.comment_text
test_sentences = test.comment_text


tokenizer = Tokenizer(num_words = max_features)
tokenizer.fit_on_texts(list(train_sentences))
list_tokenizer_test = tokenizer.texts_to_sequences(test_sentences)

X_test = pad_sequences(list_tokenizer_test, maxlen = maxlen)

# Calling model creation
model = md.create_model()

# Load saved model weights
model.load_weights('trained_model/keras_model_weights.h5')
print('Trained weights loaded')

print('Running predictions on test data')
output = model.predict(X_test)
print('Predictions calculated')




utils.make_submission_csv(test, output, prediction_classes)