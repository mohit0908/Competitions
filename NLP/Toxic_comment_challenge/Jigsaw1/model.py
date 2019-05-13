# Model architecture details


import keras
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalAveragePooling1D,GlobalMaxPool1D
from keras.models import Model, load_model
from keras import initializers, regularizers, constraints, optimizers, layers
import config


# Model building

max_features = config.max_features

def create_model():
    # Define input layer
    inp = Input(shape=(200, ))

    # Define embedding layer
    embed_size = 128
    x = Embedding(input_dim = max_features, output_dim=embed_size)(inp)
    x = LSTM(60, return_sequences=True, name = 'lstm_layer1')(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(50, activation = 'relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation = 'sigmoid')(x)

    model = Model(inputs = inp, outputs = x)
    model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    return model

