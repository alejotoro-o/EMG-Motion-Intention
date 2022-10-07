################################################################
##### MODELOS DE DEEP LEARNING IMPLEMENTADOS EN TENSORFLOW #####
################################################################

from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv2D, MaxPool2D, Flatten, GRU, attention, Bidirectional

# Multi layer perceptron
def mlp():

    mlp_model = Sequential()
    mlp_model.add(Dense(32, activation='relu'))
    mlp_model.add(Dense(16, activation='relu'))
    mlp_model.add(Dense(5, activation='softmax'))

    return mlp_model

# Convolutional neural network
def cnn():

    cnn_model = Sequential()
    cnn_model.add(Conv2D(16, (10,2), activation='relu', padding='same'))
    cnn_model.add(MaxPool2D((4,1), padding='same'))
    cnn_model.add(Conv2D(8, (10,2), activation='relu', padding='same'))
    cnn_model.add(MaxPool2D((4,1), padding='same'))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(8, activation='relu'))
    cnn_model.add(Dense(5, activation='softmax'))

    return cnn_model

# Recurrent neural network
def lstm_rnn():

    lstm_model = Sequential()
    lstm_model.add(Dense(32, activation='relu'))
    lstm_model.add(LSTM(16))
    lstm_model.add(Dense(32, activation='relu'))
    lstm_model.add(Dense(5, activation='softmax'))

    return lstm_model

def gru_rnn():

    gru_model = Sequential()
    gru_model.add(Dense(32, activation='relu'))
    gru_model.add(GRU(16))
    gru_model.add(Dense(32, activation='relu'))
    gru_model.add(Dense(5, activation='softmax'))

    return gru_model
