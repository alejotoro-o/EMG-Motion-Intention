########################
##### AUTOENCODERS #####
########################

from unicodedata import name
import tensorflow as tf
from keras import Model, Input, regularizers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D, LSTM, GRU, Attention, RepeatVector

def autoencoder(encoder, decoder, input_shape):

    X = Input(shape=input_shape)

    encoded = encoder(X)
    decoded = decoder(encoded)

    autoencoder_model = Model(X, decoded, name='Autoencoder')

    return autoencoder_model

# Deep autoencoder 
def ann_encoder(latent_dim):

    encoder_model = Sequential(name='Deep_encoder')
    encoder_model.add(Dense(128, activation='relu', name='Encoder_l1'))
    encoder_model.add(Dense(latent_dim, activation='tanh', name='Encoder_output', activity_regularizer=regularizers.l1(10e-5)))

    return encoder_model

def ann_decoder(output_shape):

    decoder_model = Sequential(name='Deep_decoder')
    decoder_model.add(Dense(128, activation='relu', name='Decoder_l1'))
    decoder_model.add(Dense(output_shape, activation='sigmoid', name='Decoder_output'))

    return decoder_model

# Convolutional autoencoder
def cnn_encoder(latent_dim):

    encoder_model = Sequential(name='Convolutional_encoder')
    encoder_model.add(Conv2D(16, (2,2), activation='relu', padding='same', name='Encoder_l1'))
    encoder_model.add(MaxPool2D(latent_dim, padding='same', name='Encoder_output'))

    return encoder_model

def cnn_decoder(output_shape):

    decoder_model = Sequential(name='Convolutional_decoder')
    decoder_model.add(Conv2D(8, (2,2), padding='same', name='Decoder_l1'))
    decoder_model.add(UpSampling2D((2,2), name='Decoder_l2'))
    decoder_model.add(Conv2D(1, (2,2), padding='same', name='Decoder_output'))


    return decoder_model

# Sequence-to-sequence autoencoder
def lstm_encoder(latent_dim):

    encoder_model = Sequential(name='sts_encoder')
    encoder_model.add(Dense(64, activation='relu'))
    encoder_model.add(LSTM(32))
    encoder_model.add(Dense(latent_dim, activation='relu'))

    return encoder_model

def lstm_decoder(Tx, output_shape):

    decoder_model = Sequential(name='sts_decoder')
    decoder_model.add(RepeatVector(Tx))
    decoder_model.add(LSTM(output_shape, return_sequences=True))

    return decoder_model

# Variational autoencoder