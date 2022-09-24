########################
##### AUTOENCODERS #####
########################

from unicodedata import name
import tensorflow as tf
from keras import Model, Input, regularizers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten

def autoencoder(encoder, decoder, input_shape):

    X = Input(shape=input_shape)

    encoded = encoder(X)
    decoded = decoder(encoded)

    autoencoder_model = Model(X, decoded, name='Autoencoder')

    return autoencoder_model

# Deep autoencoder 
def ann_encoder(latent_dim):

    encoder_model = Sequential(name='Deep_encoder')
    encoder_model.add(Dense(128, activation='tanh', name='Encoder_Layer_1'))
    encoder_model.add(Dense(64, activation='tanh', name='Encoder_Layer_2'))
    encoder_model.add(Dense(64, activation='tanh', name='Encoder_Layer_3'))
    encoder_model.add(Dense(latent_dim, activation='tanh', name='Encoder_output', activity_regularizer=regularizers.l1(10e-5)))

    return encoder_model

def ann_decoder(output_shape):

    decoder_model = Sequential(name='Deep_decoder')
    decoder_model.add(Dense(64, activation='tanh', name='Decoder_Layer_1'))
    decoder_model.add(Dense(64, activation='tanh', name='Decoder_Layer_2'))
    decoder_model.add(Dense(128, activation='tanh', name='Decoder_Layer_3'))
    decoder_model.add(Dense(output_shape, activation='sigmoid', name='Decoder_output'))

    return decoder_model

# Convolutional autoencoder

# Sequence-to-sequence autoencoder

# Variational autoencoder