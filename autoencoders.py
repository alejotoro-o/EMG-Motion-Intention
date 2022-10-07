########################
##### AUTOENCODERS #####
########################

from turtle import shape
from unicodedata import name
import keras
from keras import Model, Input, regularizers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D, LSTM, RepeatVector, Lambda
from keras.losses import binary_crossentropy

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
    encoder_model.add(Dense(64, activation='relu', name='Encoder_l2'))
    encoder_model.add(Dense(latent_dim, activation='tanh', name='Encoder_output', activity_regularizer=regularizers.l1(10e-5)))

    return encoder_model

def ann_decoder(output_shape):

    decoder_model = Sequential(name='Deep_decoder')
    decoder_model.add(Dense(64, activation='relu', name='Decoder_l1'))
    decoder_model.add(Dense(128, activation='relu', name='Decoder_l2'))
    decoder_model.add(Dense(output_shape, activation='sigmoid', name='Decoder_output'))

    return decoder_model

# Convolutional autoencoder
def cnn_encoder(latent_dim):

    encoder_model = Sequential(name='Convolutional_encoder')
    encoder_model.add(Conv2D(16, (4,4), activation='relu', padding='same'))
    encoder_model.add(MaxPool2D(latent_dim, padding='same'))
    encoder_model.add(Conv2D(8, (2,2), activation='relu', padding='same'))
    encoder_model.add(MaxPool2D(latent_dim, padding='same'))

    return encoder_model

def cnn_decoder(latent_dim):

    decoder_model = Sequential(name='Convolutional_decoder')
    decoder_model.add(Conv2D(8, (2,2), padding='same'))
    decoder_model.add(UpSampling2D(latent_dim))
    decoder_model.add(Conv2D(16, (4,4), padding='same'))
    decoder_model.add(UpSampling2D(latent_dim))
    decoder_model.add(Conv2D(1, (2,2), padding='same'))


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
def vae_autoencoder(input_shape, output_shape, intermediate_dim, latent_dim):

    inputs = Input(shape=input_shape)
    h = Dense(intermediate_dim, activation='relu')(inputs)
    z_mean = Dense(latent_dim)(h)
    z_log_sigma = Dense(latent_dim)(h)

    z = Lambda(vae_sampling)([z_mean, z_log_sigma, latent_dim])

    encoder_model = Model(inputs, [z_mean, z_log_sigma, z], name='vae_encoder')

    latent_inputs = Input(shape=(latent_dim,))
    x = Dense(intermediate_dim, activation='relu')(latent_inputs)
    out = Dense(output_shape, activation='sigmoid')(x)

    decoder_model = Model(latent_inputs, out, name='vae_decoder')

    outputs = decoder_model(encoder_model(inputs)[2])
    vae_autoencoder = keras.Model(inputs, outputs, name='vae_autoencoder')

    reconstruction_loss = binary_crossentropy(inputs, outputs)
    reconstruction_loss *= output_shape
    kl_loss = 1 + z_log_sigma - keras.backend.square(z_mean) - keras.backend.exp(z_log_sigma)
    kl_loss = keras.backend.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = keras.backend.mean(reconstruction_loss + kl_loss)

    vae_autoencoder.add_loss(vae_loss)

    return vae_autoencoder, encoder_model, decoder_model

def vae_sampling(args):

    z_mean, z_log_sigma, latent_dim = args
    epsilon = keras.backend.random_normal(shape=(keras.backend.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=0.1)
    return z_mean + keras.backend.exp(z_log_sigma) * epsilon

