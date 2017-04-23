'''This script demonstrates how to build a variational autoencoder with Keras.
Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Layer
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist

def sampling(args):
    z_mean, z_log_var = args

    # sample epsilon according to N(O,I)
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                              std=epsilon_std)

    # generate z0 according to N(z_mean, z_log_var)
    z0 = z_mean + K.exp(z_log_var / 2) * epsilon
    print('z0', z0)
    return z0

def logdet_loss(args):
    z0, w, u, b = args
    b2 = K.squeeze(b, 1)
    beta = K.sum(tf.multiply(w, z0), 1)  # <w|z0>
    linear_trans = beta + b2  # <w|z0> + b

    # change u2 so that the transformation z0->z1 is invertible
    alpha = K.sum(tf.multiply(w, u), 1)  # 
    diag1 = tf.diag(K.softplus(alpha) - 1 - alpha)
    u2 = u + K.dot(diag1, w) / K.sum(K.square(w)+1e-7)
    gamma = K.sum(tf.multiply(w,u2), 1)

    logdet = K.log(K.abs(1 + (1 - K.square(K.tanh(linear_trans)))*gamma) + 1e-6)

    return logdet

def transform_z0(args):
    z0, w, u, b = args
    b2 = K.squeeze(b, 1)
    beta = K.sum(tf.multiply(w, z0), 1)

    # change u2 so that the transformation z0->z1 is invertible
    alpha = K.sum(tf.multiply(w, u), 1)
    diag1 = tf.diag(K.softplus(alpha) - 1 - alpha)
    u2 = u + K.dot(diag1, w) / K.sum(K.square(w)+1e-7)
    diag2 = tf.diag(K.tanh(beta + b2))

    # generate z1
    z1 = z0 + K.dot(diag2,u2) 
    return z1

# and here is the loss
def vae_loss(x, x_decoded_mean):
    xent_loss = K.mean(objectives.categorical_crossentropy(x, x_decoded_mean), -1)
    ln_q0z0 = K.sum(log_normal2(z0, z_mean, z_log_var), -1)
    ln_pz1 = K.sum(log_stdnormal(z), -1)
    result = K.mean(l + ln_pz1 + xent_loss - ln_q0z0)
    return result    

# the encoder
h = encoder_block(x)  # a convnet taking proteins as input (matrices of size 400x22), I don't describe it since it isn't very important
z_log_var = Dense(latent_dim)(h)
z_mean = Dense(latent_dim)(h)
h_ = Dense(latent_dim)(h)
encoder = Model(x, [z_mean,z_log_var, h_])

# the latent variables (only one transformation to keep it simple)
latent_input = Input(shape=(latent_dim, 2), batch_shape=(batch_size, latent_dim, 2))
hl = Convolution1D(1, filter_length, activation="relu", border_mode="same")(latent_input)
hl = Reshape((latent_dim,))(hl)
mean_1 = Dense(latent_dim)(hl)
std_1 = Dense(latent_dim)(hl)
latent_model = Model(latent_input, [mean_1, std_1])

# the decoder
decoder_input = Input((latent_dim,), batch_shape=(batch_size, latent_dim))
decoder=decoder_block()  # a convnet that I don't describe
x_decoded_mean = decoder(decoder_input)
generator = Model(decoder_input, x_decoded_mean)

# the VAE
z_mean, z_log_var, other = encoder(vae_input)
eps = Lambda(sample_eps, name='sample_eps')([z_mean, z_log_var, other])
z0 = Lambda(sample_z0, name='sample_z0')([z_mean, z_log_var, eps])
l = Lambda(sample_l, name='sample_l')([eps, z_log_var])
mean, std = latent_model(merge([Reshape((latent_dim,1))(z0), Reshape((latent_dim,1))(other)], mode="concat", concat_axis=-1))
z = Lambda(transform_z0)([z0, mean, std])
l = Lambda(transform_l)([l, std])
x_decoded_mean = generator(z)
vae = Model(vae_input, x_decoded_mean)

