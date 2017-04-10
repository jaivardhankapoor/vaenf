import os.path
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST')

input_dim = 784
hidden_encoder_dim = 400
hidden_decoder_dim = 400
latent_dim = 20
lam = 0

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

def sampling(args):
    z_mean, z_log_var = args

    # sample epsilon according to N(O,I)
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                              std=epsilon_std)

    # generate z0 according to N(z_mean, z_log_var)
    z0 = z_mean + K.exp(z_log_var / 2) * epsilon
    print('z0', z0)
    return z0

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

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.001)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0., shape=shape)
  return tf.Variable(initial)

x = tf.placeholder("float", shape=[None, input_dim])
l2_loss = tf.constant(0.0)

W_encoder_input_hidden = weight_variable([input_dim,hidden_encoder_dim])
b_encoder_input_hidden = bias_variable([hidden_encoder_dim])
l2_loss += tf.nn.l2_loss(W_encoder_input_hidden)

# Hidden layer encoder
hidden_encoder = tf.nn.relu(tf.matmul(x, W_encoder_input_hidden) + b_encoder_input_hidden)

W_encoder_hidden_mu = weight_variable([hidden_encoder_dim,latent_dim])
b_encoder_hidden_mu = bias_variable([latent_dim])
l2_loss += tf.nn.l2_loss(W_encoder_hidden_mu)

# Mu encoder
mu_encoder = tf.matmul(hidden_encoder, W_encoder_hidden_mu) + b_encoder_hidden_mu

W_encoder_hidden_logvar = weight_variable([hidden_encoder_dim,latent_dim])
b_encoder_hidden_logvar = bias_variable([latent_dim])
l2_loss += tf.nn.l2_loss(W_encoder_hidden_logvar)

# Sigma encoder
logvar_encoder = tf.matmul(hidden_encoder, W_encoder_hidden_logvar) + b_encoder_hidden_logvar

# Sample epsilon
epsilon = tf.random_normal(tf.shape(logvar_encoder), name='epsilon')

# Sample latent variable
std_encoder = tf.exp(0.5 * logvar_encoder)
z = mu_encoder + tf.multiply(std_encoder, epsilon)

W_decoder_z_hidden = weight_variable([latent_dim,hidden_decoder_dim])
b_decoder_z_hidden = bias_variable([hidden_decoder_dim])
l2_loss += tf.nn.l2_loss(W_decoder_z_hidden)

# Hidden layer decoder
hidden_decoder = tf.nn.relu(tf.matmul(z, W_decoder_z_hidden) + b_decoder_z_hidden)

W_decoder_hidden_reconstruction = weight_variable([hidden_decoder_dim, input_dim])
b_decoder_hidden_reconstruction = bias_variable([input_dim])
l2_loss += tf.nn.l2_loss(W_decoder_hidden_reconstruction)

KLD = -0.5 * tf.reduce_sum(1 + logvar_encoder - tf.pow(mu_encoder, 2) - tf.exp(logvar_encoder), reduction_indices=1)

x_hat = tf.matmul(hidden_decoder, W_decoder_hidden_reconstruction) + b_decoder_hidden_reconstruction
BCE = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_hat, labels=x), reduction_indices=1)

loss = tf.reduce_mean(BCE + KLD)

regularized_loss = loss + lam * l2_loss

loss_summ = tf.summary.scalar("lowerbound", loss)
train_step = tf.train.AdamOptimizer(0.01).minimize(regularized_loss)

# add op for merging summary
summary_op = tf.summary.merge_all()

# add Saver ops
saver = tf.train.Saver()

n_steps = int(1e6)
batch_size = 100

with tf.Session() as sess:
  summary_writer = tf.summary.FileWriter('experiment',
                                          graph=sess.graph)
  if os.path.isfile("save/model.ckpt"):
    print("Restoring saved parameters")
    saver.restore(sess, "save/model.ckpt")
  else:
    print("Initializing parameters")
    sess.run(tf.global_variables_initializer())

  for step in range(1, n_steps):
    batch = mnist.train.next_batch(batch_size)
    feed_dict = {x: batch[0]}
    _, cur_loss, summary_str = sess.run([train_step, loss, summary_op], feed_dict=feed_dict)
    summary_writer.add_summary(summary_str, step)

    if step % 50 == 0:
      save_path = saver.save(sess, "save/model.ckpt")
      print("Step {0} | Loss: {1}".format(step, cur_loss))



# import tensorflow as tf
#
# class VariationalAutoencoder(object):
#
#   def __init__(self, n_input, n_hidden, optimizer = tf.train.AdamOptimizer()):
#       self.n_input = n_input
#       self.n_hidden = n_hidden
#
#       network_weights = self._initialize_weights()
#       self.weights = network_weights
#
#       # model
#       self.x = tf.placeholder(tf.float32, [None, self.n_input])
#       self.z_mean = tf.add(tf.matmul(self.x, self.weights['w1']), self.weights['b1'])
#       self.z_log_sigma_sq = tf.add(tf.matmul(self.x, self.weights['log_sigma_w1']), self.weights['log_sigma_b1'])
#
#       # sample from gaussian distribution
#       eps = tf.random_normal(tf.stack([tf.shape(self.x)[0], self.n_hidden]), 0, 1, dtype = tf.float32)
#       self.z = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))
#
#       self.reconstruction = tf.add(tf.matmul(self.z, self.weights['w2']), self.weights['b2'])
#
#       # cost
#       reconstr_loss = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
#       latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq
#                                          - tf.square(self.z_mean)
#                                          - tf.exp(self.z_log_sigma_sq), 1)
#       self.cost = tf.reduce_mean(reconstr_loss + latent_loss)
#       self.optimizer = optimizer.minimize(self.cost)
#
#       init = tf.global_variables_initializer()
#       self.sess = tf.Session()
#       self.sess.run(init)
#
#   def _initialize_weights(self):
#       all_weights = dict()
#       all_weights['w1'] = tf.get_variable("w1", shape=[self.n_input, self.n_hidden],
#           initializer=tf.contrib.layers.xavier_initializer())
#       all_weights['log_sigma_w1'] = tf.get_variable("log_sigma_w1", shape=[self.n_input, self.n_hidden],
#           initializer=tf.contrib.layers.xavier_initializer())
#       all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
#       all_weights['log_sigma_b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
#       all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
#       all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
#       return all_weights
#
#   def partial_fit(self, X):
#       cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X})
#       return cost
#
#   def calc_total_cost(self, X):
#       return self.sess.run(self.cost, feed_dict = {self.x: X})
#
#   def transform(self, X):
#       return self.sess.run(self.z_mean, feed_dict={self.x: X})
#
#   def generate(self, hidden = None):
#       if hidden is None:
#           hidden = self.sess.run(tf.random_normal([1, self.n_hidden]))
#       return self.sess.run(self.reconstruction, feed_dict={self.z: hidden})
#
#   def reconstruct(self, X):
#       return self.sess.run(self.reconstruction, feed_dict={self.x: X})
#
#   def getWeights(self):
#       return self.sess.run(self.weights['w1'])
#
#   def getBiases(self):
#       return self.sess.run(self.weights['b1'])
