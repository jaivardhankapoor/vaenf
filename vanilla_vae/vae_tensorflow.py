import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
# from torch.autograd import Variable
from tensorflow.examples.tutorials.mnist import input_data
config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )





mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
mb_size = 1
z_dim = 100
X_dim = mnist.train.images.shape[1]
y_dim = mnist.train.labels.shape[1]
h_dim = 128
c = 0
lr = 1e-3


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


# =============================== Q(z|X) ======================================

X = tf.placeholder(tf.float32, shape=[None, X_dim])
z = tf.placeholder(tf.float32, shape=[None, z_dim])

Q_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
Q_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

Q_W2_mu = tf.Variable(xavier_init([h_dim, z_dim]))
Q_b2_mu = tf.Variable(tf.zeros(shape=[z_dim]))

Q_W2_sigma = tf.Variable(xavier_init([h_dim, z_dim]))
Q_b2_sigma = tf.Variable(tf.zeros(shape=[z_dim]))


def Q(X):
    h = tf.nn.relu(tf.matmul(X, Q_W1) + Q_b1)
    z_mu = tf.matmul(h, Q_W2_mu) + Q_b2_mu
    z_logvar = tf.matmul(h, Q_W2_sigma) + Q_b2_sigma
    return z_mu, z_logvar

# z_mu, z_logvar = Q(X)
# z_sample = sample_z(z_mu, z*logvar)
def sample_z(mu, log_var):
    eps = tf.random_normal(shape=tf.shape(mu))
    return mu + tf.exp(log_var / 2) * eps


# =============================== P(X|z) ======================================

P_W1 = tf.Variable(xavier_init([z_dim, h_dim]))
P_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

P_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
P_b2 = tf.Variable(tf.zeros(shape=[X_dim]))


def P(z):
    h = tf.nn.relu(tf.matmul(z, P_W1) + P_b1)
    logits = tf.matmul(h, P_W2) + P_b2
    prob = tf.nn.sigmoid(logits)
    return prob, logits

# 
# =============================== TRAINING ====================================
# 
z_mu, z_logvar = Q(X)
# print(z_sample.get_shape())

z_sample = sample_z(z_mu, z_logvar)
# print(z_sample.get_shape()) #?,100

u =  tf.Variable(xavier_init([z_dim,1]),name="U")
# print(u.get_shape())

w =  tf.Variable(xavier_init([z_dim,1]),name="V")
# print(w.get_shape()) #100,1

b =  tf.Variable(xavier_init([1,1])) #scalar
# print(b.get_shape())

# z = z_sample + tf.multiply(u,tf.tanh(tf.matmul(tf.transpose(w),z_sample)+b))
uw = tf.matmul(tf.transpose(w),u)
# print(uw.get_shape())

muw = -1 + tf.nn.softplus(uw) # = -1 + T.log(1 + T.exp(uw))

# print(muw.get_shape())

u_hat = u + (muw - uw) * w / tf.reduce_sum(tf.matmul(tf.transpose(w),w))

# print(u_hat.get_shape()) #100,1

zwb = tf.matmul(z_sample,w) + b
# print(zwb.get_shape()) #?,1


f_z= z_sample + tf.multiply( tf.transpose(u_hat), tf.tanh(zwb))
# print(f_z.get_shape()) #?,100

psi = tf.matmul(w,tf.transpose(1-tf.multiply(tf.tanh(zwb), tf.tanh(zwb)))) # tanh(x)dx = 1 - tanh(x)**2
# print(psi.get_shape()) #100,?


psi_u = tf.matmul(tf.transpose(psi), u_hat)
# print(psi_u.get_shape())#?,1

logdet_jacobian = tf.log(tf.abs(1 + psi_u))

# print(logdet_jacobian.get_shape())#?,1

_, logits = P(f_z)  # add flows thing in P
# print(logits.get_shape()) #?,784

# Sampling from random z
X_samples, _ = P(z)
# print(X_samples.get_shape()) #?,784 

# E[log P(X|z_k)]
recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=X), 1)
# D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
kl_loss = 0.5 * tf.reduce_sum(tf.exp(z_logvar) + z_mu**2 - 1. - z_logvar, 1)
 # VAE loss
vae_loss = tf.reduce_mean(recon_loss + kl_loss - logdet_jacobian)

solver = tf.train.AdamOptimizer().minimize(vae_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0
writer = tf.summary.FileWriter("../", graph=tf.get_default_graph())
# distribution = []
for it in range(1000000):
    X_mb, _ = mnist.train.next_batch(mb_size)

    _, loss = sess.run([solver, vae_loss], feed_dict={X: X_mb})
    # distribution.append(sess.run())

    if it % 1000 == 0:
        print('Iter: {}'.format(it))
        print('Loss: {:.4}'. format(loss))
        print()

        samples = sess.run(X_samples, feed_dict={z: np.random.randn(16, z_dim)})

        fig = plot(samples)
        plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)
