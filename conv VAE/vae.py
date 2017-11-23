
import tensorflow as tf
import numpy as np

def weight_variable(shape, name, initializer=None):
  initializer = tf.truncated_normal_initializer(stddev=0.1)
  if initializer:
    initializer = initializer
  return tf.get_variable(shape=shape, initializer=initializer, name=name)

def bias_variable(shape, name, initializer=None):
  initializer = tf.constant_initializer(0.)
  if initializer:
    initializer = initializer
  return tf.get_variable(shape=shape, initializer=initializer, name=name)

def conv2d(x, filter_size, n_in, n_out, strides, name=None):
  w = weight_variable(shape=[filter_size, filter_size, n_in, n_out], name=name+'_w')
  b = bias_variable(shape=[n_out], name=name+'_b')
  x = tf.nn.conv2d(input=x, 
                   filter=w, 
                   strides=[1, strides, strides, 1],
                   padding='SAME',
                   name=name)
  x = tf.nn.bias_add(x, b)
  x = tf.nn.leaky_relu(x)
  return x

def deconv2d(x, filter_size, n_in, n_out, strides, output_shape, use_activation=True, name=None):
  w = weight_variable(shape=[filter_size, filter_size, n_in, n_out], name=name+'_w')
  b = bias_variable(shape=[n_in], name=name+'_b')
  output_shape = tf.stack(output_shape)
  x = tf.nn.conv2d_transpose(value=x,
                             filter=w,
                             output_shape=output_shape,
                             strides=[1, strides, strides, 1],
                             padding='SAME',
                             name=name)
  x = tf.nn.bias_add(x, b)

  if use_activation:
    x = tf.nn.leaky_relu(x)
  return x

class ConvVAE(object):
  def __init__(self, args, sess, name="vae"):
    self.input_dim = args.input_dim
    self.hidden_dim = args.hidden_dim
    self.latent_dim = args.latent_dim
    self.sess = sess
    self.max_grad_norm = args.max_grad_norm
    self.learning_rate = args.learning_rate
    self.batch_size = args.batch_size

    self.add_placeholder()
    
    self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
    self.global_step = tf.Variable(0, trainable=False)
    self.build_graph()
    self.build_loss()
    self.build_train()
    
    init_op = tf.global_variables_initializer()
    self.sess.run(init_op)

  def add_placeholder(self):
    self.x_images = tf.placeholder(tf.float32,
                                   [self.batch_size, self.input_dim],
                                   name='x_images')
    self.z = tf.placeholder(tf.float32, [self.batch_size, self.latent_dim], name="z")

  def build_graph(self):
    with tf.name_scope('model'):
      with tf.variable_scope('encoder'):
        self.mu, self.log_sigma_sq = self.build_encoder(self.x_images)

      with tf.variable_scope('decoder'):
        sample_z = self.sample_z(self.mu, self.log_sigma_sq)
        self.logits, self.x_reconstruct = self.build_decoder(sample_z)

  def build_encoder(self, x_images):
    #x, filter_size, n_in, n_out, strides, name
    x = tf.reshape(x_images, [-1, 28, 28, 1])
    x = conv2d(x, 3, 1, 10, 2, 'en_layer1')
    x = conv2d(x, 3, 10, 10, 2, 'en_layer2')
    x = conv2d(x, 3, 10, 10, 2, 'en_layer3')
    b, h, w, c = x.get_shape().as_list()
    x = tf.reshape(x, [-1, h * w * c])
    z_mu = tf.layers.dense(x, self.latent_dim, name='layer_mu')
    z_log_sigma_sq = tf.layers.dense(x, self.latent_dim, name='layer_log_sigma_sq')
    return z_mu, z_log_sigma_sq    

  def sample_z(self, mu, log_sigma_sq):
    eps = tf.random_normal(shape=tf.shape(mu))
    return mu + tf.exp(log_sigma_sq / 2) * eps

  def build_decoder(self, z):
    #x, filter_size, n_in, n_out, strides, name
    x = z
    x = tf.layers.dense(x, 160, activation=tf.nn.relu, name='de_layer1')
    x = tf.reshape(x, [-1, 4, 4, 10])
    x = deconv2d(x, 3, 10, 10, 2, [self.batch_size, 7, 7, 10], name='de_layer2')
    x = deconv2d(x, 3, 10, 10, 2, [self.batch_size, 14, 14, 10], name='de_layer3')
    x = deconv2d(x, 3, 1, 10, 2, [self.batch_size, 28, 28, 1], False, name='de_layer4')
    x = tf.reshape(x, [-1, self.input_dim])
    x_reconstruct = tf.nn.sigmoid(x)
    return x, x_reconstruct

  def build_loss(self):
    with tf.name_scope("loss"):
      rec_loss = 0.5 * tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
          logits=self.logits, labels=self.x_images), 1)
      kl_loss = -0.5 * tf.reduce_sum(
          1. + self.log_sigma_sq - self.mu ** 2 - tf.exp(self.log_sigma_sq), 1)
      self.loss_op = tf.reduce_mean(rec_loss + kl_loss)
      self.loss_rec, self.loss_kl = tf.reduce_mean(rec_loss), tf.reduce_mean(kl_loss)

  def build_train(self):
    with tf.name_scope('train'):
      grads_and_vars = self.optimizer.compute_gradients(self.loss_op)
      grads_and_vars = [(tf.clip_by_norm(g, self.max_grad_norm), v) for g, v in grads_and_vars]
      self.train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

  def train(self, x_images, get_summary=False):
    feed_dict = {self.x_images: x_images}
    _, loss, loss_rec, loss_kl = self.sess.run([self.train_op, self.loss_op, self.loss_rec, self.loss_kl], feed_dict=feed_dict)
    return loss, loss_rec, loss_kl

  def generate(self, z):
    feed_dict= {self.z: z}
    with tf.variable_scope("decoder", reuse=True):
      logits, x_recons = self.build_decoder(self.z)
    return self.sess.run(x_recons, feed_dict=feed_dict)

  def reconstruct(self, x_images):
    feed_dict = {self.x_images: x_images}
    return self.sess.run(self.x_reconstruct, feed_dict)
