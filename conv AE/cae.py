
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

def deconv2d(x, filter_size, n_in, n_out, strides, output_shape, use_lrelu=True, name=None):
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
  if use_lrelu:
    x = tf.nn.leaky_relu(x)
  else:
    x = tf.nn.relu(x)
  return x

class CAE(object):
  def __init__(self, args, sess, name='conv_vae'):
    self.input_dim = args.input_dim
    self.latent_dim = args.latent_dim
    self.hidden_dim = args.hidden_dim
    self.batch_size = args.batch_size
    self.sess = sess
    self.max_grad_norm = args.max_grad_norm
    self.learning_rate = args.learning_rate

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

  def build_graph(self):
    with tf.variable_scope('encoder'):
      self.hidden = self.build_encoder(self.x_images)

    with tf.variable_scope('decoder'):
      self.x_reconstruct = self.build_decoder(self.hidden)

  def build_encoder(self, x_images):
    #x, filter_size, n_in, n_out, strides, name
    x = tf.reshape(self.x_images, [-1, 28, 28, 1])
    x = conv2d(x, 3, 1, 10, 2, 'en_layer1')
    x = conv2d(x, 3, 10, 10, 2, 'en_layer2')
    x = conv2d(x, 3, 10, 10, 2, 'en_layer3')
    b, h, w, c = x.get_shape().as_list()
    x = tf.reshape(x, [-1, h * w * c])
    x = tf.layers.dense(x, self.latent_dim, activation=tf.nn.relu, name='en_layer4')
    print x.get_shape().as_list()
    return x

  def build_decoder(self, hidden):
    #x, filter_size, n_in, n_out, strides, name
    x = hidden
    x = tf.layers.dense(x, 160, activation=tf.nn.relu, name='de_layer0')
    x = tf.reshape(x, [-1, 4, 4, 10])
    x = deconv2d(x, 3, 10, 10, 2, [self.batch_size, 7, 7, 10], name='de_layer1')
    x = deconv2d(x, 3, 10, 10, 2, [self.batch_size, 14, 14, 10], name='de_layer2')
    x = deconv2d(x, 3, 1, 10, 2, [self.batch_size, 28, 28, 1], False, name='de_layer3')
    print x.get_shape().as_list()
    return x

  def build_loss(self):
    x_images = tf.reshape(self.x_images, [-1, 28, 28, 1])
    self.loss_op = tf.reduce_sum(tf.square(self.x_reconstruct - x_images))

  def build_train(self):
    grads_and_vars = self.optimizer.compute_gradients(self.loss_op)
    grads_and_vars = [(tf.clip_by_norm(g, self.max_grad_norm), v) for g, v in grads_and_vars]
    self.train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

  def train(self, x_images, get_summary=False):
    feed_dict = {self.x_images: x_images}
    _, loss = self.sess.run([self.train_op, self.loss_op], feed_dict=feed_dict)
    return loss

  def reconstruct(self, x_images):
    feed_dict = {self.x_images: x_images}
    return self.sess.run(self.x_reconstruct, feed_dict)
