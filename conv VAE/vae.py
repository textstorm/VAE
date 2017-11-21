
import tensorflow as tf

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

def conv2d(x, k, num_features, strides=1, linear=False, name=None):
  #filter=[h, w, in_c, out_c]
  in_channel  = x.get_shape().as_list()[3]
  weight = weight_variable([k, k, in_channel, num_features], name=name+"_weight")
  bias = bias_variable([num_features], name=name+'_bias')
  x = tf.nn.conv2d(input=x, 
                   filter=weight, 
                   strides=[1, strides, strides, 1], 
                   padding='SAME')
  x = tf.nn.bias_add(x, bias)
  if linear: 
    return x
  else:
    return tf.nn.relu(x)

def deconv2d(x, k, num_features, strides=1, linear=False, name=None):
  #filter=[h, w, out_c, in_c]
  #output_shape=[batch, h, w, c]
  in_channel = x.get_shape().as_list()[3]
  weight = weight_variable([k, k, num_features, in_channel], name=name+'_weight')
  bias = bias_variable([num_features], name=name+'_bias')
  batch_size, height, width, channel = x.get_shape().as_list()
  output_shape = tf.stack([batch_size, height * strides, width * strides, num_features])
  x = tf.nn.conv2d_transpose(value=x, 
                             filter=weight, 
                             output_shape=output_shape, 
                             strides=[1, strides, strides, 1], 
                             padding='SAME')
  x = tf.nn.bias_add(x, bias)
  if linear:
    return x
  else:
    return tf.nn.relu(x)

class ConvVAE(object):
  def __init__(self, args, sess, name='conv_vae'):
    self.input_dim = args.input_dim
    self.hidden_dim = args.hidden_dim
    self.latent_dim = args.latent_dim
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
    self.z = tf.placeholder(tf.float32, [self.batch_size, self.latent_dim], name="z")
    self.keep_prob = tf.placeholder(tf.float32, [], name='keep_prob')

  def build_graph(self):
    with tf.name_scope('model'):
      with tf.variable_scope('encoder'):
        self.mu, self.log_sigma_sq = self.build_encoder(self.x_images)

      with tf.variable_scope('decoder'):
        sample_z = self.sample_z(self.mu, self.log_sigma_sq)
        self.logits, self.reconstruct = self.build_decoder(sample_z)

  def build_encoder(self, x_images):
    x = tf.reshape(x_images, [-1, 28, 28, 1])
    x = conv2d(x, 3, 8, 2, name='en_layer1')
    x = conv2d(x, 3, 8, 1, name='en_layer2')
    x = conv2d(x, 3, 8, 2, name='en_layer3')
    x = conv2d(x, 1, 4, 1, name='en_layer4')
    b, h, w, c = x.get_shape().as_list()
    x = tf.reshape(x, [-1, h * w * c])
    x = tf.layers.dense(x, 128, name='en_layer5')
    x = tf.nn.relu(x)
    z_mu = tf.layers.dense(x, self.latent_dim, name='layer_mu')
    z_log_sigma_sq = tf.layers.dense(x, self.latent_dim, name='layer_log_sigma_sq')
    return z_mu, z_log_sigma_sq

  def sample_z(self, mu, log_sigma_sq):
    eps = tf.random_normal(shape=tf.shape(mu))
    return mu + tf.exp(log_sigma_sq / 2) * eps

  def build_decoder(self, sample_z):
    x = sample_z
    x = tf.layers.dense(x, 128, name='de_layer1')
    x = tf.nn.relu(x)
    x = tf.layers.dense(x, 196, name='de_layer2')
    x = tf.nn.relu(x)
    x = tf.reshape(x, [-1, 7, 7, 4])
    x = deconv2d(x, 1, 8, 1, name='de_layer3')
    x = deconv2d(x, 3, 8, 2, name='de_layer4')
    x = deconv2d(x, 3, 8, 1, name='de_layer5')
    x = deconv2d(x, 3, 1, 2, name='de_layer6')
    b, h, w, c = x.get_shape().as_list()
    x = tf.reshape(x, [-1, h * w * c])
    x_recons = tf.nn.sigmoid(x)
    return x, x_recons

  def build_loss(self):
    with tf.name_scope("loss"):
      rec_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
          logits=self.logits, labels=self.x_images), 1)
      kl_loss = -0.5 * tf.reduce_sum(
          1. + self.log_sigma_sq - self.mu ** 2 - tf.exp(self.log_sigma_sq), 1)
      self.loss_op = tf.reduce_mean(rec_loss + kl_loss)

  def build_train(self):
    with tf.name_scope('train'):
      grads_and_vars = self.optimizer.compute_gradients(self.loss_op)
      grads_and_vars = [(tf.clip_by_norm(g, self.max_grad_norm), v) for g, v in grads_and_vars]
      self.train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

  def train(self, x_images, get_summary=False):
    feed_dict = {self.x_images: x_images}
    loss, _, = self.sess.run([self.loss_op, self.train_op], feed_dict=feed_dict)
    return loss

  def generate(self, z):
    feed_dict= {self.z: z}
    with tf.variable_scope("decoder", reuse=True):
      logits, x_recons = self.build_decoder(self.z)
    return self.sess.run(x_recons, feed_dict=feed_dict)

  def reconstruct(self, x_images):
    feed_dict = {self.x_images: x_images}
    return self.sess.run(self.reconstruct, feed_dict)
