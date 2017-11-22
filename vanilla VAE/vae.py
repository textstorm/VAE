
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

class VAE(object):
  def __init__(self, args, sess, name="vae"):
    self.input_dim = args.input_dim
    self.hidden_dim = args.hidden_dim
    self.latent_dim = args.latent_dim
    self.sess = sess
    self.max_grad_norm = args.max_grad_norm
    self.learning_rate = args.learning_rate

    self._build_placeholder()
    
    self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
    self.global_step = tf.Variable(0, trainable=False)
    self._build_graph()
    self._build_loss()
    self._build_train()
    
    self.summary = tf.summary.merge_all()
    init_op = tf.global_variables_initializer()
    self.sess.run(init_op)

  def _build_placeholder(self):
    with tf.name_scope("data"):
      self.x_images = tf.placeholder(tf.float32, [None, self.input_dim])
      self.z = tf.placeholder(tf.float32, [None, self.latent_dim])

  def _build_graph(self):
    with tf.name_scope("model"):
      with tf.variable_scope("encoder"):
        self.mu, self.log_sigma_sq = self.build_encoder(self.x_images)
      with tf.variable_scope("decoder"):
        sample_z = self.sample_z(self.mu, self.log_sigma_sq)
        self.logits, self.reconstruct = self.build_decoder(sample_z)

  def _build_loss(self):
    with tf.name_scope("loss"):
      rec_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
          logits=self.logits, labels=self.x_images), 1)
      kl_loss = -0.5 * tf.reduce_sum(
          1. + self.log_sigma_sq - self.mu ** 2 - tf.exp(self.log_sigma_sq), 1)
      self.loss_op = tf.reduce_mean(rec_loss + kl_loss)

  def _build_train(self):
    with tf.name_scope('train'):
      grads_and_vars = self.optimizer.compute_gradients(self.loss_op)
      grads_and_vars = [(tf.clip_by_norm(g, self.max_grad_norm), v) for g, v in grads_and_vars]
      self.train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

  def train(self, x_images, get_summary=False):
    feed_dict = {self.x_images: x_images}
    loss, _, = self.sess.run([self.loss_op, self.train_op], feed_dict=feed_dict)

    return loss

  def build_encoder(self, x_images):
    with tf.variable_scope("encoder"):
      x = x_images
      x = tf.layers.dense(x, self.hidden_dim, name='en_layer1')
      x = tf.nn.relu(x)
      x = tf.layers.dense(x, self.hidden_dim, name='en_layer2')
      x = tf.nn.relu(x)
      x = tf.layers.dense(x, self.hidden_dim, name='en_layer3')
      x = tf.nn.relu(x)
      z_mu = tf.layers.dense(x, self.latent_dim, name='layer_mu')
      z_log_sigma_sq = tf.layers.dense(x, self.latent_dim, name='layer_log_sigma_sq')
      return z_mu, z_log_sigma_sq

  def build_decoder(self, z):
    with tf.variable_scope("decoder"):
      x = z
      x = tf.layers.dense(x, self.hidden_dim, name='de_layer1')
      x = tf.nn.relu(x)
      x = tf.layers.dense(x, self.hidden_dim, name='de_layer2')
      x = tf.nn.relu(x)
      x = tf.layers.dense(x, self.input_dim, name='de_layer3')
      x_recons = tf.nn.sigmoid(x)
      return x, x_recons

  def sample_z(self, mu, log_sigma_sq):
    eps = tf.random_normal(shape=tf.shape(mu))
    return mu + tf.exp(log_sigma_sq / 2) * eps

  def generate(self, z):
    feed_dict= {self.z: z}
    with tf.variable_scope("decoder", reuse=True):
      logits, x_recons = self.build_decoder(self.z)
    return self.sess.run(x_recons, feed_dict=feed_dict)

  def reconstruct(self, x_images):
    feed_dict = {self.x_images: x_images}
    return self.sess.run(self.reconstruct, feed_dict)
