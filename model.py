
import tensorflow as tf

class Base(object):
  def __init__(self, args, sess, name=None):
    self.input_dim = args.input_dim
    self.latent_dim = args.latent_dim
    self.hidden_dim = args.hidden_dim
    self.sess = sess
    self.max_grad_norm = args.max_grad_norm
    self.learning_rate = tf.Variable(float(args.learning_rate), trainable=False, name="learning_rate")
    self.learning_rate_decay_op = self.learning_rate.assign(
        tf.multiply(self.learning_rate, args.lr_decay))
    self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
    self.global_step = tf.Variable(0, trainable=False)

    with tf.name_scope("data"):
      self.x_images = tf.placeholder(tf.float32, [None, self.input_dim])
      self.z = tf.placeholder(tf.float32, [None, self.latent_dim])

  def sample_z(self, mu, log_sigma_sq):
    eps = tf.random_normal(shape=tf.shape(mu))
    return mu + tf.exp(log_sigma_sq / 2) * eps

  def build_encoder(self, x_images):
    x = tf.layers.dense(x_images, self.hidden_dim, activation=tf.nn.relu, name='en_layer1')
    x = tf.layers.dense(x, self.hidden_dim, activation=tf.nn.relu, name='en_layer2')
    x = tf.layers.dense(x, self.hidden_dim, activation=tf.nn.relu, name='en_layer3')
    z_mu = tf.layers.dense(x, self.latent_dim, name='layer_mu')
    z_log_sigma_sq = tf.layers.dense(x, self.latent_dim, name='layer_log_sigma_sq')
    return z_mu, z_log_sigma_sq

  def build_decoder(self, z):
    x = tf.layers.dense(z, self.hidden_dim, activation=tf.nn.relu, name='de_layer1')
    x = tf.layers.dense(x, self.hidden_dim, activation=tf.nn.relu, name='de_layer2')
    x = tf.layers.dense(x, self.input_dim, activation=None, name='de_layer3')
    x_recons = tf.nn.sigmoid(x)
    return x, x_recons

  def train(self, x_images):
    feed_dict = {self.x_images: x_images}
    return self.sess.run([self.train_op, 
                          self.loss_op, 
                          self.loss_rec, 
                          self.loss_kl,
                          self.global_step,
                          self.summary], feed_dict=feed_dict)

  def generate(self, z):
    feed_dict= {self.z: z}
    with tf.variable_scope("decoder", reuse=True):
      logits, x_recons = self.build_decoder(self.z)
    return self.sess.run(x_recons, feed_dict=feed_dict)

  def reconstruct(self, x_images):
    feed_dict = {self.x_images: x_images}
    return self.sess.run(self.reconstruct, feed_dict)

  def weight_variable(self, shape, name, initializer=None):
    initializer = tf.truncated_normal_initializer(stddev=0.1)
    if initializer:
      initializer = initializer
    return tf.get_variable(shape=shape, initializer=initializer, name=name)

  def bias_variable(self, shape, name, initializer=None):
    initializer = tf.constant_initializer(0.)
    if initializer:
      initializer = initializer
    return tf.get_variable(shape=shape, initializer=initializer, name=name)

class VAE(Base):
  def __init__(self, args, sess, name="vae"):
    super(VAE, self).__init__(args=args, sess=sess, name=name)
    with tf.name_scope("vae"):
      with tf.variable_scope("encoder"):
        self.mu, self.log_sigma_sq = self.build_encoder(self.x_images)
      with tf.variable_scope("decoder"):
        sample_z = self.sample_z(self.mu, self.log_sigma_sq)
        tf.summary.histogram('sample_gaussian', sample_z)
        self.logits, self.reconstruct = self.build_decoder(sample_z)

    with tf.name_scope("loss"):
      rec_loss = tf.reduce_sum(
          tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.x_images), 1)
      kl_loss = -0.5 * tf.reduce_sum(
          1. + self.log_sigma_sq - self.mu ** 2 - tf.exp(self.log_sigma_sq), 1)
      self.loss_op = tf.reduce_mean(rec_loss + kl_loss)
      self.loss_rec, self.loss_kl = tf.reduce_mean(rec_loss), tf.reduce_mean(kl_loss)
      tf.summary.scalar("reconstruct_loss", self.loss_rec)
      tf.summary.scalar("kl_loss", self.loss_kl)
      tf.summary.scalar("total_loss", self.loss_op)

    with tf.name_scope('train'):
      grads_and_vars = self.optimizer.compute_gradients(self.loss_op)
      grads_and_vars = [(tf.clip_by_norm(g, self.max_grad_norm), v) for g, v in grads_and_vars]
      self.train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

    init_op = tf.global_variables_initializer()
    self.sess.run(init_op)
    self.summary = tf.summary.merge_all()
    self.saver = tf.train.Saver(tf.global_variables())

class DCVAE(Base):
  def __init__(self, args, sess, name="dcvae"):
    super(DCVAE, self).__init__(args=args, sess=sess, name=name)
    with tf.name_scope('dcvae'):
      with tf.variable_scope('encoder'):
        self.mu, self.log_sigma_sq = self.build_encoder(self.x_images)
      with tf.variable_scope('decoder'):
        sample_z = self.sample_z(self.mu, self.log_sigma_sq)
        tf.summary.histogram('sample_gaussian', sample_z)
        self.logits, self.x_reconstruct = self.build_decoder(sample_z)

    with tf.name_scope("loss"):
      rec_loss = 0.5 * tf.reduce_sum(
          tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.x_images), 1)
      kl_loss = -0.5 * tf.reduce_sum(
          1. + self.log_sigma_sq - self.mu ** 2 - tf.exp(self.log_sigma_sq), 1)
      self.loss_op = tf.reduce_mean(rec_loss + kl_loss)
      self.loss_rec, self.loss_kl = tf.reduce_mean(rec_loss), tf.reduce_mean(kl_loss)
      tf.summary.scalar("reconstruct_loss", self.loss_rec)
      tf.summary.scalar("kl_loss", self.loss_kl)
      tf.summary.scalar("total_loss", self.loss_op)

    with tf.name_scope('train'):
      grads_and_vars = self.optimizer.compute_gradients(self.loss_op)
      grads_and_vars = [(tf.clip_by_norm(g, self.max_grad_norm), v) for g, v in grads_and_vars]
      self.train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

    init_op = tf.global_variables_initializer()
    self.sess.run(init_op)
    self.summary = tf.summary.merge_all()
    self.saver = tf.train.Saver(tf.global_variables())

  def build_encoder(self, x_images):
    x = tf.reshape(x_images, [-1, 28, 28, 1])
    x = conv2d(x, 3, 1, 10, 2, 'en_layer1')
    x = conv2d(x, 3, 10, 10, 2, 'en_layer2')
    x = conv2d(x, 3, 10, 10, 2, 'en_layer3')
    b, h, w, c = x.get_shape().as_list()
    x = tf.reshape(x, [-1, h * w * c])
    z_mu = tf.layers.dense(x, self.latent_dim, name='layer_mu')
    z_log_sigma_sq = tf.layers.dense(x, self.latent_dim, name='layer_log_sigma_sq')
    return z_mu, z_log_sigma_sq    

  def build_decoder(self, z):
    x = tf.layers.dense(z, 160, activation=tf.nn.relu, name='de_layer1')
    x = tf.reshape(x, [-1, 4, 4, 10])
    x = deconv2d(x, 3, 10, 10, 2, [-1, 7, 7, 10], name='de_layer2')
    x = deconv2d(x, 3, 10, 10, 2, [-1, 14, 14, 10], name='de_layer3')
    x = deconv2d(x, 3, 1, 10, 2, [-1, 28, 28, 1], False, name='de_layer4')
    x = tf.reshape(x, [-1, self.input_dim])
    x_reconstruct = tf.nn.sigmoid(x)
    return x, x_reconstruct