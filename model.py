
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

    self.batch_size = tf.shape(self.x_images)[0]

  def sample_z(self, mu, log_sigma_sq):
    eps = tf.random_normal(shape=tf.shape(mu))
    return mu + tf.exp(log_sigma_sq / 2) * eps

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
    x = self.conv2d(x, 3, 1, 10, 2, 'en_layer1')
    x = self.conv2d(x, 3, 10, 10, 2, 'en_layer2')
    x = self.conv2d(x, 3, 10, 10, 2, 'en_layer3')
    b, h, w, c = x.get_shape().as_list()
    x = tf.reshape(x, [-1, h * w * c])
    z_mu = tf.layers.dense(x, self.latent_dim, name='layer_mu')
    z_log_sigma_sq = tf.layers.dense(x, self.latent_dim, name='layer_log_sigma_sq')
    return z_mu, z_log_sigma_sq    

  def build_decoder(self, z):
    x = tf.layers.dense(z, 160, activation=tf.nn.relu, name='de_layer1')
    x = tf.reshape(x, [-1, 4, 4, 10])
    x = self.deconv2d(x, 3, 10, 10, 2, [self.batch_size, 7, 7, 10], name='de_layer2')
    x = self.deconv2d(x, 3, 10, 10, 2, [self.batch_size, 14, 14, 10], name='de_layer3')
    x = self.deconv2d(x, 3, 1, 10, 2, [self.batch_size, 28, 28, 1], False, name='de_layer4')
    x = tf.reshape(x, [-1, self.input_dim])
    x_reconstruct = tf.nn.sigmoid(x)
    return x, x_reconstruct

  def conv2d(self, x, filter_size, n_in, n_out, strides, name=None):
    w = self.weight_variable(shape=[filter_size, filter_size, n_in, n_out], name=name+'_w')
    b = self.bias_variable(shape=[n_out], name=name+'_b')
    x = tf.nn.conv2d(input=x, 
                     filter=w, 
                     strides=[1, strides, strides, 1],
                     padding='SAME',
                     name=name)
    x = tf.nn.bias_add(x, b)
    x = tf.nn.relu(x)
    return x

  def deconv2d(self, x, filter_size, n_in, n_out, strides, output_shape, use_activation=True, name=None):
    w = self.weight_variable(shape=[filter_size, filter_size, n_in, n_out], name=name+'_w')
    b = self.bias_variable(shape=[n_in], name=name+'_b')
    output_shape = tf.stack(output_shape)
    x = tf.nn.conv2d_transpose(value=x,
                               filter=w,
                               output_shape=output_shape,
                               strides=[1, strides, strides, 1],
                               padding='SAME',
                               name=name)
    x = tf.nn.bias_add(x, b)
    if use_activation:
      x = tf.nn.relu(x)
    return x

  def generate(self, z, batch_size):
    feed_dict= {self.z: z, self.batch_size: batch_size}
    with tf.variable_scope("decoder", reuse=True):
      logits, x_recons = self.build_decoder(self.z)
    return self.sess.run(x_recons, feed_dict=feed_dict)

class LVAE(Base):
  def __init__(self, args, sess, name="lvae"):
    super(LVAE, self).__init__(args=args, sess=sess, name=name)
    self.hidden_sizes = args.hidden_sizes
    self.latent_sizes = args.latent_sizes
    self.latent_length = len(self.latent_sizes)
    # self.encoder_mus, self.encoder_logvars = [0] * self.latent_length, [0] * self.latent_length
    # self.decoder_mus, self.decoder_logvars = [0] * self.latent_length, [0] * self.latent_length
    # self.prior_mus, self.prior_logvars = [0] * self.latent_length, [0] * self.latent_length

    with tf.name_scope('lvae'):
      with tf.variable_scope('encoder'):
        encoder_mus, encoder_logvars = self.build_encoder(self.x_images)
      with tf.variable_scope('decoder'):
        logits, x_reconstruct = self.build_decoder(encoder_mus, encoder_logvars)
        # tf.summary.histogram('sample_gaussian', sample_z)

    with tf.name_scope("loss"):
      rec_loss = 0.5 * tf.reduce_sum(
          tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.x_images), 1)
      # kl_loss = -0.5 * tf.reduce_sum(
      #     1. + self.log_sigma_sq - self.mu ** 2 - tf.exp(self.log_sigma_sq), 1)
      kl_losses, kl_loss = self.kl_loss_compute()
      self.loss_op = tf.reduce_mean(rec_loss + kl_loss)
      self.loss_rec, self.loss_kl = tf.reduce_mean(rec_loss), tf.reduce_mean(kl_loss)
      tf.summary.scalar("reconstruct_loss", self.loss_rec)
      tf.summary.scalar("kl_loss", self.loss_kl)
      tf.summary.scalar("total_loss", self.loss_op)

  def build_encoder(self, x_images):
    h = x_images
    encoder_mus, encoder_logvars = [0] * self.latent_length, [0] * self.latent_length
    for idx, hsize in enumerate(self.hidden_sizes):
      h = tf.layers.dense(h, hsize)
      h = tf.layers.batch_normalization(h)
      h = tf.nn.relu(h)
      self.encoder_mus[idx], self.encoder_logvars[idx] = self.build_latent(h)
    return encoder_mus, encoder_logvars

  def build_decoder(self, encoder_mus, encoder_logvars):
    self.decoder_mus, self.decoder_logvars = [0] * self.latent_length, [0] * self.latent_length
    prior_mus, prior_logvars = [0] * self.latent_length, [0] * self.latent_length
    for idx in reversed(range(self.latent_length)):
      if idx == self.latent_length - 1:
        mu, logvars = encoder_mus[idx], encoder_logvars[idx]
        decoder_mus[idx], decoder_logvars[idx] = mu, logvars
        z = self.sample_z(mu, logvars)
        prior_mus[idx], prior_logvars[idx] = tf.zeros((mu.get_shape())), tf.zeros((logvars.get_shape()))
      else:
        prior_mus[idx] = tf.layers.dense(z, self.latent_sizes[idx], activation=tf.nn.softplus)
        prior_logvars[idx] = tf.layers.dense(z, self.latent_sizes[idx], activation=tf.nn.softplus)
        mu_t, sigma_t, mu_d, sigma_d = encoder_mus[idx], encoder_logvars[idx], decoder_mus[idx], decoder_logvars[idx]
        decoder_mus[idx], decoder_logvars[idx] = self.precision_weighted(mu_t, sigma_t, mu_d, sigma_d)
        z = self.sample_z(decoder_mus[idx], decoder_logvars[idx])

    x = tf.layers.dense(z, self.input_dim, activation=None)
    x_recons = tf.nn.sigmoid(x)
    return x, x_recons 

  def build_latent(self, hidden):
    z_mu = tf.layers.dense(x, self.latent_dim, name='layer_mu')
    z_log_sigma_sq = tf.layers.dense(x, self.latent_dim, name='layer_log_sigma_sq')
    return z_mu, z_log_sigma_sq

  def precision_weighted(self, mu_t, sigma_t, mu_d, sigma_d):
    sigma_t_reci = sigma_t ** (-1)
    sigma_d_reci = sigma_d ** (-1)
    mu = (mu_t * sigma_t_reci + mu_d * sigma_d_reci) / (sigma_t_reci + sigma_d_reci)
    sigma = (sigma_d_reci + sigma_t_reci) ** (-1)
    return mu, sigma

  def kl_loss_compute(self, ):
    kl_loss = [0] * self.latent_length
    for idx in range(self.latent_length):

      kl_loss[idx] = -0.5 * tf.reduce_sum(
          1. + self.log_sigma_sq - self.mu ** 2 - tf.exp(self.log_sigma_sq), 1)

    return kl_loss, kl_loss[-1]
