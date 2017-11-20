
import tensorflow as tf

def weight_variable(shape, name):
  initializer = tf.truncated_normal_initializer(stddev=0.1)
  return tf.get_variable(shape=shape, initializer=initializer, name=name)

def bias_variable(shape, name):
  initializer = tf.constant_initializer(0.)
  return tf.get_variable(shape=shape, initializer=initializer, name=name)

class VAE(object):
  def __init__(self, args, session):
    self._input_dim = args.input_dim
    self._hidden_dim = args.hidden_dim
    self._latent_dim = args.latent_dim
    self._sess = session
    self._max_grad_norm = args.max_grad_norm

    self._build_placeholder()
    
    self._optimizer = tf.train.AdamOptimizer(self.lr)
    self.global_step = tf.get_variable('global_step', [], 'int32', tf.constant_initializer(0), trainable=False)

    self._build_forward()
    self._build_loss()
    self._build_train()
    
    self.summary = tf.summary.merge_all()
    init_op = tf.global_variables_initializer()
    self._sess.run(init_op)

  def _build_placeholder(self):
    with tf.name_scope("Data"):
      self.x_images = tf.placeholder(tf.float32, [None, self._input_dim])
      self.z = tf.placeholder(tf.float32, [None, self._latent_dim])
    with tf.name_scope("Hyparameter"):
      self.lr = tf.placeholder(tf.float32, [], name="learning_rate")

  def _build_forward(self):
    with tf.name_scope("Model"):
      with tf.variable_scope("Encoder"):
        self.mu, self.log_sigma_sq = self.encoder(self.x_images)
      with tf.variable_scope("Decoder"):
        sample_z = self.sample_z(self.mu, self.log_sigma_sq)
        self.logits, self.reconstruct = self.decoder(sample_z)

  def _build_loss(self):
    with tf.name_scope("Loss"):
      rec_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.x_images), 1)
      kl_loss = -0.5 * tf.reduce_sum(1. + self.log_sigma_sq - self.mu ** 2 - tf.exp(self.log_sigma_sq), 1)
      self.loss_op = tf.reduce_mean(rec_loss + kl_loss)

  def _build_train(self):
    with tf.name_scope('Train'):
      grads_and_vars = self._optimizer.compute_gradients(self.loss_op)
      grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v) for g, v in grads_and_vars]
      self.train_op = self._optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

  def batch_fit(self, x_images, lr, get_summary=False):
    feed_dict = {}
    feed_dict[self.x_images] = x_images
    feed_dict[self.lr] = lr
    loss, _, = self._sess.run([self.loss_op, self.train_op], feed_dict=feed_dict)

    return loss

  def encoder(self, x_images):
    e_w1 = weight_variable([self._input_dim, self._hidden_dim], name="E_W1")
    e_b1 = bias_variable([self._hidden_dim], name="E_b1")
    e_h1 = tf.nn.relu(tf.matmul(x_images, e_w1) + e_b1)
    e_wmu = weight_variable([self._hidden_dim, self._latent_dim], name="E_Wmu")
    e_bmu = bias_variable([self._latent_dim], name="E_bmu")
    e_hmu = tf.matmul(e_h1, e_wmu) + e_bmu
    e_wlog_sigma_sq = weight_variable([self._hidden_dim, self._latent_dim], name="E_Wlog_sigma_sq")
    e_blog_sigma_sq = bias_variable([self._latent_dim], name="E_blog_sigma_sq")
    e_hlog_sigma_sq = tf.matmul(e_h1, e_wlog_sigma_sq) + e_blog_sigma_sq

    return e_hmu, e_hlog_sigma_sq

  def sample_z(self, mu, log_sigma_sq):
    eps = tf.random_normal(shape=tf.shape(mu))

    return mu + tf.exp(log_sigma_sq / 2) * eps

  def decoder(self, z):
    d_w1 = weight_variable([self._latent_dim, self._hidden_dim], name="D_W1")
    d_b1 = bias_variable([self._hidden_dim], name="D_b1")
    d_h1 = tf.nn.relu(tf.matmul(z, d_w1) + d_b1)
    d_w2 = weight_variable([self._hidden_dim, self._input_dim], name="D_W2")
    d_b2 = bias_variable([self._input_dim], name="D_b2")
    d_l2 = tf.matmul(d_h1, d_w2) + d_b2
    d_h2 = tf.nn.sigmoid(d_l2)

    return d_l2, d_h2

  def generate(self, z):
    feed_dict= {self.z: z}
    with tf.variable_scope("Decoder", reuse=True):
      _, d_h2 = self.decoder(self.z)
    
    return self._sess.run(d_h2, feed_dict=feed_dict)

  def reconstruct(self, x_images):
    feed_dict = {self.x_images: x_images}
    
    return self._sess.run(self.reconstruct, feed_dict)