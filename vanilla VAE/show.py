
import vae
import utils
import config

import os
import numpy as np
import tensorflow as tf
from functools import reduce

def gen_z(size, img_num):
  z_sample_1 = np.random.normal(size=size)
  z_sample_2 = np.random.normal(size=size)
  delta_z = z_sample_2 - z_sample_1
  gap_z = delta_z / (img_num - 1)
  all_z = []
  for i in range(11):
    all_z.append(z_sample_1 + i*gap_z)
  def f(a, b):
    return np.row_stack([a, b])

  return reduce(f, all_z)

with tf.device('/gpu:0'):
  args = config.get_args()
  config_proto = utils.get_config_proto()

  sess = tf.Session(config=config_proto)
  model = vae.VAE(args, sess)

  saver = tf.train.Saver()
  save_path = os.path.join(args.save_dir, "model.ckpt")
  saver.restore(sess, save_path)
  print "Model restored."

  z = []
  for _ in range(11):
    z.append(gen_z([1, 100], 11))
  z = np.row_stack(z)
  gen_images = np.reshape(model.generate(z), (121, 28, 28, 1))
  utils.save_images(gen_images, [11, 11], "temp.png")
