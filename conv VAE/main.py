
import vae
import utils
import config

import numpy as np
import tensorflow as tf
import os

def main(args):
  
  mnist = utils.read_data_sets(args.train_dir)
  config_proto = utils.get_config_proto()

  with tf.device('/gpu:1'):
    if not os.path.exists("../saves"):
      os.mkdir("../saves")
    sess = tf.Session(config=config_proto)
    model = vae.ConvVAE(args, sess)
    total_batch = mnist.train.num_examples // args.batch_size

    for epoch in range(1, args.nb_epoch + 1):
      for i in range(1, total_batch + 1):
        global_step = sess.run(model.global_step)
        x_batch, y_batch = mnist.train.next_batch(args.batch_size)
        loss = model.train(x_batch)

        if i % args.log_period == 0:
          print "Epoch: %1d, Batch: %04d, loss: %9.9f" % (epoch, i, loss)
      
      if epoch % 50 == 0:
        print "- " * 50
      if epoch % args.save_period == 0:
        if not os.path.exists("../saves/imgs"):
          os.mkdir("../saves/imgs")
        z = np.random.normal(size=[100, args.latent_dim])
        gen_images = np.reshape(model.generate(z), (100, 28, 28, 1))
        utils.save_images(gen_images, [10, 10], os.path.join(args.save_dir, "imgs/sample%s.jpg" % epoch))

    save_path="../saves/model.ckpt"
    saver = tf.train.Saver()
    saver.save(session, save_path)
    print "Model stored...."

if __name__ == "__main__":
  args = config.get_args()
  main(args)