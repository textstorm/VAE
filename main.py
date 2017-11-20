
import vae
import utils
import config

import numpy as np
import tensorflow as tf
import os

def main(args):
  
  mnist = utils.read_data_sets(args.train_dir)
  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth = True

  with tf.device('/gpu:1'):
    if not os.path.exists("../saves"):
      os.mkdir("../saves")
    session = tf.Session(config=config)
    model = vae.VAE(args, session)
    total_batch = mnist.train.num_examples // args.batch_size

    for epoch in range(1, args.nb_epoch + 1):
      for i in range(total_batch):
        global_step = session.run(model.global_step)
        x_batch, y_batch = mnist.train.next_batch(args.batch_size)
        loss = model.batch_fit(x_batch, args.learning_rate)
          
        if i % 100 == 0:
          print "Epoch: ", '%1d' % epoch, "Batch: ", '%04d' % i, "loss: ", '%9.9f' % loss
      
      if epoch % 100 == 0:
        if not os.path.exists("../saves/imgs"):
          os.mkdir("../saves/imgs")
        z = np.random.normal(size=[64, args.latent_dim])
        gen_images = np.reshape(model.generate(z), (64, 28, 28, 1))
        utils.save_images(gen_images, [8, 8], os.path.join(args.save_dir, "imgs/sample%s.jpg" % epoch))

    save_path="../saves/model.ckpt"
    saver = tf.train.Saver()
    saver.save(session, save_path)
    print ("Model stored....")

if __name__ == "__main__":
  args = config.get_args()
  main(args)