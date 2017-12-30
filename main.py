
import tensorflow as tf
import numpy as np
import utils
import config
import os

from model import VAE, DCVAE

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main(args):
  #
  save_dir, log_dir = args.save_dir, args.log_dir
  train_dir = args.train_dir

  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)

  mnist = utils.read_data_sets(args.train_dir)
  config_proto = utils.get_config_proto()

  sess = tf.Session(config=config_proto)
  if args.model_type == "vae":
    model = VAE(args, sess, name="vae")
  elif args.model_type == "dcvae":
    model = DCVAE(args, sess, name="dcvae")

  total_batch = mnist.train.num_examples // args.batch_size

  for epoch in range(1, args.nb_epoch + 1):
    for i in range(1, total_batch + 1):
      global_step = sess.run(model.global_step)
      x_batch, y_batch = mnist.train.next_batch(args.batch_size)
      loss, loss_rec, loss_kl = model.train(x_batch, args.learning_rate)

      if i % args.log_period == 0:
        print "Epoch: %1d, Batch: %04d, loss: %9.9f, loss_rec: %9.9f, loss_kl: %9.9f" \
            % (epoch, i, loss, loss_rec, loss_kl)
    
    if epoch % 50 == 0:
      print "- " * 50
    if epoch % args.save_period == 0:
      if not os.path.exists("../saves/imgs"):
        os.mkdir("../saves/imgs")
      z = np.random.normal(size=[100, args.latent_dim])
      gen_images = np.reshape(model.generate(z), (100, 28, 28, 1))
      utils.save_images(gen_images, [10, 10], os.path.join(args.save_dir, "imgs/sample%s.jpg" % epoch))

  save_path = os.path.join(args.save_dir, "model.ckpt")
  saver = tf.train.Saver()
  saver.save(sess, save_path)
  print "Model stored...."

if __name__ == "__main__":
  args = config.get_args()
  main(args)