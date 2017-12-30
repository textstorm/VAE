
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
    print "Epoch %d start with learning rate %f" % (epoch, args.learning_rate)
    print "- " * 50
    epoch_start_time = time.time()
    step_start_time = epoch_start_time
    for i in range(1, total_batch + 1):
      x_batch, y_batch = mnist.train.next_batch(args.batch_size)
      _, loss, loss_rec, loss_kl, global_step = model.train(x_batch)
      step_start_time = time.time()

      if global_step % args.log_period == 0:
        print "global step %d, loss %.9f, loss_rec %.9f, loss_kl %.9f, time %.2fs" \
            % (global_step, loss, loss_rec, loss_kl, time.time()-step_start_time)
    
    if epoch % args.save_period == 0:
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