
import utils
import config
import numpy as np

def main(args):
  mnist = utils.read_data_sets(args.train_dir)
  n_examples = 100
  sample, _ = mnist.test.next_batch(n_examples)
  sample = np.reshape(sample, [-1, 28, 28, 1])
  utils.save_images(sample, [10, 10], "sample.jpg")


if __name__ == '__main__':
  args = config.get_args()
  main(args)