import argparse

def get_args():
  """
      The argument parser
  """
  parser = argparse.ArgumentParser()

  parser.add_argument('--random_seed', type=int, default=827, help='Random seed')

  parser.add_argument('--train_dir', type=str, default='data', help='data path')
  parser.add_argument('--log_dir', type=str, default='save/logs', help='log path')
  parser.add_argument('--save_dir', type=str, default='save/saves', help='save path')
  parser.add_argument('--nb_classes', type=int, default=10, help='number classe')
  parser.add_argument('--model_type', type=str, default="vae", help='model type')

  parser.add_argument('--input_dim', type=int, default=784, help='Dimension of input data')
  parser.add_argument('--hidden_dim', type=int, default=256, help="Dimension of hidden layer")
  parser.add_argument('--latent_dim', type=int, default=100, help="Dimension of latent variable")

  parser.add_argument('--batch_size', type=int, default=100, help='Train batch size')
  parser.add_argument('--nb_epoch', type=int, default=1000, help='The number of epoch')
  parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
  parser.add_argument('--max_grad_norm', type=float, default=10.0, help='Max norm of gradient')
  parser.add_argument('--log_period', type=int, default=100, help='number step to print')
  parser.add_argument('--save_period', type=int, default=50, help='number epoch to save')

  return parser.parse_args()