import pathlib
import random

from absl import app
from absl import flags
import tensorflow as tf
import tensorflow_federated as tff

import data_helpers as data
import model_helpers as model


# Training hparams
flags.DEFINE_integer("num_rounds", default=10,
    help="Number of rounds of federated averaging.")
flags.DEFINE_integer("clients_per_round", default=10,
    help="Number of clients to sample for training per round.")
flags.DEFINE_float("client_learning_rate", default=.02,
    help="Learning rate for client optimizers.")
flags.DEFINE_float("server_learning_rate", default=1.0,
    help="Learning rate for client optimizers.")
flags.DEFINE_integer("client_batch_size", default=4,
    help="Local batch size for each client.")

# Data flags
flags.DEFINE_string("data_root", default="./data",
    help="Path to the root folder containing chest xray data")
flags.DEFINE_string("train_clients_subdir", default="train_clients",
    help="Subdirectory of `data_root` containing data allocated to the "
         "training subset of clients.")
flags.DEFINE_string("test_clients_subdir", default="test_clients",
    help="Subdirectory of `data-root` containing data allocated to the "
         "evaluation subset of clients.")

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Expected no command-line arguments, '
                         'got: {}'.format(argv))
  dataroot = pathlib.Path(FLAGS.data_root)
  train_path = dataroot.joinpath(FLAGS.train_clients_subdir)
  test_path = dataroot.joinpath(FLAGS.test_clients_subdir)
  train_client_ids = data.make_client_ids(train_path)
  test_client_ids = data.make_client_ids(test_path)

  train_client_fn = data.provide_client_data_fn(
      train_path, FLAGS.client_batch_size)
  test_client_fn = data.provide_client_data_fn(
      test_path, FLAGS.client_batch_size)

  train_clients = tff.simulation.ClientData.from_clients_and_fn(
      train_client_ids, train_client_fn)
  test_clients = tff.simulation.ClientData.from_clients_and_fn(
      test_client_ids, test_client_fn)

  federated_train_data = [
      train_clients.create_tf_dataset_for_client(client_id)
      for client_id in train_client_ids
  ]
  federated_test_data = [
      test_clients.create_tf_dataset_for_client(client_id)
      for client_id in test_client_ids
  ]

  model_fn = model.model_fn_factory(FLAGS.model)
  client_opt_fn = lambda: tf.keras.optimizers.SGD(FLAGS.client_learning_rate)
  server_opt_fn = lambda: tf.keras.optimizers.SGD(FLAGS.server_learning_rate)

  iterative_process = tff.learning.build_federated_averaging_process(
      model_fn, client_opt_fn, server_opt_fn)

  state = iterative_process.initialize()
  for rnd in range(FLAGS.num_rounds):
    round_clients = random.sample(
        federated_train_data, FLAGS.clients_per_round)
    state, metrics = iterative_process.next(state, round_clients)
    print('round  {rnd}, metrics={metrics}'.format(rnd=rnd, metrics=metrics))


if __name__ == "__main__":
  app.run(main)
