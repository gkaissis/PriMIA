import pathlib
import random

from absl import app
from absl import flags
import tensorflow as tf
import tensorflow_federated as tff

from data_helpers import make_client_ids
from data_helpers import provide_client_data_fn
from model_helpers import build_vgg16


# Hyperparams
flags.DEFINE_integer("num_rounds", default=10,
    help="Number of rounds of federated averaging.")
flags.DEFINE_integer("clients_per_round", default=10,
    help="Number of clients to sample for training per round.")
flags.DEFINE_float("client_learning_rate", default=.02,
    help="Learning rate for client optimizers.")
flags.DEFINE_float("server_learning_rate", default=1.0,
    help="Learning rate for client optimizers.")
flags.DEFINE_bool("freeze_model", default=True,
    help="Freeze early layers in the model (if its builder fn allows)")
flags.DEFINE_integer("image_width", default=224,
    help="Width dimension of input radiology images.")
flags.DEFINE_integer("image_height", default=224,
    help="Height dimension of input radiology images.")
flags.DEFINE_integer("batch_size", default=4,
    help="Local batch size for each client.")
flags.DEFINE_enum("model", default="vgg16", enum_values=["vgg16"],
    help="Which model to use. Must have a builder in model_helpers.")

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
  dataroot = pathlib.Path(FLAGS.data_root)
  train_path = dataroot.joinpath(FLAGS.train_clients_subdir)
  test_path = dataroot.joinpath(FLAGS.test_clients_subdir)
  train_client_ids = make_client_ids(train_path)
  test_client_ids = make_client_ids(test_path)

  img_dims = (FLAGS.image_width, FLAGS.image_height)
  train_client_fn = provide_client_data_fn(train_path, *img_dims, FLAGS.batch_size)
  test_client_fn = provide_client_data_fn(test_path, *img_dims, FLAGS.batch_size)

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

  client_opt_fn = lambda: tf.keras.optimizers.SGD(FLAGS.client_learning_rate)
  server_opt_fn = lambda: tf.keras.optimizers.SGD(FLAGS.server_learning_rate)

  iterative_process = tff.learning.build_federated_averaging_process(
      model_fn, client_opt_fn, server_opt_fn)

  state = iterative_process.initialize()
  for rnd in range(FLAGS.num_rounds):
    round_clients = random.sample(federated_train_data, FLAGS.clients_per_round)
    state, metrics = iterative_process.next(state, round_clients)
    print('round  {rnd}, metrics={metrics}'.format(rnd=rnd, metrics=metrics))


def model_fn():
  x_spec = (tf.float32, [None, 224, 224, 3])
  y_spec = (tf.int64, [None])
  input_spec = (x_spec, y_spec)
  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  model = build_vgg16(freeze=FLAGS.freeze_model)
  return tff.learning.from_keras_model(
      model, loss_fn, input_spec=input_spec,
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


if __name__ == "__main__":
  app.run(main)
