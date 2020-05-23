import functools
import pathlib

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import tensorflow_federated as tff

import data_helpers as data
import model_helpers as model
from tff_optim import fed_avg_schedule
from tff_optim import iterative_process_builder
from tff_optim.utils import training_loop
from tff_optim.utils import training_utils
from tff_optim.utils import utils_impl

with utils_impl.record_hparam_flags():
  # Experiment hyperparameters
  flags.DEFINE_integer('client_epochs_per_round', 1,
                       'Number of epochs in the client to take per round.')
  flags.DEFINE_integer('client_batch_size', 32, 'Batch size on the clients.')
  flags.DEFINE_integer('clients_per_round', 2,
                       'How many clients to sample per round.')

  # End of hyperparameter flags.

# Data flags
flags.DEFINE_string('data_root', default='./data',
    help='Path to the root folder containing chest xray data')
flags.DEFINE_string('train_clients_subdir', default='train_clients',
    help='Subdirectory of `data_root` containing data allocated to the '
         'training subset of clients.')
flags.DEFINE_string('test_clients_subdir', default='test_clients',
    help='Subdirectory of `data-root` containing data allocated to the '
         'evaluation subset of clients.')

FLAGS = flags.FLAGS

def main(argv):
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

  input_spec = train_clients.create_tf_dataset_for_client(
      train_clients.client_ids[0]).element_spec
  model_builder = model.model_fn_factory(FLAGS.model)

  logging.info('Training model:')
  logging.info(model_builder().summary())

  loss_builder = tf.keras.losses.SparseCategoricalCrossentropy
  metrics_builder = lambda: [tf.keras.metrics.SparseCategoricalAccuracy()]

  training_process = iterative_process_builder.from_flags(
      input_spec=input_spec,
      model_builder=model_builder,
      loss_builder=loss_builder,
      metrics_builder=metrics_builder)

  client_datasets_fn = training_utils.build_client_datasets_fn(
      train_clients, FLAGS.clients_per_round)

  assign_weights_fn = fed_avg_schedule.ServerState.assign_weights_to_keras_model

  eval_dataset = test_clients.create_tf_dataset_from_all_clients()
  evaluate_fn = training_utils.build_evaluate_fn(
      eval_dataset=eval_dataset,
      model_builder=model_builder,
      loss_builder=loss_builder,
      metrics_builder=metrics_builder,
      assign_weights_to_keras_model=assign_weights_fn)

  training_loop.run(
      iterative_process=training_process,
      client_datasets_fn=client_datasets_fn,
      evaluate_fn=evaluate_fn)


if __name__ == '__main__':
  app.run(main)
