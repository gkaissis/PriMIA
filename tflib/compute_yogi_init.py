# Lint as: python3
# Copyright 2019, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Computes an estimate for the Yogi initial accumulator using TFF."""

import functools
import pathlib

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import tensorflow_federated as tff

import data_helpers as data
import model_helpers as model
from tff_optim import optimizer_utils

# Experiment hyperparameters
flags.DEFINE_integer('client_batch_size', 32, 'Batch size on the clients.')
flags.DEFINE_integer('num_clients', 10,
                     'Number of clients to use for estimating the L2-norm'
                     'squared of the batch gradients.')

# End of hyperparameter flags.

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
  train_client_ids = data.make_client_ids(train_path)
  train_client_fn = data.provide_client_data_fn(
      train_path, FLAGS.client_batch_size)

  train_clients = tff.simulation.ClientData.from_clients_and_fn(
      train_client_ids, train_client_fn)
  input_spec = train_clients.create_tf_dataset_for_client(
      train_clients.client_ids[0]).element_spec
  model_builder = model.model_fn_factory(FLAGS.model)
  tff_model = tff.learning.from_keras_model(
      keras_model=model_builder(),
      input_spec=input_spec,
      loss=tf.keras.losses.SparseCategoricalCrossentropy())

  yogi_init_accum_estimate = optimizer_utils.compute_yogi_init(
      train_clients, tff_model, num_clients=FLAGS.num_clients)
  logging.info('Yogi initializer: {:s}'.format(
      format(yogi_init_accum_estimate, '10.6E')))


if __name__ == '__main__':
  app.run(main)
