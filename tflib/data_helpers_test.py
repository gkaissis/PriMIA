import pathlib

from absl.testing import parameterized
from absl.testing import absltest
import numpy as np
import tensorflow as tf

import data_helpers


class ClientFnTest(parameterized.TestCase):
  @parameterized.parameters("data/train_clients", "data/test_clients")
  def test_client_fn(self, clients_dir):
    clients_dir = pathlib.Path(clients_dir)
    client_data_fn = data_helpers.provide_client_data_fn(
        clients_dir, 224, 224)
    
    ds = client_data_fn("client-08")
    elt = next(ds.take(1).as_numpy_iterator())

    self.assertIsInstance(ds, tf.data.Dataset)
    self.assertEqual(elt[0].dtype, np.float32)
    self.assertEqual(elt[0].shape, (224, 224, 3))
    self.assertIsInstance(elt[1], np.int32)
    

if __name__ == '__main__':
  absltest.main()