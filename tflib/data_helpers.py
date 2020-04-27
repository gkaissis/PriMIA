import os
import pathlib

import tensorflow as tf
import tensorflow_federated as tff


AUTOTUNE = tf.data.experimental.AUTOTUNE

def make_client_ids(clients_dir: pathlib.Path):
  return [p.name for p in clients_dir.iterdir() if p.is_dir()]


def provide_client_data_fn(
    clients_dir: pathlib.Path,
    img_height: int,
    img_width: int,
    batch_size: int,
    augment_fn: callable = None):
  process_path = _provide_process_fn(
      img_width=img_width, img_height=img_height)

  def create_tf_dataset_for_client(client_id):
    this_client = clients_dir.joinpath(client_id)
    image_glob = this_client / '*/*'
    ds = tf.data.Dataset.list_files(str(image_glob))
    ds = ds.map(process_path, num_parallel_calls=AUTOTUNE)
    if augment_fn is not None:
      ds = ds.map(augment_fn, num_parallel_calls=AUTOTUNE)
    return ds.batch(batch_size)

  return create_tf_dataset_for_client


def _get_label(file_path):
  # convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  return tf.cast(int(parts[-2]), tf.int64)


def _decode_img(img, img_width, img_height):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.
  return tf.image.resize(img, [img_width, img_height])


def _provide_process_fn(**decode_params):

  def process_path(file_path):
    label = _get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = _decode_img(img, **decode_params)
    return img, label

  return process_path


# data = ClientData.from_clients_and_fn(client_ids, create_tf_dataset_for_client)