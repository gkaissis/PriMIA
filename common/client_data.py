import abc
import argparse
import functools
import os
import pathlib
import shutil

import numpy as np
import pandas as pd


def parse_args():
  parser = argparse.ArgumentParser(description="Client allocation")
  parser.add_argument('-c', '--train-clients', default=100, type=int)
  parser.add_argument('-t', '--test-clients', default=12, type=int)
  parser.add_argument('-s', '--seed', default=42, type=int)
  parser.add_argument('-d', '--data-root', default='./data')
  parser.add_argument('--train-clients-subdir', default='train_clients')
  parser.add_argument('--test-clients-subdir', default='test_clients')
  return parser.parse_args()


def split_dataframe(df, split):
  assert split in ['train', 'test']
  imgs = df[df['Dataset_type'] == split.upper()].drop('Dataset_type', 1)
  return imgs


def make_client_ids(num_clients):
  return ["client-{:02d}".format(i) for i in range(num_clients)]


def split_dataframe_for_clients(df, client_ids):
  splits = np.array_split(
      df.loc[np.random.permutation(df.index)], len(client_ids))
  return dict(zip(client_ids, splits))


def allocate_samples_on_disk(
    client_samples,
    data_root: pathlib.Path,
    split_subdir: str,
    clients_subdir: str,
):
  split_root = data_root / split_subdir  # e.g. 4P/data/train
  clients_root = data_root / clients_subdir  # e.g. 4P/data/train_clients
  
  for client_id, sample in client_samples.items():
    # e.g. 4P/data/train_clients/03/
    client_dir_name = "{:2d}".format(client_id)
    client_path = clients_root / client_dir_name

    for label in ["0", "1", "2"]:
      (client_path / label).mkdir(parents=True, exist_ok=True)

    for imname, label in zip(sample.X_ray_image_name, sample.Numeric_Label):
      shutil.copy(split_root / imname, client_path / str(label) / imname)


def main(args):
  np.random.seed(args.seed)
  data_root = pathlib.Path(args.data_root)
  df = pd.read_csv(data_root.joinpath("Labels.csv"))

  train_df = split_dataframe(df, 'train')
  test_df = split_dataframe(df, 'test')

  train_ids = make_client_ids(args.train_clients)
  test_ids = make_client_ids(args.test_clients)

  train_splits = split_dataframe_for_clients(train_df, train_ids)
  test_splits = split_dataframe_for_clients(test_df, test_ids)

  allocate_samples_on_disk(
      train_splits, data_root, 'train', args.train_clients_subdir)
  allocate_samples_on_disk(
      test_splits, data_root, 'test', args.test_clients_subdir)


if __name__ == '__main__':
  args = parse_args()
  main(args)
  


