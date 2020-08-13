import os
import pandas as pd
from random import shuffle, seed
from tqdm import tqdm
import argparse
from shutil import copyfile
import sys, os.path

from torchvision.datasets import ImageFolder

sys.path.insert(0, os.path.split(os.path.split(sys.path[0])[0])[0])

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--symbolic",
        help="Create symlinks instead of copying files.",
        default=False,
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "--num_workers",
        default=3,
        type=int,
        help="How many servers should be simulated.",
    )
    parser.add_argument(
        "--train_data_src",
        default="../train",
        type=str,
        help="Source data folder for training data.",
    )
    # parser.add_argument(
    #     "--test_data_src",
    #     default="../test",
    #     type=str,
    #     help="Source data folder for test data.",
    # )
    parser.add_argument
    args = parser.parse_args()
    cur_path = os.path.abspath(os.getcwd())

    if os.path.split(cur_path)[1] != "server_simulation":
        print(
            "Be very careful, this script creates folders and distributes data."
            "Only execute this from 4P/data/server_simulation"
        )
        inpt = input("Do you really wish to proceed? [y/N]\t").lower()
        if inpt not in ["y", "yes"] or input(
            "Are you really sure? [y/N]\t"
        ).lower() not in ["y", "yes"]:
            print("aborting")
            exit()

    # actual code
    worker_dirs = ["worker{:d}".format(i + 1) for i in range(args.num_workers)]
    worker_imgs = {name: [] for name in worker_dirs}

    train_imgs = ImageFolder(args.train_data_src)
    L = len(train_imgs)
    shuffled_idcs = list(range(L))
    seed(0)
    shuffle(shuffled_idcs)
    split_idx = len(shuffled_idcs) // 10
    ## first ten per cent are validation set
    train_idcs, val_idcs = shuffled_idcs[split_idx:], shuffled_idcs[:split_idx]
    for c in train_imgs.classes:
        # for ps in ["train_total", "test"]:
        #     p = os.path.join(ps, c)
        #     if not os.path.isdir(p):
        #         os.makedirs(p)
        for w in worker_imgs.keys():
            p = os.path.join(w, c)
            if not os.path.isdir(p):
                os.makedirs(p)
    for i in range(args.num_workers):
        idcs_worker = train_idcs[i :: args.num_workers]
        worker_imgs["worker{:d}".format(i + 1)] = idcs_worker
    worker_imgs["validation"] = val_idcs
    for name, idcs in tqdm(worker_imgs.items(), total=len(worker_imgs), leave=False):
        for idx in tqdm(
            idcs, total=len(idcs), leave=False, desc="save data to {:s}".format(name)
        ):
            src_file, class_idx = train_imgs.samples[idx]
            file_name = os.path.split(src_file)[1]
            target_file = os.path.join(name, train_imgs.classes[class_idx], file_name)
            if args.symbolic:
                os.symlink(os.path.abspath(src_file), target_file)
            else:
                copyfile(src_file, target_file)
    # test_imgs = ImageFolder(args.test_data_src)
    # for path, class_idx in tqdm(
    #     test_imgs.samples, total=len(test_imgs), desc="create test folder", leave=False,
    # ):
    #     src_file = path
    #     file_name = os.path.split(src_file)[1]
    #     dst_file = os.path.join("test", test_imgs.classes[class_idx], file_name)
    #     print("Copy {:s} to \t{:s}".format(src_file, dst_file))
    #     if args.symbolic:
    #         os.symlink(os.path.abspath(src_file), dst_file)
    #     else:
    #         copyfile(src_file, dst_file)
