import os
import pandas as pd
from random import shuffle, seed
from tqdm import tqdm
import argparse
from shutil import copyfile
import sys, os.path

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

    # Settings
    label_file = pd.read_csv("../Labels.csv")
    labels = label_file[label_file["Dataset_type"] == "TRAIN"]
    path_to_data = "../train"
    class_names = {0: "normal", 1: "bacterial pneumonia", 2: "viral pneumonia"}
    num_workers = 3

    # actual code
    worker_dirs = ["worker{:d}".format(i + 1) for i in range(num_workers)]
    worker_imgs = {name: [] for name in worker_dirs}
    L = len(labels)
    shuffled_idcs = list(range(L))
    seed(0)
    shuffle(shuffled_idcs)
    split_idx = len(shuffled_idcs) // 10
    ## first ten per cent are validation set
    train_set, val_set = shuffled_idcs[split_idx:], shuffled_idcs[:split_idx]
    for i in range(num_workers):
        idcs_worker = train_set[i::num_workers]
        worker_imgs["worker{:d}".format(i + 1)] = [labels.iloc[j] for j in idcs_worker]
    worker_imgs["validation"] = [labels.iloc[j] for j in val_set]
    """for i in tqdm(range(L), total=L, leave=False):
        sample = labels.iloc[i]
        worker_imgs[worker_dirs[i % num_workers]].append(sample)"""
    for c in class_names.values():
        for ps in ["train_total", "test"]:
            p = os.path.join(ps, c)
            if not os.path.isdir(p):
                os.makedirs(p)
        for w in worker_imgs.keys():
            p = os.path.join(w, c)
            if not os.path.isdir(p):
                os.makedirs(p)
    for name, samples in tqdm(worker_imgs.items(), total=len(worker_imgs), leave=False):
        for s in tqdm(samples, total=len(samples), leave=False):
            src_file = os.path.join(path_to_data, s["X_ray_image_name"])
            dst_file = os.path.join(
                name, class_names[s["Numeric_Label"]], s["X_ray_image_name"]
            )
            all_dst = os.path.join(
                "train_total", class_names[s["Numeric_Label"]], s["X_ray_image_name"]
            )
            if args.symbolic:
                os.symlink(os.path.abspath(src_file), dst_file)
                os.symlink(os.path.abspath(src_file), all_dst)
            else:
                copyfile(src_file, dst_file)
                copyfile(src_file, all_dst)

    labels = label_file[label_file["Dataset_type"] == "TEST"]
    path_to_data = "../test"
    for i in tqdm(
        range(len(labels)), total=len(labels), desc="create test folder", leave=False
    ):
        s = labels.iloc[i]
        src_file = os.path.join(path_to_data, s["X_ray_image_name"])
        dst_file = os.path.join(
            "test", class_names[s["Numeric_Label"]], s["X_ray_image_name"]
        )
        if args.symbolic:
            os.symlink(os.path.abspath(src_file), dst_file)
        else:
            copyfile(src_file, dst_file)
