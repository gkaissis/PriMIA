import os
import pandas as pd
from random import shuffle, seed
from tqdm import tqdm
from shutil import copyfile

if __name__ == "__main__":

    cur_path = os.path.abspath(os.getcwd())

    if os.path.split(cur_path)[1] != "server_simulation":
        print(
            """Be very careful, this script creates folders and distributes data.
Only execute this from 4P/data/server_simulation"""
        )
        inpt = input("Do you really wish to proceed? [y/N]\t").lower()
        if inpt not in ["y", "yes"] or input("Are you really sure? [y/N]\t").lower() not in ["y", "yes"]:
            print("aborting")
            exit()

    labels = pd.read_csv("../Labels.csv")
    labels = labels[labels["Dataset_type"] == "TRAIN"]
    path_to_data = "../train"
    class_names = {0: "normal", 1: "bacterial pneumonia", 2: "viral pneumonia"}
    worker_dirs = ["worker1", "worker2", "worker3"]
    worker_imgs = {name: [] for name in worker_dirs}
    L = len(labels)
    num_workers = len(worker_dirs)
    shuffled_idcs = list(range(L))
    seed(0)
    shuffle(shuffled_idcs)
    for i in range(num_workers):
        idcs_worker = shuffled_idcs[i::num_workers]
        worker_imgs[worker_dirs[i]] = [labels.iloc[j] for j in idcs_worker]
    """for i in tqdm(range(L), total=L, leave=False):
        sample = labels.iloc[i]
        worker_imgs[worker_dirs[i % num_workers]].append(sample)"""
    for c in class_names.values():
        p = os.path.join("all_samples", c)
        if not os.path.isdir(p):
            os.makedirs(p)
        for w in worker_dirs:
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
                "all_samples", class_names[s["Numeric_Label"]], s["X_ray_image_name"]
            )
            copyfile(src_file, dst_file)
            copyfile(src_file, all_dst)
