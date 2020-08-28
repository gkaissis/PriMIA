import csv
from time import time
from copy import deepcopy
from argparse import Namespace, ArgumentParser
import sys, os.path
import pandas as pd
from matplotlib import pyplot as plt
from torch import set_num_threads


sys.path.insert(0, os.path.split(sys.path[0])[0])

from train import main


def writefile(file_name: str, input_dict: dict, headers: list):
    file_exists = os.path.isfile(file_name)
    with open(file_name, "a" if file_exists else "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[str(i) for i in headers])
        if not file_exists:
            writer.writeheader()
        writer.writerow(dict(input_dict))


def visualize_file(file_name: str):
    df = pd.read_csv(file_name)
    if "None" in df:
        df = df.rename(columns={"None": "random"})
    mean, std = df.mean(axis=0), df.std(axis=0)

    # plt.ylim(0.0, 100.0)
    plt.xticks(range(len(mean.index)), mean.index)
    plt.fill_between(range(len(mean.index)), mean - std, mean + std)
    plt.errorbar(
        range(len(mean.index)),
        mean,
        yerr=2 * std,
        fmt="o",
        color="black",
        ecolor="lightblue",
        elinewidth=3,
        capsize=0,
    )
    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--visualize", action="store_true", help="just visualize the results so far."
    )
    parser.add_argument("--num_runs", type=int, help="How many runs for each lambda?")
    parser.add_argument("--num_threads", type=int, help="How threads to use?")
    cmd_args = parser.parse_args()
    set_num_threads(args.num_threads)
    if cmd_args.visualize:
        visualize_file("figure_scripts/sigmas.csv")
        visualize_file("figure_scripts/sigma_times.csv")
        exit()
    args = Namespace(
        config="lambdafigure",
        train_federated=True,
        data_dir="data/server_simulation/",
        visdom=False,
        encrypted_inference=False,
        unencrypted_aggregation=False,
        differentially_private=False,
        cuda=False,
        websockets=False,
        batch_size=128,
        train_resolution=224,
        inference_resolution=224,
        test_batch_size=1,
        test_interval=1,
        validation_split=10,
        epochs=3,
        lr=1e-4,
        end_lr=1e-5,
        restarts=0,
        beta1=0.5,
        beta2=0.99,
        weight_decay=5e-4,
        deterministic=True,
        seed=42,
        log_interval=10,
        optimizer="Adam",
        model="resnet-18",
        pretrained=True,
        weight_classes=False,
        pooling_type="max",
        rotation=30,
        translate=0.0,
        scale=0.15,
        shear=10,
        mixup=True,
        mixup_lambda=None,
        mixup_prob=0.9,
        clahe=True,
        albu_prob=0.75,
        individual_albu_probs=0.2,
        noise_std=0.05,
        noise_prob=0.5,
        randomgamma=True,
        randombrightness=True,
        blur=True,
        elastic=True,
        optical_distortion=True,
        grid_distortion=True,
        grid_shuffle=False,
        hsv=False,
        invert=False,
        cutout=False,
        shadow=False,
        fog=True,
        sun_flare=False,
        solarize=False,
        equalize=False,
        grid_dropout=False,
        # sync_every_n_batch=1,
        wait_interval=0.1,
        keep_optim_dict=False,
        repetitions_dataset=5,
        weighted_averaging=False,
    )
    sigmas = range(1, 8)
    for i in range(cmd_args.num_runs):
        processes = []
        results_dict = {}
        time_dict = {}
        for s in sigmas:
            args_copy = deepcopy(args)
            args_copy.sync_every_n_batch = s
            t1 = time()
            results_dict[str(s)] = main(args_copy, verbose=False)
            t = time() - t1
            time_dict[str(s)] = t
        writefile("figure_scripts/sigmas.csv", results_dict, sigmas)
        writefile("figure_scripts/sigma_times.csv", time_dict, sigmas)
        print("Finished {:d} repetition".format(i + 1))

