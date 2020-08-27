import csv
from time import time
from copy import deepcopy
from argparse import Namespace, ArgumentParser
import sys, os.path

sys.path.insert(0, os.path.split(sys.path[0])[0])

from train import main


def writefile(file_name: str, input_dict: dict, headers: list):
    file_exists = os.path.isfile(file_name)
    with open(file_name, "a" if file_exists else "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[str(i) for i in headers])
        if not file_exists:
            writer.writeheader()
        writer.writerow(dict(input_dict))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num_runs", type=int, help="How many runs for each lambda?")
    cmd_args = parser.parse_args()
    # if cmd_args.visualize:
    #     import pandas as pd
    #     from matplotlib import pyplot as plt

    #     df = pd.read_csv("lambda_results.csv").rename(columns={"None": "random"})
    #     mean, std = df.mean(axis=0), df.std(axis=0)

    #     # plt.ylim(0.0, 100.0)
    #     plt.xticks(range(len(mean.index)), mean.index)
    #     plt.fill_between(range(len(mean.index)), mean - std, mean + std)
    #     plt.errorbar(
    #         range(len(mean.index)),
    #         mean,
    #         yerr=2 * std,
    #         fmt="o",
    #         color="black",
    #         ecolor="lightblue",
    #         elinewidth=3,
    #         capsize=0,
    #     )
    #     plt.show()
    #     exit()
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
        epochs=40,
        lr=1e-4,
        end_lr=1e-5,
        restarts=0,
        beta1=0.5,
        beta2=0.99,
        weight_decay=5e-4,
        deterministic=False,
        seed=42,
        log_interval=10,
        optimizer="Adam",
        model="resnet-18",
        pretrained=True,
        weight_classes=False,
        pooling_type="max",
        rotation=0,
        translate=0.0,
        scale=0.0,
        shear=0,
        mixup=True,
        mixup_lambda=0.0,
        mixup_prob=0.0,
        clahe=False,
        albu_prob=0.0,
        individual_albu_probs=0.0,
        noise_std=0.0,
        noise_prob=0.0,
        randomgamma=False,
        randombrightness=False,
        blur=False,
        elastic=False,
        optical_distortion=False,
        grid_distortion=False,
        grid_shuffle=False,
        hsv=False,
        invert=False,
        cutout=False,
        shadow=False,
        fog=False,
        sun_flare=False,
        solarize=False,
        equalize=False,
        grid_dropout=False,
        # sync_every_n_batch=1,
        wait_interval=0.1,
        keep_optim_dict=False,
        repetitions_dataset=1,
        weighted_averaging=False,
    )
    sigmas = range(10)
    for i in range(cmd_args.num_runs):
        processes = []
        results_dict = {}
        time_dict = {}
        for s in sigmas:
            args_copy = deepcopy(args)
            args_copy.sync_every_n_batch = s
            t1 = time()
            results_dict[s] = main(args_copy, verbose=False)
            t = time() - t1
            time_dict[s] = t

        writefile("figure_scripts/sigmas.csv", results_dict, sigmas)
        writefile("figure_scripts/sigma_times.csv", time_dict, sigmas)
