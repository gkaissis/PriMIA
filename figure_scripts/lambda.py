import multiprocessing as mp
import sys, os.path
import csv
from time import sleep
from copy import deepcopy
from argparse import Namespace, ArgumentParser
from sigma import writefile, visualize_file

sys.path.insert(0, os.path.split(sys.path[0])[0])

from train import main


def parallel_execution(args, results_dict):
    best_value = main(args, verbose=False)
    results_dict[str(args.mixup_lambda)] = best_value
    return best_value


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--visualize", action="store_true", help="just visuallize the results so far."
    )
    parser.add_argument("--num_runs", type=int, help="How many runs for each lambda?")
    cmd_args = parser.parse_args()
    if cmd_args.visualize:
        visualize_file("figure_scripts/lambda_results.csv")
        exit()
    args = Namespace(
        config="lambdafigure",
        train_federated=False,
        data_dir="data/server_simulation/worker1",
        visdom=False,
        encrypted_inference=False,
        cuda=True,
        websockets=False,
        differentially_private=False,
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
        mixup_prob=1.0,
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
        num_threads=16,
        save_file=None,
        name="lambda",
        # sync_every_n_batch=1,
        # wait_interval=0.1,
        # keep_optim_dict=False,
        # repetitions_dataset=1,
        # weighted_averaging=False,
    )
    lambdas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, None]
    mng = mp.Manager()
    for i in range(cmd_args.num_runs):
        processes = []
        results_dict = mng.dict()
        for l in lambdas:
            results_dict[str(l)] = None
            args_copy = deepcopy(args)
            args_copy.mixup_lambda = l
            processes.append(
                mp.Process(
                    name="lambda={:s}".format(str(l)),
                    target=parallel_execution,
                    args=(args_copy, results_dict),
                )
            )
        for p in processes:
            p.start()
            sleep(5)
        for p in processes:
            p.join()
        print(results_dict)
        writefile("figure_scripts/lambda_results.csv", results_dict, lambdas)
