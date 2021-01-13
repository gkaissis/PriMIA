import optuna as opt
from argparse import Namespace
import sys, os.path
import torch
from psutil import cpu_count
from sqlalchemy.exc import OperationalError

torch.set_num_threads(cpu_count())  # pylint:disable=no-member

sys.path.insert(0, os.path.split(sys.path[0])[0])

from train import main

global cmdln_args


def objective(trial: opt.trial):
    lr = trial.suggest_loguniform("lr", 1e-5, 2e-1,)
    repetitions_dataset = (
        trial.suggest_int("repetitions_dataset", 1, 3) if cmdln_args.federated else 1
    )
    epochs = 25
    if cmdln_args.federated:
        epochs = int(epochs // repetitions_dataset)
    args = Namespace(
        config=f"optuna_DP{cmdln_args.trial_name}",
        resume_checkpoint=None,
        train_federated=cmdln_args.federated,
        data_dir=cmdln_args.data_dir,
        visdom=False,
        encrypted_inference=False,
        cuda=not cmdln_args.federated,
        websockets=cmdln_args.websockets,
        batch_size=trial.suggest_int("batch_size", 32, 200),
        train_resolution=224,
        inference_resolution=224,
        test_batch_size=200,
        test_interval=1,
        validation_split=5,  # meaningless in pneumonia, uses data/test by default
        epochs=epochs,
        lr=lr,
        end_lr=trial.suggest_loguniform("end_lr", 1e-6, lr),
        restarts=trial.suggest_int("restarts", 0, 1),
        beta1=trial.suggest_float("beta1", 0.25, 0.95),
        beta2=trial.suggest_float("beta2", 0.9, 1.0),
        ## zero not possible but loguniform makes most sense
        weight_decay=0,
        seed=1,
        log_interval=10,
        deterministic=False,
        optimizer="Adam",
        model="resnet-18",
        pretrained=True,
        weight_classes=trial.suggest_categorical("weight_classes", [True, False]),
        pooling_type="max",
        rotation=0,
        translate=0.0,  # trial.suggest_float("translate", 0, 0.2),
        scale=0,
        shear=0,
        noise_std=0,
        noise_prob=0,
        mixup=False,
        repetitions_dataset=repetitions_dataset,
        num_threads=0,  ## somehow necessary for optuna
        save_file=f"model_weights/completed_trainings{cmdln_args.trial_name}.csv",
        name=f"optuna{cmdln_args.trial_name}",
    )
    args.albu_prob = 0.0
    args.individual_albu_probs = 0.0
    args.clahe = False
    args.randomgamma = False
    args.randombrightness = False
    args.blur = False
    args.elastic = False
    args.optical_distortion = False
    args.grid_distortion = False
    args.grid_shuffle = False
    args.hsv = False
    args.invert = False
    args.cutout = False
    args.shadow = False
    args.fog = False
    args.sun_flare = False
    args.solarize = False
    args.equalize = False
    args.grid_dropout = False

    args.differentially_private = True
    args.target_delta = 1e-5
    args.noise_multiplier = trial.suggest_float("DP_noise", 0.1, 1.0)
    args.max_grad_norm = trial.suggest_float("max_grad_norm", 0.1, 2.0)

    if cmdln_args.federated:
        args.unencrypted_aggregation = cmdln_args.unencrypted_aggregation
        args.sync_every_n_batch = trial.suggest_int("sigma", 1, 5)
        args.wait_interval = 0.1
        args.keep_optim_dict = False
        if not cmdln_args.unencrypted_aggregation:
            args.precision_fractional = 16
        # trial.suggest_categorical(
        #     "keep_optim_dict", [True, False]
        # )
        args.weighted_averaging = trial.suggest_categorical(
            "weighted_averaging", [True, False]
        )
        args.DPSSE = False
        args.dpsse_eps = 1.0
        args.microbatch_size = args.batch_size
    try:
        best_val_acc, epsilon = main(args, verbose=False, optuna_trial=trial)
    except Exception as e:
        print(f"Trial failed with exception {e} and arguments {str(args)}.")
        return 0
    if epsilon < 1:
        warn(f"Epsilon is only {epsilon:.2f}. Seems very low.")
    return best_val_acc / max(epsilon, 1)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--federated", action="store_true", help="Search on federated setting"
    )
    parser.add_argument("--data_dir", type=str, help="Path to data")
    parser.add_argument("--trial_name", type=str, default="", help="Assign identifier")
    parser.add_argument("--websockets", action="store_true", help="Use websockets")
    parser.add_argument(
        "--num_trials", default=40, type=int, help="how many trials to perform"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Visualize optuna results."
    )
    parser.add_argument(
        "--db_file",
        type=str,
        default="sqlite:///model_weights/pneumonia_search_DP.db",
        help="Database file to store results.",
    )
    parser.add_argument(
        "--unencrypted_aggregation",
        action="store_true",
        help="Train model without secure aggregation",
    )
    cmdln_args = parser.parse_args()
    try:
        study = opt.create_study(
            study_name="federated_pneumonia{:s}".format(
                "_unencrypted" if cmdln_args.unencrypted_aggregation else ""
            )
            if cmdln_args.federated
            else "vanilla_pneumonia",
            storage=cmdln_args.db_file,
            load_if_exists=True,
            direction="maximize",
            pruner=opt.pruners.NopPruner()
            # pruner=opt.pruners.PercentilePruner(
            #     0.95, n_startup_trials=10, n_warmup_steps=10
            # )
            # pruner=opt.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10),
        )
    except OperationalError:
        print(
            "Error: SQLite cannot find the specified database file. Please make sure the path exists!"
        )
        exit(0)

    if cmdln_args.visualize:

        vis = opt.visualization.plot_param_importances(study)
        vis.show()
        vis = opt.visualization.plot_slice(study)
        vis.show()
        # vis = opt.visualization.plot_contour(study)
        # vis.show()
        vis = opt.visualization.plot_edf(study)
        vis.show()
        vis = opt.visualization.plot_optimization_history(study)
        vis.show()
        vis = opt.visualization.plot_parallel_coordinate(study)
        vis.show()
        vis = opt.visualization.plot_intermediate_values(study)
        vis.show()

    else:
        study.optimize(
            objective,
            n_trials=cmdln_args.num_trials,
            catch=(Exception,),
            n_jobs=1,
            gc_after_trial=True,
        )
