import optuna as opt
from train import main
from argparse import Namespace

global cmdln_args


def objective(trial: opt.trial):
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-3,)
    repetitions_dataset = (
        trial.suggest_int("repetitions_dataset", 1, 2) if cmdln_args.federated else 0
    )
    epochs = int(trial.suggest_int("epochs", 15, 40) // repetitions_dataset)
    args = Namespace(
        config="optuna",
        train_federated=cmdln_args.federated,
        dataset="pneumonia",
        visdom=False,
        encrypted_inference=False,
        no_cuda=cmdln_args.federated,
        websockets=False,
        batch_size=200,
        train_resolution=224,
        inference_resolution=224,
        test_batch_size=1,
        test_interval=1,
        validation_split=10,
        epochs=epochs,
        lr=lr,
        end_lr=trial.suggest_loguniform("end_lr", 1e-6, lr),
        restarts=trial.suggest_int("restarts", 0, 2),
        beta1=trial.suggest_float("beta1", 0.25, 0.9),
        beta2=trial.suggest_float("beta2", 0.9, 1.0),
        weight_decay=trial.suggest_float("weight_decay", 0, 1e-5),
        seed=1,
        log_interval=10,
        optimizer="Adam",
        model="resnet-18",
        pretrained=True,
        weight_classes=trial.suggest_categorical("weight_classes", [True, False]),
        pooling_type="max",
        vertical_flip_prob=trial.suggest_float("vertical_flip_prob", 0.0, 1.0),
        rotation=trial.suggest_int("rotation", 0, 90),
        translate=trial.suggest_float("translate", 0, 0.2),
        scale=trial.suggest_float("scale", 0.0, 0.5),
        shear=trial.suggest_int("shear", 0, 30),
        noise_std=trial.suggest_float("noise_std", 0.0, 0.1),
        noise_prob=trial.suggest_float("noise_prob", 0.0, 1.0),
        mixup=trial.suggest_categorical("mixup", [True, False]),
        mixup_lambda=trial.suggest_categorical("mixup_lambda", (0.1, 0.25, 0.49, None)),
        mixup_prob=trial.suggest_float("mixup_prob", 0.0, 1.0),
        sync_every_n_batch=5,
        wait_interval=0.1,
        repetitions_dataset=repetitions_dataset,
        keep_optim_dict=trial.suggest_categorical("keep_optim_dict", [True, False])
        if cmdln_args.federated
        else False,
        weighted_averaging=trial.suggest_categorical(
            "weighted_averaging", [True, False]
        )
        if cmdln_args.federated
        else False,
    )
    print(args)
    best_val_acc = main(args, verbose=True)
    return best_val_acc


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--federated", action="store_true", help="Search on federated setting"
    )
    cmdln_args = parser.parse_args()
    study = opt.create_study(
        study_name="federated_pneumonia"
        if cmdln_args.federated
        else "vanilla_pneumonia",
        storage="sqlite:///hyperparam_search/pneumonia_search.db",
        load_if_exists=True,
        direction="maximize",
    )
    study.optimize(objective, n_trials=1)
