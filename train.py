from os import path, remove, environ

environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import argparse
import configparser
import random
import shutil
from datetime import datetime
from warnings import warn

import numpy as np
import syft as sy
import torch

# torch.set_num_threads(36)

import torch.nn as nn
import torch.optim as optim
import torchdp as tdp
import tqdm
import visdom
import albumentations as a
from tabulate import tabulate
from torchvision import datasets, transforms
from optuna import TrialPruned
from math import ceil, floor
from torchlib.dataloader import (
    calc_mean_std,
    AlbumentationsTorchTransform,
    random_split,
    create_albu_transform,
    CombinedLoader,
)  # pylint:disable=import-error
from torchlib.models import (
    conv_at_resolution,  # pylint:disable=import-error
    resnet18,
    vgg16,
)
from torchlib.utils import (
    Arguments,
    Cross_entropy_one_hot,
    LearningRateScheduler,
    MixUp,
    save_config_results,
    save_model,
    test,
    train,
    train_federated,
    setup_pysyft,
    calc_class_weights,
)


def main(args, verbose=True, optuna_trial=None):

    use_cuda = args.cuda and torch.cuda.is_available()
    if args.deterministic and args.websockets:
        warn(
            "Training with GridNodes is not compatible with deterministic training.\n"
            "Switching deterministic flag to False"
        )
        args.deterministic = False
    if args.deterministic:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if use_cuda else "cpu")  # pylint: disable=no-member

    kwargs = {"num_workers": args.num_threads, "pin_memory": True,} if use_cuda else {}

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_name = "{:s}_{:s}_{:s}".format(
        "federated" if args.train_federated else "vanilla",
        args.data_dir.replace("/", ""),
        timestamp,
    )
    num_classes = 10 if args.data_dir == "mnist" else 3
    class_names = None
    # Dataset creation and definition
    if args.train_federated:
        if hasattr(torch, "torch_hooked"):
            hook = sy.hook
        else:
            hook = sy.TorchHook(torch)

        (
            train_loader,
            val_loader,
            total_L,
            workers,
            worker_names,
            crypto_provider,
            val_mean_std,
        ) = setup_pysyft(args, hook, verbose=verbose,)
    else:
        if args.data_dir == "mnist":
            val_mean_std = torch.tensor(  # pylint:disable=not-callable
                [[0.1307], [0.3081]]
            )
            mean, std = val_mean_std
            if args.pretrained:
                mean, std = mean[None, None, :], std[None, None, :]
            train_tf = [
                transforms.Resize(args.train_resolution),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
            if args.pretrained:
                repeat = transforms.Lambda(
                    lambda x: torch.repeat_interleave(  # pylint: disable=no-member
                        x, 3, dim=0
                    )
                )
                train_tf.append(repeat)
            dataset = datasets.MNIST(
                "../data",
                train=True,
                download=True,
                transform=transforms.Compose(train_tf),
            )
            total_L = len(dataset)
            fraction = 1.0 / args.validation_split
            dataset, valset = random_split(
                dataset,
                [int(ceil(total_L * (1.0 - fraction))), int(floor(total_L * fraction))],
            )
        else:
            # Different train and inference resolution only works with adaptive
            # pooling in model activated
            stats_tf = AlbumentationsTorchTransform(
                a.Compose(
                    [
                        a.Resize(args.inference_resolution, args.inference_resolution),
                        a.RandomCrop(args.train_resolution, args.train_resolution),
                        a.ToFloat(max_value=255.0),
                    ]
                )
            )
            # dataset = PPPP(
            #     "data/Labels.csv",
            loader = CombinedLoader()
            if not args.pretrained:
                loader.change_channels(1)
            dataset = datasets.ImageFolder(
                args.data_dir, transform=stats_tf, loader=loader,
            )
            assert (
                len(dataset.classes) == 3
            ), "Dataset must have exactly 3 classes: normal, bacterial and viral"
            val_mean_std = calc_mean_std(dataset)
            mean, std = val_mean_std
            if args.pretrained:
                mean, std = mean[None, None, :], std[None, None, :]
            dataset.transform = create_albu_transform(args, mean, std)
            class_names = dataset.classes
            stats_tf.transform.transforms.transforms.append(
                a.Normalize(mean, std, max_pixel_value=1.0)
            )
            valset = datasets.ImageFolder(
                "data/test", transform=stats_tf, loader=loader
            )
            # occurances = dataset.get_class_occurances()

        # total_L = total_L if args.train_federated else len(dataset)
        # fraction = 1.0 / args.validation_split
        # dataset, valset = random_split(
        #     dataset,
        #     [int(ceil(total_L * (1.0 - fraction))), int(floor(total_L * fraction))],
        # )
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, **kwargs
        )

        # val_tf = [
        #     a.Resize(args.inference_resolution, args.inference_resolution),
        #     a.CenterCrop(args.inference_resolution, args.inference_resolution),
        #     a.ToFloat(max_value=255.0),
        #     a.Normalize(mean, std, max_pixel_value=1.0),
        # ]
        # if not args.pretrained:
        #     val_tf.append(a.Lambda(image=lambda x, **kwargs: x[:, :, np.newaxis]))
        # valset.dataset.transform = AlbumentationsTorchTransform(a.Compose(val_tf))

        val_loader = torch.utils.data.DataLoader(
            valset, batch_size=args.test_batch_size, shuffle=False, **kwargs,
        )
        # del total_L, fraction

    cw = None
    if args.weight_classes:
        cw = calc_class_weights(args, train_loader, num_classes)
        cw = cw.to(device)

    scheduler = LearningRateScheduler(
        args.epochs, np.log10(args.lr), np.log10(args.end_lr), restarts=args.restarts
    )

    ## visdom
    vis_params = None
    if args.visdom:
        vis = visdom.Visdom()
        assert vis.check_connection(
            timeout_seconds=3
        ), "Connection to the visdom server could not be established!"
        vis_env = path.join(
            "federated" if args.train_federated else "vanilla", timestamp
        )
        plt_dict = dict(
            name="training loss",
            ytickmax=10,
            xlabel="epoch",
            ylabel="loss",
            legend=["train_loss"],
        )
        vis.line(
            X=np.zeros((1, 3)),
            Y=np.zeros((1, 3)),
            win="loss_win",
            opts={
                "legend": ["train_loss", "val_loss", "matthews coeff"],
                "xlabel": "epochs",
                "ylabel": "loss / m coeff [%]",
            },
            env=vis_env,
        )
        vis.line(
            X=np.zeros((1, 1)),
            Y=np.zeros((1, 1)),
            win="lr_win",
            opts={"legend": ["learning_rate"], "xlabel": "epochs", "ylabel": "lr"},
            env=vis_env,
        )
        vis_params = {"vis": vis, "vis_env": vis_env}
    if args.model == "vgg16":
        model_type = vgg16
        model_args = {
            "pretrained": args.pretrained,
            "num_classes": num_classes,
            "in_channels": 1 if args.data_dir == "mnist" or not args.pretrained else 3,
            "adptpool": False,
            "input_size": args.inference_resolution,
            "pooling": args.pooling_type,
        }
    elif args.model == "simpleconv":
        if args.pretrained:
            warn("No pretrained version available")

        model_type = conv_at_resolution[args.train_resolution]
        model_args = {
            "num_classes": num_classes,
            "in_channels": 1 if args.data_dir == "mnist" or not args.pretrained else 3,
            "pooling": args.pooling_type,
        }
    elif args.model == "resnet-18":
        model_type = resnet18
        model_args = {
            "pretrained": args.pretrained,
            "num_classes": num_classes,
            "in_channels": 1 if args.data_dir == "mnist" or not args.pretrained else 3,
            "adptpool": False,
            "input_size": args.inference_resolution,
            "pooling": args.pooling_type,
        }
    else:
        raise ValueError(
            "Model name not understood. Please choose one of 'vgg16, 'simpleconv', resnet-18'."
        )
    if args.train_federated:
        model = model_type(**model_args)
        model = {
            key: model.copy()
            for key in [w.id for w in workers.values()] + ["local_model"]
        }
    else:
        model = model_type(**model_args)

    opt_kwargs = {"lr": args.lr, "weight_decay": args.weight_decay}
    if args.optimizer == "SGD":
        opt = optim.SGD
    elif args.optimizer == "Adam":
        opt = optim.Adam
        opt_kwargs["betas"] = (args.beta1, args.beta2)
    else:
        raise ValueError(
            "Optimizer name not understood. Please use one of 'SGD' or 'Adam'."
        )
        # if args.train_federated and not args.secure_aggregation:
        #     from syft.federated.floptimizer import Optims

        # optimizer = Optims(worker_names, optimizer)

    optimizer = (
        {
            idt: opt(m.parameters(), **opt_kwargs)
            for idt, m in model.items()
            if idt not in ["local_model", "crypto_provider"]
        }
        if args.train_federated
        else opt(model.parameters(), **opt_kwargs)
    )
    privacy_engines = None
    if args.differentially_private:
        if type(optimizer) == dict:
            warn(
                "Differential Privacy is currently only implemented for local training and models without BatchNorm."
            )
            exit()
            privacy_engines = {
                idt.id: tdp.PrivacyEngine(
                    model[idt.id],
                    args.batch_size,
                    len(tl.federated_dataset),
                    alphas=[1, 10, 100],
                    noise_multiplier=1.3,
                    max_grad_norm=1.0,
                )
                for idt, tl in train_loader.items()
                if idt.id not in ["local_model", "crypto_provider"]
            }
            for w, pe in privacy_engines.items():
                pe.attach(optimizer[w])
        else:
            privacy_engine = tdp.PrivacyEngine(
                model,
                args.batch_size,
                len(train_loader.dataset),
                alphas=[1, 10, 100],
                noise_multiplier=1.3,
                max_grad_norm=1.0,
            )
            privacy_engine.attach(optimizer)
    loss_args = {"weight": cw, "reduction": "mean"}
    if args.mixup or (args.weight_classes and args.train_federated):
        loss_fn = Cross_entropy_one_hot
    else:
        loss_fn = nn.CrossEntropyLoss
    loss_fn = loss_fn(**loss_args).to(device)
    if args.train_federated:
        loss_fn = {w: loss_fn.copy() for w in [*workers, "local_model"]}

    start_at_epoch = 1
    if "cmd_args" in locals() and cmd_args.resume_checkpoint:
        print("resuming checkpoint - args will be overwritten")
        state = torch.load(cmd_args.resume_checkpoint, map_location=device)
        start_at_epoch = state["epoch"]
        args = state["args"]
        if cmd_args.train_federated and args.train_federated:
            opt_state_dict = state["optim_state_dict"]
            for w in worker_names:
                optimizer.get_optim(w).load_state_dict(opt_state_dict[w])
        elif not cmd_args.train_federated and not args.train_federated:
            optimizer.load_state_dict(state["optim_state_dict"])
        else:
            pass  # not possible to load previous optimizer if setting changed
        args.incorporate_cmd_args(cmd_args)
        model.load_state_dict(state["model_state_dict"])
    if args.train_federated:
        for m in model.values():
            m.to(device)
    else:
        model.to(device)

    test(
        args,
        model["local_model"] if args.train_federated else model,
        device,
        val_loader,
        start_at_epoch - 1,
        loss_fn["local_model"] if args.train_federated else loss_fn,
        num_classes,
        vis_params=vis_params,
        class_names=class_names,
        verbose=verbose,
    )
    matthews_scores = []
    model_paths = []
    """if args.train_federated:
        test_params = {
            "device": device,
            "val_loader": val_loader,
            "loss_fn": loss_fn,
            "num_classes": num_classes,
            "class_names": class_names,
            "exp_name": exp_name,
            "optimizer": optimizer,
            "matthews_scores": matthews_scores,
            "model_paths": model_paths,
        }"""
    for epoch in (
        range(start_at_epoch, args.epochs + 1)
        if verbose
        else tqdm.tqdm(
            range(start_at_epoch, args.epochs + 1),
            leave=False,
            desc="training",
            total=args.epochs + 1,
            initial=start_at_epoch,
        )
    ):
        if args.train_federated:
            for w in worker_names:
                new_lr = scheduler.adjust_learning_rate(
                    optimizer[
                        w
                    ],  # if args.secure_aggregation else optimizer.get_optim(w),
                    epoch - 1,
                )
        else:
            new_lr = scheduler.adjust_learning_rate(optimizer, epoch - 1)
        if args.visdom:
            vis.line(
                X=np.asarray([epoch - 1]),
                Y=np.asarray([new_lr]),
                win="lr_win",
                name="learning_rate",
                update="append",
                env=vis_env,
            )

        if args.train_federated:
            model = train_federated(
                args,
                model,
                device,
                train_loader,
                optimizer,
                epoch,
                loss_fn,
                crypto_provider,
                # In future test_params could be changed if testing
                # during epoch should be enabled
                test_params=None,
                vis_params=vis_params,
                verbose=verbose,
                privacy_engines=privacy_engines,
            )

        else:
            model = train(
                args,
                model,
                device,
                train_loader,
                optimizer,
                epoch,
                loss_fn,
                num_classes,
                vis_params=vis_params,
                verbose=verbose,
            )
        # except Exception as e:

        if (epoch % args.test_interval) == 0:
            _, matthews = test(
                args,
                model["local_model"] if args.train_federated else model,
                device,
                val_loader,
                epoch,
                loss_fn["local_model"] if args.train_federated else loss_fn,
                num_classes=num_classes,
                vis_params=vis_params,
                class_names=class_names,
                verbose=verbose,
            )
            model_path = "model_weights/{:s}_epoch_{:03d}.pt".format(
                exp_name,
                epoch
                * (
                    args.repetitions_dataset
                    if "repetitions_dataset" in vars(args)
                    else 1
                ),
            )
            if optuna_trial:
                optuna_trial.report(
                    matthews,
                    epoch
                    * (args.repetitions_dataset if args.repetitions_dataset else 1),
                )
                if optuna_trial.should_prune():
                    raise TrialPruned()

            save_model(model, optimizer, model_path, args, epoch, val_mean_std)
            matthews_scores.append(matthews)
            model_paths.append(model_path)
    # reversal and formula because we want last occurance of highest value
    matthews_scores = np.array(matthews_scores)[::-1]
    best_score_idx = np.argmax(matthews_scores)
    highest_score = len(matthews_scores) - best_score_idx - 1
    best_epoch = (
        highest_score + 1
    ) * args.test_interval  # actually -1 but we're switching to 1 indexed here
    best_model_file = model_paths[highest_score]
    print(
        "Highest matthews coefficient was {:.1f}% in epoch {:d}".format(
            matthews_scores[best_score_idx],
            best_epoch * (args.repetitions_dataset if args.train_federated else 1),
        )
    )
    # load best model on val set
    state = torch.load(best_model_file, map_location=device)
    if args.train_federated:
        model = model["local_model"]
    model.load_state_dict(state["model_state_dict"])

    shutil.copyfile(
        best_model_file, "model_weights/final_{:s}.pt".format(exp_name),
    )
    if args.save_file:
        save_config_results(
            args, matthews_scores[best_score_idx], timestamp, args.save_file,
        )

    # delete old model weights
    for model_file in model_paths:
        remove(model_file)

    return matthews_scores[best_score_idx]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file (.ini).",
    )
    parser.add_argument(
        "--train_federated", action="store_true", help="Train with federated learning."
    )
    parser.add_argument(
        "--unencrypted_aggregation",
        action="store_true",
        help="Turns off secure aggregation."
        "Slight advantages in terms of model performance and training speed.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        # required=True,
        default="data/train",
        help='Select a data folder [if "mnist" is passed, the torchvision MNIST dataset will be downloaded and used].',
    )
    parser.add_argument(
        "--visdom", action="store_true", help="Use Visdom for monitoring training."
    )
    parser.add_argument("--cuda", action="store_true", help="Use CUDA acceleration.")
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Start training from older model checkpoint",
    )
    parser.add_argument(
        "--websockets", action="store_true", help="Train using WebSockets."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Sets Syft workers to verbose mode"
    )
    parser.add_argument(
        "--save_file",
        type=str,
        default="model_weights/completed_trainings.csv",
        help="Store args and result in csv file.",
    )
    parser.add_argument(
        "--training_name",
        default=None,
        type=str,
        help="Optional name to be stored in csv file to later identify training.",
    )
    cmd_args = parser.parse_args()

    config = configparser.ConfigParser()
    assert path.isfile(cmd_args.config), "Configuration file not found"
    config.read(cmd_args.config)

    args = Arguments(cmd_args, config, mode="train")
    if args.websockets:
        if not args.train_federated:
            raise RuntimeError("WebSockets can only be used when in federated mode.")
    if args.cuda and args.train_federated:
        warn(
            "CUDA is currently not supported by the backend. This option will be available at a later release",
            category=FutureWarning,
        )
        exit(0)
    if args.train_federated and (args.mixup or args.weight_classes):
        if args.mixup and args.mixup_lambda == 0.5:
            warn(
                "Class weighting and a lambda value of 0.5 are incompatible, setting lambda to 0.499",
                category=RuntimeWarning,
            )
            args.mixup_lambda = 0.499
    print(str(args))
    main(args)
