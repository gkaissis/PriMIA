import argparse
import configparser
import multiprocessing as mp
import random
import shutil
from datetime import datetime
from os import path, remove
from warnings import warn

import numpy as np
import syft as sy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import visdom
from tabulate import tabulate
from torchvision import datasets, models, transforms
from optuna import TrialPruned
from torchlib.dataloader import (
    LabelMNIST,
    calc_mean_std,
    single_channel_loader,
)  # pylint:disable=import-error
from torchlib.models import (
    conv_at_resolution,  # pylint:disable=import-error
    resnet18,
    vgg16,
)
from torchlib.utils import (
    AddGaussianNoise,  # pylint:disable=import-error
    Arguments,
    Cross_entropy_one_hot,
    LearningRateScheduler,
    MixUp,
    To_one_hot,
    save_config_results,
    save_model,
    test,
    train,
    train_federated,
)


def calc_class_weights(args, train_loader):
    num_classes = {"pneumonia": 3, "mnist": 10}[args.dataset]
    comparison = list(
        torch.split(
            torch.zeros((num_classes, args.batch_size)), 1  # pylint:disable=no-member
        )
    )
    for i in range(num_classes):
        comparison[i] += i
    occurances = torch.zeros(num_classes)  # pylint:disable=no-member
    if not args.train_federated:
        train_loader = {0: train_loader}
    for worker, tl in tqdm.tqdm(
        train_loader.items(),
        total=len(train_loader),
        leave=False,
        desc="calc class weights",
    ):
        if args.train_federated:
            for i in range(num_classes):
                comparison[i] = comparison[i].send(worker.id)
        for _, target in tqdm.tqdm(
            tl,
            leave=False,
            desc="calc class weights on {:s}".format(worker.id)
            if args.train_federated
            else "calc_class_weights",
            total=len(tl),
        ):
            if target.shape[0] < args.batch_size:
                # the last batch is not considered
                # TODO: find solution for this
                continue
            if args.train_federated and (args.mixup or args.weight_classes):
                if args.mixup and args.mixup_lambda == 0.5:
                    raise ValueError(
                        "it's currently not supported to weight classes "
                        "while mixup has a lambda of 0.5"
                    )
                target = target.max(dim=1)
                target = target[1]  # without pysyft it should be target.indices
            for i in range(num_classes):
                n = target.eq(comparison[i]).sum().copy()
                if args.train_federated:
                    n = n.get()
                occurances[i] += n.item()
        if args.train_federated:
            for i in range(num_classes):
                comparison[i] = comparison[i].get()
    if torch.sum(occurances).item() == 0:  # pylint:disable=no-member
        warn("class weights could not be calculated - no weights are used")
        return torch.ones((num_classes,))  # pylint:disable=no-member
    cw = 1.0 / occurances
    cw /= torch.sum(cw)  # pylint:disable=no-member
    return cw


def setup_pysyft(args, hook, verbose=False):
    from torchlib.websocket_utils import (  # pylint:disable=import-error
        read_websocket_config,
    )

    worker_dict = read_websocket_config("configs/websetting/config.csv")
    worker_names = [id_dict["id"] for _, id_dict in worker_dict.items()]
    """if "validation" in worker_names:
        worker_names.remove("validation")"""

    crypto_in_config = "crypto_provider" in worker_names
    crypto_provider = None
    assert (not args.secure_aggregation) or (
        crypto_in_config
    ), "No crypto provider in configuration"
    if crypto_in_config:
        worker_names.remove("crypto_provider")
        cp_key = [
            key
            for key, worker in worker_dict.items()
            if worker["id"] == "crypto_provider"
        ]
        assert len(cp_key) == 1
        cp_key = cp_key[0]
        if args.secure_aggregation:
            crypto_provider_data = worker_dict[cp_key]
        worker_dict.pop(cp_key)

    if args.websockets:
        if args.weight_classes or (args.mixup and args.dataset == "mnist"):
            raise NotImplementedError(
                "weighted loss as well as mixup in combination"
                " with mnist are not implemented currently"
            )
        workers = {
            worker["id"]: sy.workers.node_client.NodeClient(
                hook,
                "http://{:s}:{:s}".format(worker["host"], worker["port"]),
                id=worker["id"],
                verbose=verbose,
            )
            for _, worker in worker_dict.items()
        }
        if args.secure_aggregation:
            crypto_provider = sy.workers.node_client.NodeClient(
                hook,
                "http://{:s}:{:s}".format(
                    crypto_provider_data["host"], crypto_provider_data["port"]
                ),
                id=crypto_provider_data["id"],
                verbose=verbose,
            )

    else:
        workers = {
            worker["id"]: sy.VirtualWorker(hook, id=worker["id"], verbose=verbose)
            for _, worker in worker_dict.items()
        }
        if args.secure_aggregation:
            crypto_provider = sy.VirtualWorker(
                hook, id="crypto_provider", verbose=verbose
            )
        train_loader = None
        for i, worker in tqdm.tqdm(
            enumerate(workers.values()),
            total=len(workers.keys()),
            leave=False,
            desc="load data",
        ):
            if args.dataset == "mnist":
                node_id = worker.id
                KEEP_LABELS_DICT = {
                    "alice": [0, 1, 2, 3],
                    "bob": [4, 5, 6],
                    "charlie": [7, 8, 9],
                    None: list(range(10)),
                }
                dataset = LabelMNIST(
                    labels=KEEP_LABELS_DICT[node_id]
                    if node_id in KEEP_LABELS_DICT
                    else KEEP_LABELS_DICT[None],
                    root="./data",
                    train=True,
                    download=True,
                    transform=transforms.Compose(
                        [
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,)),
                        ]
                    ),
                )
            elif args.dataset == "pneumonia":
                data_dir = path.join(
                    "data/server_simulation/", "worker{:d}".format(i + 1)
                )
                stats_dataset = datasets.ImageFolder(
                    data_dir,
                    transform=transforms.Compose(
                        [
                            transforms.Resize(args.train_resolution),
                            transforms.CenterCrop(args.train_resolution),
                            transforms.ToTensor(),
                        ]
                    ),
                )
                mean, std = calc_mean_std(stats_dataset, save_folder=data_dir,)
                del stats_dataset
                train_tf = [
                    transforms.RandomVerticalFlip(p=args.vertical_flip_prob),
                    transforms.RandomAffine(
                        degrees=args.rotation,
                        translate=(args.translate, args.translate),
                        scale=(1.0 - args.scale, 1.0 + args.scale),
                        shear=args.shear,
                        #    fillcolor=0,
                    ),
                    transforms.Resize(args.inference_resolution),
                    transforms.RandomCrop(args.train_resolution),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                    AddGaussianNoise(mean=0.0, std=args.noise_std, p=args.noise_prob),
                ]
                target_dict_pneumonia = {0: 1, 1: 0, 2: 2}
                target_tf = [lambda x: target_dict_pneumonia[x]]
                if args.mixup or args.weight_classes:
                    target_tf.append(
                        lambda x: torch.tensor(x)  # pylint:disable=not-callable
                    )
                    target_tf.append(To_one_hot(3))
                dataset = datasets.ImageFolder(
                    # path.join("data/server_simulation/", "validation")
                    # if worker.id == "validation"
                    # else
                    data_dir,
                    transform=transforms.Compose(train_tf),
                    target_transform=transforms.Compose(target_tf),
                )
                mean.tag("#datamean")
                std.tag("#datastd")
                worker.load_data([mean, std])

            else:
                raise NotImplementedError(
                    "federation for virtual workers for this dataset unknown"
                )
            data, targets = [], []
            # repetitions = 1 if worker.id == "validation" else args.repetitions_dataset
            if args.mixup:
                dataset = torch.utils.data.DataLoader(
                    dataset, batch_size=1, shuffle=True
                )
                mixup = MixUp(λ=args.mixup_lambda, p=args.mixup_prob)
                last_set = None
            for j in tqdm.tqdm(
                range(args.repetitions_dataset),
                total=args.repetitions_dataset,
                leave=False,
                desc="register data on {:s}".format(worker.id),
            ):
                for d, t in tqdm.tqdm(
                    dataset,
                    total=len(dataset),
                    leave=False,
                    desc="register data {:d}. time".format(j + 1),
                ):
                    if args.mixup:
                        original_set = (d, t)
                        if last_set:
                            # pylint:disable=unsubscriptable-object
                            d, t = mixup(((d, last_set[0]), (t, last_set[1])))
                        last_set = original_set
                    data.append(d)
                    targets.append(t)
            selected_data = torch.stack(data)  # pylint:disable=no-member
            selected_targets = (
                torch.stack(targets)  # pylint:disable=no-member
                if args.mixup or args.weight_classes
                else torch.tensor(targets)  # pylint:disable=not-callable
            )
            if args.mixup:
                selected_data = selected_data.squeeze(1)
                selected_targets = selected_targets.squeeze(1)
            del data, targets
            selected_data.tag(
                args.dataset,  # "#valdata" if worker.id == "validation" else
                "#traindata",
            )
            selected_targets.tag(
                args.dataset,
                # "#valtargets" if worker.id == "validation" else
                "#traintargets",
            )
            worker.load_data([selected_data, selected_targets])

    grid: sy.PrivateGridNetwork = sy.PrivateGridNetwork(*workers.values())
    data = grid.search(args.dataset, "#traindata")
    target = grid.search(args.dataset, "#traintargets")
    train_loader = {}
    total_L = 0
    for worker in data.keys():
        dist_dataset = [  # TODO: in the future transform here would be nice but currently raise errors
            sy.BaseDataset(
                data[worker][0], target[worker][0],
            )  # transform=federated_tf
        ]
        fed_dataset = sy.FederatedDataset(dist_dataset)
        total_L += len(fed_dataset)
        tl = sy.FederatedDataLoader(
            fed_dataset, batch_size=args.batch_size, shuffle=True
        )
        train_loader[workers[worker]] = tl

    # data = grid.search(args.dataset, "#valdata")
    # target = grid.search(args.dataset, "#valtargets")
    if args.dataset == "mnist":
        valset = LabelMNIST(
            labels=list(range(10)),
            root="./data",
            train=False,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)),]
            ),
        )
    elif args.dataset == "pneumonia":
        means = [m[0] for m in grid.search("#datamean").values()]
        stds = [s[0] for s in grid.search("#datastd").values()]
        if len(means) > 0 and len(stds) > 0 and len(means) == len(stds):
            mean = (
                means[0]
                .fix_precision()
                .share(*workers, crypto_provider=crypto_provider)
                .get()
            )
            std = (
                stds[0]
                .fix_precision()
                .share(*workers, crypto_provider=crypto_provider)
                .get()
            )
            for m, s in zip(means[1:], stds[1:]):
                mean += (
                    m.fix_precision()
                    .share(*workers, crypto_provider=crypto_provider)
                    .get()
                )
                std += (
                    s.fix_precision()
                    .share(*workers, crypto_provider=crypto_provider)
                    .get()
                )
            mean = mean.get().float_precision() / len(stds)
            std = std.get().float_precision() / len(stds)
        else:
            ## default values
            mean, std = 0.5, 0.2
        val_mean_std = torch.stack([mean, std])  # pylint:disable=no-member
        val_tf = [
            transforms.Resize(args.inference_resolution),
            transforms.CenterCrop(args.train_resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
        target_dict_pneumonia = {0: 1, 1: 0, 2: 2}
        valset = datasets.ImageFolder(
            path.join("data/server_simulation/", "validation"),
            transform=transforms.Compose(val_tf),
            target_transform=lambda x: target_dict_pneumonia[x],
        )
    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=args.test_batch_size, shuffle=False
    )
    assert len(train_loader.keys()) == (
        len(workers.keys())
    ), "data was not correctly loaded"

    print(
        "Found a total dataset with {:d} samples on remote workers".format(
            sum([len(dl.federated_dataset) for dl in train_loader.values()])
        )
    )
    print(
        "Found a total validation set with {:d} samples (locally)".format(
            len(val_loader.dataset)
        )
    )
    return (
        train_loader,
        val_loader,
        total_L,
        workers,
        worker_names,
        crypto_provider,
        val_mean_std,
    )


def main(args, verbose=True, optuna_trial=None):

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if use_cuda else "cpu")  # pylint: disable=no-member

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_name = "{:s}_{:s}_{:s}".format(
        "federated" if args.train_federated else "vanilla", args.dataset, timestamp
    )

    # Dataset creation and definition
    dataset_classes = {"mnist": 10, "pneumonia": 3}
    num_classes = dataset_classes[args.dataset]
    class_name_dict = {
        "pneumonia": {0: "normal", 1: "bacterial pneumonia", 2: "viral pneumonia"}
    }
    class_names = (
        class_name_dict[args.dataset]
        if args.dataset in class_name_dict.keys()
        else None
    )
    if args.train_federated:

        hook = sy.TorchHook(torch)
        (
            train_loader,
            val_loader,
            total_L,
            workers,
            worker_names,
            crypto_provider,
            val_mean_std,
        ) = setup_pysyft(
            args,
            hook,
            verbose=cmd_args.verbose
            if "cmd_args" in locals() and hasattr(cmd_args, "verbose")
            else False,
        )
    else:
        if args.dataset == "mnist":
            train_tf = [
                transforms.Resize(args.train_resolution),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
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

        elif args.dataset == "pneumonia":
            # Different train and inference resolution only works with adaptive
            # pooling in model activated
            train_tf = [
                transforms.RandomVerticalFlip(p=args.vertical_flip_prob),
                transforms.RandomAffine(
                    degrees=args.rotation,
                    translate=(args.translate, args.translate),
                    scale=(1.0 - args.scale, 1.0 + args.scale),
                    shear=args.shear,
                    #    fillcolor=0,
                ),
                transforms.Resize(args.inference_resolution),
                transforms.RandomCrop(args.train_resolution),
                transforms.ToTensor(),
            ]
            # dataset = PPPP(
            #     "data/Labels.csv",
            target_dict_pneumonia = {0: 1, 1: 0, 2: 2}
            target_tf = lambda x: target_dict_pneumonia[x]
            dataset = datasets.ImageFolder(
                "data/server_simulation/train_total",
                transform=transforms.Compose(train_tf),
                target_transform=target_tf,
                loader=datasets.folder.default_loader
                if args.pretrained
                else single_channel_loader,
            )
            val_mean_std = calc_mean_std(dataset)
            mean, std = val_mean_std
            train_tf = [
                transforms.Normalize(tuple(mean), tuple(std)),
                AddGaussianNoise(mean=0.0, std=args.noise_std, p=args.noise_prob),
            ]

            dataset.transform.transforms.extend(train_tf)

            # occurances = dataset.get_class_occurances()

        else:
            raise NotImplementedError("dataset not implemented")

        total_L = total_L if args.train_federated else len(dataset)
        fraction = 1.0 / args.validation_split
        dataset, valset = torch.utils.data.random_split(
            dataset,
            [int(round(total_L * (1.0 - fraction))), int(round(total_L * fraction))],
        )
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, **kwargs
        )
        val_loader = torch.utils.data.DataLoader(
            valset, batch_size=args.test_batch_size, shuffle=False, **kwargs,
        )
        del total_L, fraction

    cw = None
    if args.weight_classes:
        if "occurances" in locals():
            cw = torch.zeros((len(occurances)))  # pylint: disable=no-member
            for c, n in occurances.items():
                cw[c] = 1.0 / float(n)
            cw /= torch.sum(cw)  # pylint: disable=no-member
        else:
            cw = calc_class_weights(args, train_loader)
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
        ), "No connection could be formed quickly"
        vis_env = path.join(
            args.dataset, "federated" if args.train_federated else "vanilla", timestamp
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
                "legend": ["train_loss", "val_loss", "ROC AUC"],
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
            "in_channels": 3 if args.dataset == "pneumonia" else 1,
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
            "in_channels": 3 if args.dataset == "pneumonia" else 1,
            "pooling": args.pooling_type,
        }
    elif args.model == "resnet-18":
        model_type = resnet18
        model_args = {
            "pretrained": args.pretrained,
            "num_classes": num_classes,
            "in_channels": 3 if args.dataset == "pneumonia" else 1,
            "adptpool": False,
            "input_size": args.inference_resolution,
            "pooling": args.pooling_type,
        }
    else:
        raise NotImplementedError("model unknown")
    if args.train_federated:
        model = model_type(**model_args)
        model = {
            key: model.copy()
            for key in [w.id for w in workers.values()] + ["local_model"]
        }
    else:
        model = model_type(**model_args)

    if args.optimizer == "SGD":
        optimizer = optim.SGD(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        )  # TODO momentum is not supported at the moment
    elif args.optimizer == "Adam":
        optimizer = (
            {
                idt: optim.Adam(
                    m.parameters(),
                    lr=args.lr,
                    betas=(args.beta1, args.beta2),
                    weight_decay=args.weight_decay,
                )
                for idt, m in model.items()
                if idt not in ["local_model", "crypto_provider"]
            }
            if args.train_federated
            else optim.Adam(
                model.parameters(),
                lr=args.lr,
                betas=(args.beta1, args.beta2),
                weight_decay=args.weight_decay,
            )
        )
    else:
        raise NotImplementedError("optimization not implemented")
        # if args.train_federated and not args.secure_aggregation:
        #     from syft.federated.floptimizer import Optims

        # optimizer = Optims(worker_names, optimizer)
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
    roc_auc_scores = []
    model_paths = []
    if args.train_federated:
        test_params = {
            "device": device,
            "val_loader": val_loader,
            "loss_fn": loss_fn,
            "num_classes": num_classes,
            "class_names": class_names,
            "exp_name": exp_name,
            "optimizer": optimizer,
            "roc_auc_scores": roc_auc_scores,
            "model_paths": model_paths,
        }
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

        try:
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
                    test_params=test_params,
                    vis_params=vis_params,
                    verbose=verbose,
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
        except Exception as e:
            if args.websockets:
                warn("An exception occured - restarting websockets")
                try:
                    (
                        train_loader,
                        val_loader,
                        total_L,
                        workers,
                        worker_names,
                        crypto_provider,
                        val_mean_std,
                    ) = setup_pysyft(args, hook, verbose=cmd_args.verbose)
                except Exception as e:
                    print("restarting failed")
                    raise e
            else:
                raise e

        if (epoch % args.test_interval) == 0:
            _, roc_auc = test(
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
                    roc_auc,
                    epoch
                    * (args.repetitions_dataset if args.repetitions_dataset else 1),
                )
                if optuna_trial.should_prune():
                    raise TrialPruned()

            save_model(model, optimizer, model_path, args, epoch, val_mean_std)
            roc_auc_scores.append(roc_auc)
            model_paths.append(model_path)
    # reversal and formula because we want last occurance of highest value
    roc_auc_scores = np.array(roc_auc_scores)[::-1]
    best_auc_idx = np.argmax(roc_auc_scores)
    highest_acc = len(roc_auc_scores) - best_auc_idx - 1
    best_epoch = (
        highest_acc + 1
    ) * args.test_interval  # actually -1 but we're switching to 1 indexed here
    best_model_file = model_paths[highest_acc]
    print(
        "Highest ROC AUC score was {:.1f}% in epoch {:d}".format(
            roc_auc_scores[best_auc_idx],
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
    save_config_results(
        args,
        roc_auc_scores[best_auc_idx],
        timestamp,
        "model_weights/completed_trainings.csv",
    )

    # delete old model weights
    for model_file in model_paths:
        remove(model_file)

    return roc_auc_scores[best_auc_idx]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config",
    )
    parser.add_argument(
        "--train_federated", action="store_true", help="Train in federated setting"
    )
    parser.add_argument(
        "--secure_aggregation", action="store_true", help="Perform secure aggregation"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="pneumonia",
        choices=["pneumonia", "mnist"],
        required=True,
        help="which dataset?",
    )
    parser.add_argument(
        "--no_visdom", action="store_false", help="dont use a visdom server"
    )
    parser.add_argument("--no_cuda", action="store_true", help="dont use gpu")
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Start training from older model checkpoint",
    )
    parser.add_argument(
        "--websockets", action="store_true", help="train on websocket config"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="set syft workers to verbose"
    )
    cmd_args = parser.parse_args()

    config = configparser.ConfigParser()
    assert path.isfile(cmd_args.config), "config file not found"
    config.read(cmd_args.config)

    args = Arguments(cmd_args, config, mode="train")
    if args.websockets:
        assert args.train_federated, "Websockets only work when it is federated"
    print(str(args))
    main(args)
