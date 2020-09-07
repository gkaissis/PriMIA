import multiprocessing as mp
from os import makedirs
from os.path import isfile, split, isdir
from random import random
from time import sleep
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch


import tqdm
from sklearn import metrics as mt
import syft.frameworks.torch.fl.utils as syft_fl_utils
from syft.frameworks.torch.fl.utils import add_model, scale_model
from tabulate import tabulate
from collections import Counter
from copy import deepcopy


class LearningRateScheduler:
    """
    Available schedule plans:
    log_linear : Linear interpolation with log learning rate scale
    log_cosine : Cosine interpolation with log learning rate scale
    """

    def __init__(
        self,
        total_epochs: int,
        log_start_lr: float,
        log_end_lr: float,
        schedule_plan: str = "log_linear",
        restarts: Optional[int] = None,
    ):
        if restarts == 0:
            restarts = None
        self.total_epochs = (
            total_epochs if not restarts else total_epochs / (restarts + 1)
        )
        if schedule_plan == "log_linear":
            self.calc_lr = lambda epoch: np.power(
                10,
                ((log_end_lr - log_start_lr) / self.total_epochs) * epoch
                + log_start_lr,
            )
        elif schedule_plan == "log_cosine":
            self.calc_lr = lambda epoch: np.power(
                10,
                (np.cos(np.pi * (epoch / self.total_epochs)) / 2.0 + 0.5)
                * abs(log_start_lr - log_end_lr)
                + log_end_lr,
            )
        else:
            raise NotImplementedError(
                "Requested learning rate schedule {} not implemented".format(
                    schedule_plan
                )
            )

    def get_lr(self, epoch: int):
        epoch = epoch % self.total_epochs
        if (type(epoch) is int and epoch > self.total_epochs) or (
            type(epoch) is np.ndarray and np.max(epoch) > self.total_epochs
        ):
            raise AssertionError("Requested epoch out of precalculated schedule")
        return self.calc_lr(epoch)

    def adjust_learning_rate(self, optimizer: torch.optim.Optimizer, epoch: int):
        new_lr = self.get_lr(epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr
        return new_lr


class Arguments:
    def __init__(self, cmd_args, config, mode: str = "train", verbose: bool = True):
        assert mode in ["train", "inference"], "no other mode known"
        self.name = (
            cmd_args.training_name
            if hasattr(cmd_args, "training_name") and cmd_args.training_name
            else "default"
        )
        self.save_file = (
            cmd_args.save_file
            if hasattr(cmd_args, "save_file")
            else "model_weights/completed_trainings.csv"
        )
        self.batch_size = config.getint("config", "batch_size")  # , fallback=1)
        self.test_batch_size = config.getint(
            "config", "test_batch_size"
        )  # , fallback=1)
        self.train_resolution = config.getint(
            "config", "train_resolution"
        )  # , fallback=224
        self.inference_resolution = config.getint(
            "config", "inference_resolution", fallback=self.train_resolution
        )
        if self.train_resolution != self.inference_resolution:
            raise FutureWarning(
                "We are not supporting different train and inference"
                " resolutions although it works for some scenarios."
            )
        self.validation_split = config.getint(
            "config", "validation_split"
        )  # , fallback=10)
        self.epochs = config.getint("config", "epochs")  # , fallback=1)
        self.lr = config.getfloat("config", "lr")  # , fallback=1e-3)
        self.end_lr = config.getfloat("config", "end_lr", fallback=self.lr)
        self.deterministic = config.getboolean("config", "deterministic")
        self.restarts = config.getint("config", "restarts")  # , fallback=None)
        self.seed = config.getint("config", "seed", fallback=1)
        self.test_interval = config.getint("config", "test_interval", fallback=1)
        self.log_interval = config.getint("config", "log_interval", fallback=10)
        # self.save_interval = config.getint("config", "save_interval", fallback=10)
        # self.save_model = config.getboolean("config", "save_model", fallback=False)
        self.optimizer = config.get("config", "optimizer")  # , fallback="SGD")
        self.differentially_private = config.getboolean(
            "config", "differentially_private", fallback=False
        )
        assert self.optimizer in ["SGD", "Adam"], "Unknown optimizer"
        if self.optimizer == "Adam":
            self.beta1 = config.getfloat("config", "beta1", fallback=0.9)
            self.beta2 = config.getfloat("config", "beta2", fallback=0.999)
        self.model = config.get("config", "model")  # , fallback="simpleconv")
        assert self.model in ["simpleconv", "resnet-18", "vgg16"]
        self.pooling_type = config.get("config", "pooling_type", fallback="max")
        self.pretrained = config.getboolean("config", "pretrained")  # , fallback=False)
        self.weight_decay = config.getfloat("config", "weight_decay")  # , fallback=0.0)
        self.weight_classes = config.getboolean(
            "config", "weight_classes"
        )  # , fallback=False)
        self.rotation = config.getfloat("augmentation", "rotation")  # , fallback=0.0)
        self.translate = config.getfloat("augmentation", "translate")  # , fallback=0.0)
        self.scale = config.getfloat("augmentation", "scale")  # , fallback=0.0)
        self.shear = config.getfloat("augmentation", "shear")  # , fallback=0.0)
        self.albu_prob = config.getfloat(
            "albumentations", "overall_prob"
        )  # , fallback=1.0)
        self.individual_albu_probs = config.getfloat(
            "albumentations", "individual_probs"
        )  # , fallback=1.0)
        self.noise_std = config.getfloat(
            "albumentations", "noise_std"
        )  # , fallback=1.0)
        self.noise_prob = config.getfloat(
            "albumentations", "noise_prob"
        )  # , fallback=0.0)
        self.clahe = config.getboolean("albumentations", "clahe")  # , fallback=False)
        self.randomgamma = config.getboolean(
            "albumentations", "randomgamma"
        )  # , fallback=False
        self.randombrightness = config.getboolean(
            "albumentations", "randombrightness"
        )  # , fallback=False
        self.blur = config.getboolean("albumentations", "blur")  # , fallback=False)
        self.elastic = config.getboolean(
            "albumentations", "elastic"
        )  # , fallback=False)
        self.optical_distortion = config.getboolean(
            "albumentations", "optical_distortion"
        )  # , fallback=False
        self.grid_distortion = config.getboolean(
            "albumentations", "grid_distortion"
        )  # , fallback=False)
        self.grid_shuffle = config.getboolean(
            "albumentations", "grid_shuffle"
        )  # , fallback=False
        self.hsv = config.getboolean("albumentations", "hsv")  # , fallback=False)
        self.invert = config.getboolean("albumentations", "invert")  # , fallback=False)
        self.cutout = config.getboolean("albumentations", "cutout")  # , fallback=False)
        self.shadow = config.getboolean("albumentations", "shadow")  # , fallback=False)
        self.fog = config.getboolean("albumentations", "fog")  # , fallback=False)
        self.sun_flare = config.getboolean(
            "albumentations", "sun_flare"
        )  # , fallback=False
        self.solarize = config.getboolean(
            "albumentations", "solarize"
        )  # , fallback=False)
        self.equalize = config.getboolean(
            "albumentations", "equalize"
        )  # , fallback=False)
        self.grid_dropout = config.getboolean(
            "albumentations", "grid_dropout"
        )  # , fallback=False
        self.mixup = config.getboolean("augmentation", "mixup")  # , fallback=False)
        self.mixup_prob = config.getfloat(
            "augmentation", "mixup_prob"
        )  # , fallback=None)
        self.mixup_lambda = config.getfloat(
            "augmentation", "mixup_lambda", fallback=None
        )
        if self.mixup and self.mixup_prob == 1.0:
            self.batch_size *= 2
            print("Doubled batch size because of mixup")
        self.train_federated = cmd_args.train_federated if mode == "train" else False
        self.unencrypted_aggregation = (
            cmd_args.unencrypted_aggregation if mode == "train" else False
        )
        if self.train_federated:
            self.sync_every_n_batch = config.getint(
                "federated", "sync_every_n_batch"
            )  # , fallback=10
            self.wait_interval = config.getfloat(
                "federated", "wait_interval", fallback=0.1
            )
            self.keep_optim_dict = config.getboolean(
                "federated", "keep_optim_dict"
            )  # , fallback=False
            self.repetitions_dataset = config.getint(
                "federated", "repetitions_dataset"
            )  # , fallback=1
            if self.repetitions_dataset > 1:
                self.epochs = int(self.epochs / self.repetitions_dataset)
                if verbose:
                    print(
                        "Number of epochs was decreased to "
                        "{:d} because of {:d} repetitions of dataset".format(
                            self.epochs, self.repetitions_dataset
                        )
                    )
            self.weighted_averaging = config.getboolean(
                "federated", "weighted_averaging"
            )  # , fallback=False

        self.visdom = cmd_args.visdom if mode == "train" else False
        self.encrypted_inference = (
            cmd_args.encrypted_inference if mode == "inference" else False
        )
        self.data_dir = cmd_args.data_dir  # options: ['pneumonia', 'mnist']
        self.cuda = cmd_args.cuda
        self.websockets = cmd_args.websockets if mode == "train" else False
        if self.websockets:
            assert self.train_federated, "If you use websockets it must be federated"
        self.num_threads = config.getint("system", "num_threads", fallback=0)

    @classmethod
    def from_namespace(cls, args):
        obj = cls.__new__(cls)
        super(Arguments, obj).__init__()
        for attr in dir(args):
            if (
                not callable(getattr(args, attr))
                and not attr.startswith("__")
                and attr in dir(args)
            ):
                setattr(obj, attr, getattr(args, attr))
        return obj

    def from_previous_checkpoint(self, cmd_args):
        self.visdom = False
        if hasattr(cmd_args, "encrypted_inference"):
            self.encrypted_inference = cmd_args.encrypted_inference
        self.cuda = cmd_args.cuda
        self.websockets = (
            cmd_args.websockets  # currently not implemented for inference
            if self.encrypted_inference and hasattr(cmd_args, "websockets")
            else False
        )
        if not "mixup" in dir(self):
            self.mixup = False

    def incorporate_cmd_args(self, cmd_args):
        exceptions = []  # just for future
        for attr in dir(self):
            if (
                not callable(getattr(self, attr))
                and not attr.startswith("__")
                and attr in dir(cmd_args)
                and attr not in exceptions
            ):
                setattr(self, attr, getattr(cmd_args, attr))

    def __str__(self):
        members = [
            attr
            for attr in dir(self)
            if not callable(getattr(self, attr)) and not attr.startswith("__")
        ]
        rows = []
        for x in members:
            rows.append([str(x), str(getattr(self, x))])
        return tabulate(rows)


class AddGaussianNoise(torch.nn.Module):
    def __init__(self, mean: float = 0.0, std: float = 1.0, p: Optional[float] = None):
        super(AddGaussianNoise, self).__init__()
        self.std = std
        self.mean = mean
        self.p = p

    def forward(self, tensor: torch.Tensor):
        if self.p and self.p < random():
            return tensor
        return (
            tensor
            + torch.randn(tensor.size()) * self.std  # pylint: disable=no-member
            + self.mean
        )

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1}{:s})".format(
            self.mean, self.std, ", apply prob={:f}".format(self.p) if self.p else ""
        )


class MixUp(torch.nn.Module):
    def __init__(self, λ: Optional[float] = None, p: Optional[float] = None):
        super(MixUp, self).__init__()
        assert 0.0 <= p <= 1.0, "probability needs to be in [0,1]"
        self.p = p
        if λ:
            assert 0.0 <= λ <= 1.0, "mix factor needs to be in [0,1]"
        self.λ = λ

    def forward(
        self, x: Tuple[Union[torch.tensor, Tuple[torch.tensor]], Tuple[torch.Tensor]],
    ):
        assert len(x) == 2, "need data and target"
        x, y = x
        if self.p:
            if random() > self.p:
                if torch.is_tensor(x):
                    return x, y
                else:
                    return x[0], y[0]
        if torch.is_tensor(x):
            L = x.shape[0]
        elif type(x) == tuple and all(
            [x[i].shape == x[i - 1].shape for i in range(1, len(x))]
        ):
            L = len(x)
        else:
            raise ValueError(
                "images need to be either list of equally shaped "
                "tensors or batch of size 2"
            )
        if not (
            (torch.is_tensor(y) and y.shape[0] == L)
            or (
                len(y) == L
                and all([y[i - 1].shape == y[i].shape for i in range(1, len(y))])
            )
        ):
            raise ValueError(
                "targets need to be tuple of equally shaped one hot encoded tensors"
            )
        if L == 1:
            return x, y
        if self.λ:
            λ = self.λ
        else:
            λ = random()
        if L % 2 == 0:
            h = L // 2
            if not torch.is_tensor(x):
                x = torch.stack(x).squeeze(1)  # pylint:disable=no-member
            if not torch.is_tensor(y):
                y = torch.stack(y).squeeze(1)  # pylint:disable=no-member
            x = λ * x[:h] + (1.0 - λ) * x[h:]
            y = λ * y[:h] + (1.0 - λ) * y[h:]
            return x, y
        else:
            # Actually there should be another distinction
            # between tensors and tuples
            # but in our use case this only happens if tensors
            # are used
            h = (L - 1) // 2
            out_x = torch.zeros(  # pylint:disable=no-member
                (h + 1, *x.shape[1:]), device=x.device
            )
            out_y = torch.zeros(  # pylint:disable=no-member
                (h + 1, *y.shape[1:]), device=y.device
            )
            out_x[-1] = x[-1]
            out_y[-1] = y[-1]
            out_x[:-1] = λ * x[:h] + (1.0 - λ) * x[h:-1]
            out_y[:-1] = λ * y[:h] + (1.0 - λ) * y[h:-1]
            return out_x, out_y


# adapted from https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/3
class Cross_entropy_one_hot(torch.nn.Module):
    def __init__(self, reduction="mean", weight=None):
        # Cross entropy that accepts soft targets
        super(Cross_entropy_one_hot, self).__init__()
        self.weight = (
            torch.nn.Parameter(weight, requires_grad=False)
            if weight is not None
            else None
        )
        self.reduction = reduction
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        if self.reduction == "mean":
            loss = torch.mean(  # pylint:disable=no-member
                (
                    torch.sum(self.weight * target, dim=1)  # pylint:disable=no-member
                    if self.weight is not None
                    else 1.0
                )
                * torch.sum(  # pylint:disable=no-member
                    -target * self.logsoftmax(output), dim=1
                )
            )
        elif self.reduction == "sum":
            loss = torch.sum(  # pylint:disable=no-member
                (
                    torch.sum(self.weight * target, dim=1)  # pylint:disable=no-member
                    if self.weight is not None
                    else 1.0
                )
                * torch.sum(  # pylint:disable=no-member
                    -target * self.logsoftmax(output), dim=1
                )
            )
        else:
            raise NotImplementedError("reduction method unknown")
        return loss


class To_one_hot(torch.nn.Module):
    def __init__(self, num_classes):
        super(To_one_hot, self).__init__()
        self.num_classes = num_classes

    def forward(self, x: Union[int, List[int], torch.Tensor]):
        if type(x) == int:
            x = torch.tensor(x)  # pylint:disable=not-callable
        elif type(x) == list:
            x = torch.tensor(x)  # pylint:disable=not-callable
        if len(x.shape) == 0:
            one_hot = torch.zeros(  # pylint:disable=no-member
                (self.num_classes,), device=x.device
            )
            one_hot.scatter_(0, x, 1)
            return one_hot
        elif len(x.shape) == 1:
            x = x.unsqueeze(1)
        one_hot = torch.zeros(  # pylint:disable=no-member
            (x.shape[0], self.num_classes), device=x.device
        )
        one_hot.scatter_(1, x, 1)
        return one_hot


def save_config_results(args, score: float, timestamp: str, table: str):
    members = [
        attr
        for attr in dir(args)
        if not callable(getattr(args, attr)) and not attr.startswith("__")
    ]
    if not isfile(table):
        print("Configuration table does not exist - Creating new")
        df = pd.DataFrame(columns=members)
    else:
        df = pd.read_csv(table)
    new_row = dict(zip(members, [getattr(args, x) for x in members]))
    new_row["timestamp"] = timestamp
    new_row["best_validation_score"] = score
    df = df.append(new_row, ignore_index=True)
    df.to_csv(table, index=False)


## Adaption of federated averaging from syft with option of weights
def federated_avg(models: dict, weights=None):
    """Calculate the federated average of a dictionary containing models.
       The models are extracted from the dictionary
       via the models.values() command.

    Args:
        models (Dict[Any, torch.nn.Module]): a dictionary of models
        for which the federated average is calculated.

    Returns:
        torch.nn.Module: the module with averaged parameters.
    """
    if weights:
        model = None
        for idt, partial_model in models.items():
            scaled_model = scale_model(
                partial_model, weights[idt]
            )  # @a1302z are we consciously overwriting the id keyword here?
            if model:
                model = add_model(model, scaled_model)
            else:
                model = scaled_model
    else:
        model = syft_fl_utils.federated_avg(models)
    return model


def training_animation(done: mp.Value, message: str = "training"):
    i = 0
    while not done.value:
        if i % 4 == 0:
            print("\r \033[K", end="{:s}".format(message), flush=True)
            i = 1
        else:
            print(".", end="", flush=True)
            i += 1
        sleep(0.5)
    print("\r \033[K")


def progress_animation(done, progress_dict):
    while not done.value:
        content, headers = [], []
        for worker, (batch, total) in progress_dict.items():
            headers.append(worker)
            content.append("{:d}/{:d}".format(batch, total))
        print(tabulate([content], headers=headers, tablefmt="plain"))
        sleep(0.1)
        print("\033[F" * 3)
    print("\033[K \n \033[K \033[F \033[F")


def dict_to_mng_dict(dictionary: dict, mng: mp.Manager):
    return_dict = mng.dict()
    for key, value in dictionary.items():
        return_dict[key] = value
    return return_dict


## Assuming train loaders is dictionary with {worker : train_loader}
def train_federated(
    args,
    model,
    device,
    train_loaders,
    optimizer,
    epoch,
    loss_fn,
    crypto_provider,
    test_params=None,
    vis_params=None,
    verbose=True,
    privacy_engines=None,
):

    total_batches = 0
    total_batches = sum([len(tl) for tl in train_loaders.values()])
    w_dict = None
    if args.weighted_averaging:
        w_dict = {
            worker.id: len(tl) / total_batches for worker, tl in train_loaders.items()
        }
    if args.unencrypted_aggregation:
        mng = mp.Manager()
        # model.train()
        result_dict, waiting_for_sync_dict, sync_dict, progress_dict, loss_dict = (
            mng.dict(),
            mng.dict(),
            mng.dict(),
            mng.dict(),
            mng.dict(),
        )
        stop_sync, sync_completed = mng.Value("i", False), mng.Value("i", False)
        # num_workers = mng.Value("d", len(train_loaders.keys()))
        for worker in train_loaders.keys():
            result_dict[worker.id] = None
            loss_dict[worker.id] = mng.dict()
            waiting_for_sync_dict[worker.id] = False
            progress_dict[worker.id] = (0, len(train_loaders[worker]))

        jobs = [
            mp.Process(
                name="{:s} training".format(worker.id),
                target=train_on_server,
                args=(
                    args,
                    model[worker.id],
                    worker,
                    device,
                    train_loader,
                    optimizer[worker.id],
                    loss_fn[worker.id],
                    result_dict,
                    waiting_for_sync_dict,
                    sync_dict,
                    sync_completed,
                    progress_dict,
                    loss_dict,
                ),
            )
            for worker, train_loader in train_loaders.items()
        ]
        for j in jobs:
            j.start()
        synchronize = mp.Process(
            name="synchronization",
            target=synchronizer,
            args=(
                args,
                result_dict,
                waiting_for_sync_dict,
                sync_dict,
                progress_dict,
                loss_dict,
                stop_sync,
                sync_completed,
                w_dict,
                epoch,
            ),
            kwargs={
                "wait_interval": args.wait_interval,
                "vis_params": vis_params,
                "test_params": test_params,
            },
        )
        synchronize.start()
        if verbose:
            done = mng.Value("i", False)
            animate = mp.Process(
                name="animation", target=progress_animation, args=(done, progress_dict)
            )
            animate.start()
        for j in jobs:
            j.join()
        stop_sync.value = True
        synchronize.join()
        if verbose:
            done.value = True
            animate.join()

        model["local_model"] = sync_dict["model"]
        for w in train_loaders.keys():
            if model[w.id].location:
                model[w.id].get()
            model[w.id].load_state_dict(sync_dict["model"].state_dict())
            if not args.keep_optim_dict:
                kwargs = {"lr": args.lr, "weight_decay": args.weight_decay}
                if args.optimizer == "Adam":
                    kwargs["betas"] = (args.beta1, args.beta2)
                optimizer[w.id].__init__(model[w.id].parameters(), **kwargs)

        avg_loss = np.average(
            [l["final"] for l in loss_dict.values()],
            weights=[w_dict[w] for w in loss_dict.keys()] if w_dict else None,
        )

    else:  # encrypted aggregation
        model, avg_loss = secure_aggregation_epoch(
            args,
            model,
            device,
            train_loaders,
            optimizer,
            epoch,
            loss_fn,
            crypto_provider,
            test_params=test_params,
            weights=w_dict,
        )
    if args.visdom:
        vis_params["vis"].line(
            X=np.asarray([epoch]),
            Y=np.asarray([avg_loss]),
            win="loss_win",
            name="train_loss",
            update="append",
            env=vis_params["vis_env"],
        )
    else:
        if verbose:
            print("Train Epoch: {} \tLoss: {:.6f}".format(epoch, avg_loss,))
    return model


def tensor_iterator(model: "torch.Model") -> "Sequence[Iterator]":
    """adding relavant iterators for the tensor elements
    """
    iterators = [
        "parameters",
        "buffers",
    ]  # all the element iterators from nn module should be listed here,
    return [getattr(model, i) for i in iterators]


def aggregation(
    local_model,
    models,
    workers,
    crypto_provider,
    args,
    test_params,
    weights=None,
    secure=True,
):
    """(Very) defensive version of the original secure aggregation relying on actually checking the parameter names and shapes before trying to load them into the model.
    """

    local_keys = local_model.state_dict().keys()

    # make sure we're not getting cheated and some key or shape has been changed behind our backs
    ids = [name if type(name) == str else name.id for name in workers]
    remote_keys = []
    for id_ in ids:
        remote_keys.extend(list(models[id_].state_dict().keys()))

    c = Counter(remote_keys)
    assert np.all(
        list(c.values()) == np.full_like(list(c.values()), len(workers))
    ) and list(c.keys()) == list(
        local_keys
    )  # we know that the keys match exactly and are all present

    for key in list(local_keys):
        if "num_batches_tracked" in key:
            continue
        local_shape = local_model.state_dict()[key].shape
        remote_shapes = [
            models[worker if type(worker) == str else worker.id].state_dict()[key].shape
            for worker in workers
        ]
        assert len(set(remote_shapes)) == 1 and local_shape == next(
            iter(set(remote_shapes))
        ), "Shape mismatch BEFORE sending and getting"
    # print(list(local_keys))
    # If we have reached here, we are pretty sure the models are identical down to the shapes
    fresh_state_dict = dict()
    for key in list(local_keys):  # which are same as remote_keys for sure now
        if "num_batches_tracked" in str(key):
            continue
        local_shape = local_model.state_dict()[key].shape
        remote_param_list = []
        for worker in workers:
            if secure:
                remote_param_list.append(
                    (
                        models[worker if type(worker) == str else worker.id]
                        .state_dict()[key]
                        .data.copy()
                        * (
                            weights[worker if type(worker) == str else worker.id]
                            if weights
                            else 1
                        )
                    )
                    .fix_prec()
                    .share(*workers, crypto_provider=crypto_provider, protocol="fss")
                    .get()
                )
            else:
                remote_param_list.append(
                    models[worker if type(worker) == str else worker.id]
                    .state_dict()[key]
                    .data.copy()
                    * (
                        weights[worker if type(worker) == str else worker.id]
                        if weights
                        else 1
                    )
                )

        remote_shapes = [p.shape for p in remote_param_list]
        assert len(set(remote_shapes)) == 1 and local_shape == next(
            iter(set(remote_shapes))
        ), "Shape mismatch AFTER sending and getting"
        if secure:
            sumstacked = (
                torch.sum(  # pylint:disable=no-member
                    torch.stack(remote_param_list), dim=0  # pylint:disable=no-member
                )
                .get()
                .float_prec()
            )
        else:
            sumstacked = torch.sum(  # pylint:disable=no-member
                torch.stack(remote_param_list), dim=0  # pylint:disable=no-member
            )
        fresh_state_dict[key] = sumstacked if weights else sumstacked / len(workers)
    local_model.load_state_dict(fresh_state_dict)
    return local_model


# def aggregation_old(
#     local_model,
#     models,
#     workers,
#     crypto_provider,
#     args,
#     test_params,
#     weights=None,
#     secure=True,
# ):
#     with torch.no_grad():
#         local_iter = tensor_iterator(local_model)[0]
#         remote_iter = {
#             key: tensor_iterator(model)[0]()
#             for key, model in models.items()
#             if key != "local_model"
#         }
#         for local_param in local_iter():
#             remote_params = {w: next(r) for w, r in remote_iter.items()}
#             dt = remote_params[list(remote_iter.keys())[0]].dtype
#             ## num_batches tracked are ints
#             ## -> irrelevant cause batch norm momentum is on by default
#             if dt != torch.float:
#                 continue
#             if secure:
#                 param_stack = torch.sum(
#                     torch.stack(
#                         [
#                             r.data.copy()
#                             .fix_prec()
#                             .share(
#                                 *workers,
#                                 crypto_provider=crypto_provider,
#                                 protocol="fss"
#                             )
#                             .get()
#                             * weights[w]
#                             for w, r in remote_params.items()
#                         ]
#                     ),
#                     dim=0,
#                 )
#             else:
#                 param_stack = torch.sum(
#                     torch.stack(
#                         [r.data.copy() * weights[w] for w, r in remote_params.items()]
#                     ),
#                     dim=0,
#                 )
#             if secure:
#                 param_stack = param_stack.get().float_prec()
#             if weights is None:
#                 param_stack /= len(remote_params)
#             local_param.set_(param_stack)
#     return local_model


def send_new_models(local_model, models):  # original version
    for worker in models.keys():
        if worker == "local_model":
            continue
        if local_model.location is not None:  # local model is at worker
            local_model.get()
        local_model.send(worker)
        models[worker].load_state_dict(local_model.state_dict())
    local_model.get()
    return models  # returns models updated on workers


def secure_aggregation_epoch(
    args,
    models,
    device,
    train_loaders,
    optimizers,
    epoch,
    loss_fns,
    crypto_provider,
    weights=None,
    test_params=None,
    verbose=True,
    privacy_engines=None,
):
    for worker in train_loaders.keys():
        if models[worker.id].location is not None:
            continue  # model + loss already at a worker, 783/784 don't happen
        else:  # location is None -> local model
            models[worker.id].send(worker)
            loss_fns[worker.id] = loss_fns[worker.id].send(
                worker
            )  # stuff sent to worker

    models = send_new_models(
        models["local_model"], models
    )  # stuff sent to worker again (on 1st epoch it's irrelevant, afterwards it's the model updating)
    # @a1302z we could refactor this to be more elegant and not do this twice in the first epoch

    if not args.keep_optim_dict:
        for worker in optimizers.keys():
            kwargs = {"lr": args.lr, "weight_decay": args.weight_decay}
            ## TODO implement for SGD (also in train_federated)
            if args.optimizer == "Adam":
                kwargs["betas"] = (args.beta1, args.beta2)
                opt = torch.optim.Adam
            elif args.optimizer == "SGD":
                opt = torch.optim.SGD
            else:
                raise NotImplementedError("only Adam or SGD supported.")
            optimizers[worker] = opt(
                models[worker].parameters(), **kwargs
            )  # no send operation here?
            if privacy_engines:
                privacy_engines[worker].attach(optimizers[worker])

    avg_loss = []

    num_batches = {key.id: len(loader) for key, loader in train_loaders.items()}
    dataloaders = {key: iter(loader) for key, loader in train_loaders.items()}
    pbar = tqdm.tqdm(
        range(max(num_batches.values())),
        total=max(num_batches.values()),
        leave=False,
        desc="Training with secure aggregation",
    )
    for batch_idx in pbar:
        for worker, dataloader in tqdm.tqdm(
            dataloaders.items(),
            total=len(dataloaders),
            leave=False,
            desc="Train batch {:d}".format(batch_idx),
        ):
            if batch_idx >= num_batches[worker.id]:
                continue
            optimizers[worker.id].zero_grad()
            data, target = next(dataloader)
            pred = models[worker.id](data)
            loss = loss_fns[worker.id](pred, target)
            loss.backward()
            optimizers[worker.id].step()
            avg_loss.append(loss.detach().cpu().get().item())
        if batch_idx > 0 and batch_idx % args.sync_every_n_batch == 0:
            pbar.set_description_str("Synchronizing")
            models["local_model"] = aggregation(
                models["local_model"],
                models,
                train_loaders.keys(),
                crypto_provider,
                args,
                test_params,
                weights=weights,
            )
            models = send_new_models(models["local_model"], models)
            pbar.set_description_str("Training with secure aggregation")
            if args.keep_optim_dict:
                # In the future we'd like to have a method here that aggregates
                # the stats of all optimizers just as we do with models.
                # However, this is a complex task and we currently have not
                # the capacities for this.
                pass
            else:
                for worker in optimizers.keys():
                    kwargs = {"lr": args.lr, "weight_decay": args.weight_decay}
                    ## TODO implement for SGD (also in train_federated)
                    if args.optimizer == "Adam":
                        kwargs["betas"] = (args.beta1, args.beta2)
                        opt = torch.optim.Adam
                    elif args.optimizer == "SGD":
                        opt = torch.optim.SGD
                    else:
                        raise NotImplementedError("only Adam or SGD supported.")
                    optimizers[worker] = opt(models[worker].parameters(), **kwargs)

    models["local_model"] = aggregation(
        models["local_model"],
        models,
        train_loaders.keys(),
        crypto_provider,
        args,
        test_params,
        weights=weights,
    )
    avg_loss = np.mean(avg_loss)

    return models, avg_loss


def synchronizer(  # never gets called on websockets
    args,
    result_dict,
    waiting_for_sync_dict,
    sync_dict,
    progress_dict,
    loss_dict,
    stop,
    sync_completed,
    weights,
    epoch,
    wait_interval=0.1,
    vis_params=None,
    test_params=None,
):
    if test_params:
        pass
        # save_iter: int = 1
    while not stop.value:
        while not all(waiting_for_sync_dict.values()) and not stop.value:
            sleep(0.1)
        # print("synchronizing: models from {:s}".format(str(result_dict.keys())))
        if len(result_dict) == 1:
            for model in result_dict.values():
                sync_dict["model"] = model
        elif len(result_dict) == 0:
            pass
        else:
            models = {}
            for idt, worker_model in result_dict.items():
                models[idt] = worker_model
            # avg_model = federated_avg(models, weights=weights)
            avg_model = aggregation(
                deepcopy(list(models.values())[0]),
                models,
                models.keys(),
                None,
                args,
                test_params,
                weights=weights,
                secure=False,
            )
            sync_dict["model"] = avg_model
        for k in waiting_for_sync_dict.keys():
            waiting_for_sync_dict[k] = False
        ## In theory we should clear the models here
        ## However, if one worker has more samples than any other worker,
        ## but has imbalanced data this destroys our training.
        ## By keeping the last model of each worker in the dict,
        ## we still assure that it's training is not lost
        # result_dict.clear()

        ## this should be commented in but it triggers some backend failure
        """
        progress = progress_dict.values()
        cur_batch = max([p[0] for p in progress])
        save_after = max([p[1] for p in progress]) / args.repetitions_dataset
        progress = sum([p[0] / p[1] for p in progress]) / len(progress)

        if vis_params:
            if progress >= 1.0:
                sync_completed.value = True
                continue
            avg_loss = np.mean(
                [l[cur_batch] for l in loss_dict.values() if cur_batch in l]
            )
            vis_params["vis"].line(
                X=np.asarray([epoch - 1 + progress]),
                Y=np.asarray([avg_loss]),
                win="loss_win",
                name="train_loss",
                update="append",
                env=vis_params["vis_env"],
            )
        if test_params and cur_batch > (save_after * save_iter):
            model = avg_model.copy()
            _, score = test(
                args,
                model,
                test_params["device"],
                test_params["val_loader"],
                epoch,
                test_params["loss_fn"],
                verbose=False,
                num_classes=test_params["num_classes"],
                vis_params=vis_params,
                class_names=test_params["class_names"],
            )
            model_path = "model_weights/{:s}_epoch_{:03d}.pt".format(
                test_params["exp_name"],
                epoch * (args.repetitions_dataset if args.repetitions_dataset else 1)
                + save_iter,
            )

            save_model(model, test_params["optimizer"], model_path, args, epoch)
            test_params["scores"].append(score)
            test_params["model_paths"].append(model_path)

            save_iter += 1"""
        sync_completed.value = True


def train_on_server(  # never gets called on websockets
    args,
    model,
    worker,
    device,
    train_loader,
    optimizer,
    loss_fn,
    result_dict,
    waiting_for_sync_dict,
    sync_dict,
    sync_completed,
    progress_dict,
    loss_dict,
):
    # optimizer = optim.get_optim(worker.id)
    avg_loss = []
    model.train()
    if not model.location:
        model.send(worker)
        loss_fn = loss_fn.send(worker)
    L = len(train_loader)
    for batch_idx, (data, target) in enumerate(train_loader):
        progress_dict[worker.id] = (batch_idx, L)
        if (
            # batch_idx % int(0.1 * L) == 0 and batch_idx > 0
            batch_idx % args.sync_every_n_batch == 0
            and batch_idx > 0
        ):  # synchronize models
            model = model.get()
            result_dict[worker.id] = model
            loss_dict[worker.id][batch_idx] = avg_loss[-1]
            sync_completed.value = False
            waiting_for_sync_dict[worker.id] = True
            while not sync_completed.value:
                sleep(args.wait_interval)
            model.load_state_dict(sync_dict["model"].state_dict())
            if not args.keep_optim_dict:
                kwargs = {"lr": args.lr, "weight_decay": args.weight_decay}
                if args.optimizer == "Adam":
                    kwargs["betas"] = (args.beta1, args.beta2)
                optimizer.__init__(model.parameters(), **kwargs)
            model.train()
            model.send(worker)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        loss = loss.get()
        avg_loss.append(loss.detach().cpu().item())
    model.get()
    loss_fn = loss_fn.get()
    result_dict[worker.id] = model
    loss_dict[worker.id]["final"] = np.mean(avg_loss)
    progress_dict[worker.id] = (batch_idx + 1, L)
    del waiting_for_sync_dict[worker.id]
    # optim.optimizers[worker.id] = optimizer


def train(  # never called on websockets
    args,
    model,
    device,
    train_loader,
    optimizer,
    epoch,
    loss_fn,
    num_classes,
    vis_params=None,
    verbose=True,
):
    model.train()
    if args.mixup:
        mixup = MixUp(λ=args.mixup_lambda, p=args.mixup_prob)
        oh_converter = To_one_hot(num_classes)
        oh_converter.to(device)

    L = len(train_loader)
    div = 1.0 / float(L)
    avg_loss = []
    for batch_idx, (data, target) in tqdm.tqdm(
        enumerate(train_loader),
        leave=False,
        desc="training epoch {:d}".format(epoch),
        total=L + 1,
    ):
        data, target = data.to(device), target.to(device)
        if args.mixup:
            with torch.no_grad():
                target = oh_converter(target)
                data, target = mixup((data, target))
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            if args.visdom:
                vis_params["vis"].line(
                    X=np.asarray([epoch + float(batch_idx) * div - 1]),
                    Y=np.asarray([loss.item()]),
                    win="loss_win",
                    name="train_loss",
                    update="append",
                    env=vis_params["vis_env"],
                )
            else:
                avg_loss.append(loss.item())
    if not args.visdom and verbose:
        print("Train Epoch: {} \tLoss: {:.6f}".format(epoch, np.mean(avg_loss),))
    return model


def stats_table(
    conf_matrix, report, roc_auc=0.0, matthews_coeff=0.0, class_names=None, epoch=0
):
    rows = []
    for i in range(conf_matrix.shape[0]):
        report_entry = report[str(i)]
        row = [
            class_names[i] if class_names else i,
            "{:.1f} %".format(report_entry["recall"] * 100.0),
            "{:.1f} %".format(report_entry["precision"] * 100.0),
            "{:.1f} %".format(report_entry["f1-score"] * 100.0),
            report_entry["support"],
        ]
        row.extend([conf_matrix[i, j] for j in range(conf_matrix.shape[1])])
        rows.append(row)
    rows.append(
        [
            "Overall (macro)",
            "{:.1f} %".format(report["macro avg"]["recall"] * 100.0),
            "{:.1f} %".format(report["macro avg"]["precision"] * 100.0),
            "{:.1f} %".format(report["macro avg"]["f1-score"] * 100.0),
            report["macro avg"]["support"],
        ]
    )
    rows.append(
        [
            "Overall (weighted)",
            "{:.1f} %".format(report["weighted avg"]["recall"] * 100.0),
            "{:.1f} %".format(report["weighted avg"]["precision"] * 100.0),
            "{:.1f} %".format(report["weighted avg"]["f1-score"] * 100.0),
            report["weighted avg"]["support"],
        ]
    )
    rows.append(["Overall stats", "micro recall", "matthews coeff", "AUC ROC score"])
    rows.append(
        [
            "",
            "{:.1f} %".format(100.0 * report["accuracy"]),
            "{:.3f}".format(matthews_coeff),
            "{:.3f}".format(roc_auc),
        ]
    )
    headers = [
        "Epoch {:d}".format(epoch),
        "Recall",
        "Precision",
        "F1 score",
        "n total",
    ]
    headers.extend(
        [class_names[i] if class_names else i for i in range(conf_matrix.shape[0])]
    )
    return tabulate(rows, headers=headers, tablefmt="fancy_grid",)


def test(
    args,
    model,
    device,
    val_loader,
    epoch,
    loss_fn,
    num_classes,
    verbose=True,
    vis_params=None,
    class_names=None,
):
    oh_converter = None
    if args.mixup or (args.train_federated and args.weight_classes):
        oh_converter = To_one_hot(num_classes)
        oh_converter.to(device)
    model.eval()
    test_loss, TP = 0, 0
    total_pred, total_target, total_scores = [], [], []
    with torch.no_grad():
        for data, target in (
            tqdm.tqdm(
                val_loader,
                total=len(val_loader),
                desc="testing epoch {:d}".format(epoch),
                leave=False,
            )
            if verbose
            else val_loader
        ):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_fn(output, oh_converter(target) if oh_converter else target)
            test_loss += loss.item()  # sum up batch loss
            total_scores.append(output)
            pred = output.argmax(dim=1)
            tgts = target.view_as(pred)
            total_pred.append(pred)
            total_target.append(tgts)
            equal = pred.eq(tgts)
            TP += (
                equal.sum().copy().get().float_precision().long().item()
                if args.encrypted_inference
                else equal.sum().item()
            )
    test_loss /= len(val_loader)
    if args.encrypted_inference:
        objective = 100.0 * TP / (len(val_loader) * args.test_batch_size)
        L = len(val_loader.dataset)
        if verbose:
            print(
                "Test set: Epoch: {:d} Average loss: {:.4f}, Recall: {}/{} ({:.0f}%)\n".format(
                    epoch, test_loss, TP, L, objective,
                ),
                # end="",
            )
    else:
        total_pred = torch.cat(total_pred).cpu().numpy()  # pylint: disable=no-member
        total_target = (
            torch.cat(total_target).cpu().numpy()  # pylint: disable=no-member
        )
        total_scores = (
            torch.cat(total_scores).cpu().numpy()  # pylint: disable=no-member
        )
        total_scores -= total_scores.min(axis=1)[:, np.newaxis]
        total_scores = total_scores / total_scores.sum(axis=1)[:, np.newaxis]
        try:
            roc_auc = mt.roc_auc_score(total_target, total_scores, multi_class="ovo")
        except ValueError:
            raise UserWarning(
                "ROC AUC score could not be calculated and was set to zero."
            )
            roc_auc = 0.0
        matthews_coeff = mt.matthews_corrcoef(total_target, total_pred)
        objective = 100.0 * matthews_coeff
        if verbose:
            conf_matrix = mt.confusion_matrix(total_target, total_pred)
            report = mt.classification_report(
                total_target, total_pred, output_dict=True, zero_division=0
            )
            print(
                stats_table(
                    conf_matrix,
                    report,
                    roc_auc=roc_auc,
                    matthews_coeff=matthews_coeff,
                    class_names=class_names,
                    epoch=epoch,
                )
            )
        if args.visdom and vis_params:
            vis_params["vis"].line(
                X=np.asarray([epoch]),
                Y=np.asarray([test_loss]),
                win="loss_win",
                name="val_loss",
                update="append",
                env=vis_params["vis_env"],
            )
            vis_params["vis"].line(
                X=np.asarray([epoch]),
                Y=np.asarray([objective / 100.0]),
                win="loss_win",
                name="matthews coeff",
                update="append",
                env=vis_params["vis_env"],
            )

    return test_loss, objective


def save_model(model, optim, path, args, epoch, val_mean_std):
    if args.train_federated:
        opt_state_dict = {key: optim.state_dict() for key, optim in optim.items()}
    # elif args.train_federated:
    #     opt_state_dict = {
    #         name: optim.get_optim(name).state_dict() for name in optim.workers
    #     }
    else:
        opt_state_dict = optim.state_dict()
    dirpath = split(path)[0]
    if not isdir(dirpath):
        makedirs(dirpath)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model["local_model"].state_dict()
            if args.train_federated
            else model.state_dict(),
            "optim_state_dict": opt_state_dict,
            "args": args,
            "val_mean_std": val_mean_std,
        },
        path,
    )
