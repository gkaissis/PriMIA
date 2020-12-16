import multiprocessing as mp
from os import makedirs
from os.path import isfile, split, isdir, join
from random import random
from time import sleep
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import segmentation_models_pytorch as smp

import tqdm
import albumentations as a
from sklearn import metrics as mt
import syft as sy
import syft.frameworks.torch.fl.utils as syft_fl_utils
from syft.frameworks.torch.fl.utils import add_model, scale_model
from tabulate import tabulate
from collections import Counter
from copy import deepcopy
from warnings import warn, filterwarnings
from torchvision import datasets, transforms

from .dataloader import (
    AlbumentationsTorchTransform,
    calc_mean_std,
    LabelMNIST,
    random_split,
    create_albu_transform,
    CombinedLoader,
    SegmentationData, # Segmentation 
    MSD_data, 
    MSD_data_images, 
)

filterwarnings("ignore", message="invalid value encountered in double_scalars")


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
            warn(
                "We are not supporting different train and inference"
                " resolutions although it works for some scenarios.",
                category=UserWarning,
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
        assert self.model in ["simpleconv", "resnet-18", "vgg16", "simple_seg_net", "monet_seg_net"] # Segmentation 
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
            self.precision_fractional = config.getfloat(
                "federated", "precision_fractional", fallback=16
            )
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
        self,
        x: Tuple[Union[torch.tensor, Tuple[torch.tensor]], Tuple[torch.Tensor]],
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


def calc_class_weights(args, train_loader, num_classes):
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
            if args.train_federated and (args.mixup or args.weight_classes):
                target = target.max(dim=1)
                target = target[1]  # without pysyft it should be target.indices
            for i in range(num_classes):
                n = target.eq(comparison[i][..., : target.shape[0]]).sum()
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
    from torchlib.run_websocket_server import (  # pylint:disable=import-error
        read_websocket_config,
    )

    worker_dict = read_websocket_config("configs/websetting/config.csv")
    worker_names = [id_dict["id"] for _, id_dict in worker_dict.items()]
    """if "validation" in worker_names:
        worker_names.remove("validation")"""

    crypto_in_config = "crypto_provider" in worker_names
    crypto_provider = None
    assert (args.unencrypted_aggregation) or (
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
        if not args.unencrypted_aggregation:
            crypto_provider_data = worker_dict[cp_key]
        worker_dict.pop(cp_key)

    loader = CombinedLoader()
    if not args.pretrained:
        loader.change_channels(1)

    if args.websockets:
        if args.weight_classes or (args.mixup and args.data_dir == "mnist"):
            raise NotImplementedError(
                "Weighted loss/ MixUp in combination with MNIST are currently not implemented."
            )
        workers = {
            worker[
                "id"
            ]: sy.grid.clients.data_centric_fl_client.DataCentricFLClient(  # pylint:disable=no-member
                hook,
                "http://{:s}:{:s}".format(worker["host"], worker["port"]),
                id=worker["id"],
                verbose=False,
                timeout=60000,
            )
            for _, worker in worker_dict.items()
        }
        if not args.unencrypted_aggregation:
            crypto_provider = sy.grid.clients.data_centric_fl_client.DataCentricFLClient(  # pylint:disable=no-member
                hook,
                "http://{:s}:{:s}".format(
                    crypto_provider_data["host"], crypto_provider_data["port"]
                ),
                id=crypto_provider_data["id"],
                verbose=False,
                timeout=60000,
            )

    else:
        workers = {
            worker["id"]: sy.VirtualWorker(hook, id=worker["id"], verbose=False)
            for _, worker in worker_dict.items()
        }
        for worker in workers.keys():
            workers[worker].object_store.clear_objects()
        if args.data_dir == "mnist":
            dataset = datasets.MNIST(
                root="./data",
                train=True,
                download=True,
                transform=AlbumentationsTorchTransform(
                    a.Compose(
                        [
                            a.ToFloat(max_value=255.0),
                            a.Lambda(image=lambda x, **kwargs: x[:, :, np.newaxis]),
                        ]
                    )
                ),
            )
            lengths = [int(len(dataset) / len(workers)) for _ in workers]
            ##assert sum of lenghts is whole dataset on the cost of the last worker
            ##-> because int() floors division, means that rest is send to last worker
            lengths[-1] += len(dataset) - sum(lengths)
            mnist_datasets = random_split(dataset, lengths)
            mnist_datasets = {worker: d for d, worker in zip(mnist_datasets, workers)}
        
        # Segmentation 
        elif args.data_dir == "seg_data": 
            # Imagepath to the two the parent directory of the two label files 
            ## MSRC dataset ##
            #dataset = SegmentationData(image_paths_file='data/segmentation_data/train.txt')

            ## MSD dataset ##
            """
            PATH = "/Volumes/NWR/TUM-EI Studium/Master/DEA/03_semester/GR-PriMIA/Task03_Liver"
            RES = 256
            RES_Z = 64
            CROP_HEIGHT = 16

            sample_limit = 9
            dataset = MSD_data(
                path_string=PATH, 
                res=RES, 
                res_z=RES_Z,
                crop_height=CROP_HEIGHT,
                sample_limit=sample_limit,
            )

            #TODO: Does it make sense to split already here? 
            #      Different than normal for the other datasets. 
            # split into val and train set 
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            dataset, valset = torch.utils.data.random_split(dataset, [train_size, val_size])
            """

            ## MSD dataset preprocessed version ##
            #PATH = "/Volumes/NWR/TUM-EI Studium/Master/DEA/03_semester/GR-PriMIA/Task03_Liver"
            PATH = "/home/NiWaRe/PriMIA/Task03_Liver"
            dataset = MSD_data_images(PATH+'/train')
            
            lengths = [int(len(dataset) / len(workers)) for _ in workers]
            ##assert sum of lenghts is whole dataset on the cost of the last worker
            ##-> because int() floors division, means that rest is send to last worker

            lengths[-1] += len(dataset) - sum(lengths)
            seg_datasets = random_split(dataset, lengths)
            seg_datasets = {worker: d for d, worker in zip(seg_datasets, workers)}

        if not args.unencrypted_aggregation:
            crypto_provider = sy.VirtualWorker(
                hook, id="crypto_provider", verbose=False
            )
        train_loader = None
        for i, worker in tqdm.tqdm(
            enumerate(workers.values()),
            total=len(workers.keys()),
            leave=False,
            desc="load data",
        ):
            if args.data_dir == "mnist":
                # node_id = worker.id
                # KEEP_LABELS_DICT = {
                #     "alice": [0, 1, 2, 3],
                #     "bob": [4, 5, 6],
                #     "charlie": [7, 8, 9],
                #     None: list(range(10)),
                # }
                # dataset = LabelMNIST(
                #     labels=KEEP_LABELS_DICT[node_id]
                #     if node_id in KEEP_LABELS_DICT
                #     else KEEP_LABELS_DICT[None],
                #     root="./data",
                #     train=True,
                #     download=True,
                #     transform=AlbumentationsTorchTransform(
                #         a.Compose(
                #             [
                #                 a.ToFloat(max_value=255.0),
                #                 a.Lambda(image=lambda x, **kwargs: x[:, :, np.newaxis]),
                #             ]
                #         )
                #     ),
                # )
                dataset = mnist_datasets[worker.id]
                mean, std = calc_mean_std(dataset)
                dataset.dataset.transform.transform.transforms.transforms.append(  # beautiful
                    a.Normalize(mean, std, max_pixel_value=1.0)
                )
            # Segmentation 
            elif args.data_dir == "seg_data": 
                #TODO: Add transforms if necessary 
                # For now only empty structure 

                dataset = seg_datasets[worker.id]
                print(len(dataset))
                print(dataset)
                mean, std = calc_mean_std(dataset)
                
                # TODO: For now no transforms necessary - possibly add later (same as in local training case)
                #dataset.dataset.transform.transform.transforms.transforms.append(  # beautiful
                #    a.Normalize(mean, std, max_pixel_value=1.0)
                #)

            else:
                data_dir = join(args.data_dir, "worker{:d}".format(i + 1))
                stats_dataset = datasets.ImageFolder(
                    data_dir,
                    loader=loader,
                    transform=AlbumentationsTorchTransform(
                        a.Compose(
                            [
                                a.Resize(
                                    args.inference_resolution, args.inference_resolution
                                ),
                                a.RandomCrop(
                                    args.train_resolution, args.train_resolution
                                ),
                                a.ToFloat(max_value=255.0),
                            ]
                        )
                    ),
                )
                assert (
                    len(stats_dataset.classes) == 3
                ), "We can only handle data that has 3 classes: normal, bacterial and viral"
                mean, std = calc_mean_std(
                    stats_dataset,
                    save_folder=data_dir,
                )
                del stats_dataset

                target_tf = None
                if args.mixup or args.weight_classes:
                    target_tf = [
                        lambda x: torch.tensor(x),  # pylint:disable=not-callable
                        To_one_hot(3),
                    ]
                dataset = datasets.ImageFolder(
                    # path.join("data/server_simulation/", "validation")
                    # if worker.id == "validation"
                    # else
                    data_dir,
                    loader=loader,
                    transform=create_albu_transform(args, mean, std),
                    target_transform=transforms.Compose(target_tf)
                    if target_tf
                    else None,
                )
                assert (
                    len(dataset.classes) == 3
                ), "We can only handle data that has 3 classes: normal, bacterial and viral"

            mean.tag("#datamean")
            std.tag("#datastd")
            worker.load_data([mean, std])

            data, targets = [], []
            # repetitions = 1 if worker.id == "validation" else args.repetitions_dataset
            if args.mixup:
                dataset = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=1,
                    shuffle=True,
                    num_workers=args.num_threads,
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
            # Segmentation 
            if args.data_dir == "seg_data": 
                # Problem with the "torch.tensor(targets)" 
                # this is normally used to convert list of scalar tensors into a torch array 
                # however in segmentation we don't have only one scalar as target per sample but a whole mask (2D array)
                selected_targets = torch.stack(targets)  

            else: 
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
                "#traindata",
            )
            selected_targets.tag(
                "#traintargets",
            )
            worker.load_data([selected_data, selected_targets])
    if crypto_provider is not None:
        grid = sy.PrivateGridNetwork(*(list(workers.values()) + [crypto_provider]))
    else:
        grid = sy.PrivateGridNetwork(*list(workers.values()))
    data = grid.search("#traindata")
    target = grid.search("#traintargets")
    train_loader = {}
    total_L = 0
    for worker in data.keys():
        dist_dataset = [  # TODO: in the future transform here would be nice but currently raise errors
            sy.BaseDataset(
                data[worker][0],
                target[worker][0],
            )  # transform=federated_tf
        ]
        fed_dataset = sy.FederatedDataset(dist_dataset)
        total_L += len(fed_dataset)
        tl = sy.FederatedDataLoader(
            fed_dataset, batch_size=args.batch_size, shuffle=True
        )
        train_loader[workers[worker]] = tl
    means = [m[0] for m in grid.search("#datamean").values()]
    stds = [s[0] for s in grid.search("#datastd").values()]
    if len(means) == len(workers) and len(stds) == len(workers):
        mean = (
            means[0]
            .fix_precision()
            .share(*workers, crypto_provider=crypto_provider, protocol="fss")
            .get()
        )
        std = (
            stds[0]
            .fix_precision()
            .share(*workers, crypto_provider=crypto_provider, protocol="fss")
            .get()
        )
        for m, s in zip(means[1:], stds[1:]):
            mean += (
                m.fix_precision()
                .share(*workers, crypto_provider=crypto_provider, protocol="fss")
                .get()
            )
            std += (
                s.fix_precision()
                .share(*workers, crypto_provider=crypto_provider, protocol="fss")
                .get()
            )
        mean = mean.get().float_precision() / len(stds)
        std = std.get().float_precision() / len(stds)
    else:
        raise RuntimeError("no datamean/standard deviation was found on (some) worker")
    val_mean_std = torch.stack([mean, std])  # pylint:disable=no-member
    if args.data_dir == "mnist":
        valset = LabelMNIST(
            labels=list(range(10)),
            root="./data",
            train=False,
            download=True,
            transform=AlbumentationsTorchTransform(
                a.Compose(
                    [
                        a.ToFloat(max_value=255.0),
                        a.Lambda(image=lambda x, **kwargs: x[:, :, np.newaxis]),
                        a.Normalize(mean, std, max_pixel_value=1.0),
                    ]
                )
            ),
        )
    # Segmentation 
    elif args.data_dir == "seg_data": 
        # TODO: possibly add transforms (also for local case)
        # Again for now WITHOUT transforms, just play loading of the valset 

        ## MSCR dataset
        #valset = SegmentationData(image_paths_file='data/segmentation_data/val.txt')

        ## MSD dataset 
        #PATH = "/Volumes/NWR/TUM-EI Studium/Master/DEA/03_semester/GR-PriMIA/Task03_Liver"
        PATH = "/home/NiWaRe/PriMIA/Task03_Liver"
        valset = MSD_data_images(PATH+'/val')
        pass
    else:

        val_tf = [
            a.Resize(args.inference_resolution, args.inference_resolution),
            a.CenterCrop(args.train_resolution, args.train_resolution),
            a.ToFloat(max_value=255.0),
            a.Normalize(mean[None, None, :], std[None, None, :], max_pixel_value=1.0),
        ]
        valset = datasets.ImageFolder(
            join(args.data_dir, "validation"),
            loader=loader,
            transform=AlbumentationsTorchTransform(a.Compose(val_tf)),
        )
        assert (
            len(valset.classes) == 3
        ), "We can only handle data that has 3 classes: normal, bacterial and viral"

    val_loader = torch.utils.data.DataLoader(
        valset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_threads,
    )
    assert len(train_loader.keys()) == (
        len(workers.keys())
    ), "data was not correctly loaded"
    if verbose:
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
            scaled_model = scale_model(partial_model, weights[idt])
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
            print(
                "Train Epoch: {} \tLoss: {:.6f}".format(
                    epoch,
                    avg_loss,
                )
            )
    return model


def tensor_iterator(model: "torch.Model") -> "Sequence[Iterator]":
    """adding relavant iterators for the tensor elements"""
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
    """(Very) defensive version of the original secure aggregation relying on actually checking the parameter names and shapes before trying to load them into the model."""

    ## CUDA in FL ##
    # important note: the shifting has to be called on the params 
    # and summed params (after sec. agg.) directly on model as whole
    # doesn't work. 
    # set device (so that we don't need to pass it all around)
    # get first param tensor to then get device 
    if secure: 
        rand_key = next(iter(local_model.state_dict().values()))
        device = rand_key.device

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
                    ## CUDA for FL ##
                    .cpu()
                    .fix_prec(precision_fractional=args.precision_fractional)
                    .share(*workers, crypto_provider=crypto_provider, protocol="fss")
                    .get()
                )
            else:
                remote_param_list.append(
                    models[worker if type(worker) == str else worker.id]
                    ## CUDA for FL ##
                    # nothing to add here because only encrypted computations are a problem in CUDA
                    .state_dict()[key]
                    .data.copy()
                    .get()
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
                ## CUDA for FL ##
                .to(device)
            )
        else:
            sumstacked = torch.sum(  # pylint:disable=no-member
                torch.stack(remote_param_list), dim=0  # pylint:disable=no-member
            )
        fresh_state_dict[key] = sumstacked if weights else sumstacked / len(workers)
    
    local_model.load_state_dict(fresh_state_dict)
    return local_model


def send_new_models(local_model, models):  # original version
    for worker in models.keys():
        if worker == "local_model":
            continue
        if local_model.location is not None:  # local model is at worker
            local_model.get()
        local_model.send(worker)
        models[worker].load_state_dict(local_model.state_dict())
    if local_model.location is not None:
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

    if not args.keep_optim_dict:
        for worker in optimizers.keys():
            kwargs = {"lr": args.lr, "weight_decay": args.weight_decay}
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
        desc="Training with{:s} secure aggregation.".format(
            "out" if args.unencrypted_aggregation else ""
        ),
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
            ## CUDA in FL ##
            # model already to cuda in train.py
            ##TODO: Special for MSD scans (ONLY FOR THE NON-PREPROCESSED)
            #res = data.shape[-1]
            #data, target = data.view(-1, 1, res, res).to(device), target.view(-1, res, res).to(device)
            data, target = data.to(device), target.to(device)
            pred = models[worker.id](data)
            loss = loss_fns[worker.id](pred, target)
            loss.backward()
            optimizers[worker.id].step()
            avg_loss.append(loss.detach().cpu().get().item())
        if batch_idx > 0 and batch_idx % args.sync_every_n_batch == 0:
            pbar.set_description_str("Aggregating")
            models["local_model"] = aggregation(
                models["local_model"],
                models,
                train_loaders.keys(),
                crypto_provider,
                args,
                test_params,
                weights=weights,
                secure=not args.unencrypted_aggregation,
            )
            updated_models = send_new_models(
                models["local_model"],
                {
                    w: model
                    for w, model in models.items()
                    if w in num_batches and num_batches[w] > batch_idx
                },
            )
            for w, model in updated_models.items():
                models[w] = model
            pbar.set_description_str(
                "Training with{:s} secure aggregation.".format(
                    "out" if args.unencrypted_aggregation else ""
                )
            )
            if args.keep_optim_dict:
                # In the future we'd like to have a method here that aggregates
                # the stats of all optimizers just as we do with models.
                # However, this is a complex task and we currently have not
                # the capacities for this.
                pass
            else:
                for worker in optimizers.keys():
                    kwargs = {"lr": args.lr, "weight_decay": args.weight_decay}
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
        secure=not args.unencrypted_aggregation,
    )
    models = send_new_models(models["local_model"], models)
    avg_loss = np.mean(avg_loss)

    return models, avg_loss


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
        # TODO: Only for MSD without preprocessing
        #res = data.shape[-1]
        #data, target = data.view(-1, 1, res, res).to(device), target.view(-1, res, res).to(device)
        data, target = data.to(device), target.unsqueeze(dim=1).to(device)
        #dim = 256*256
        #dim_2 = int(dim/2)
        #model = nn.Sequential(
        #                        nn.Flatten(),
        #                        nn.Linear(dim, 100), 
        #                        nn.ReLU(), 
        #                        nn.Linear(100, dim), 
        #                    ).to(device)
        num_classes = 1
        model = smp.Unet("resnet18", classes=num_classes, activation="sigmoid")
        inpt_channels = 1
        if inpt_channels != 3:
            new_encoder = [nn.Conv2d(inpt_channels, 3, 1), model.encoder.conv1]
            model.encoder.conv1 = nn.Sequential(*new_encoder)
        if args.mixup:
            with torch.no_grad():
                target = oh_converter(target)
                data, target = mixup((data, target))
        optimizer.zero_grad()

        model = model.to(device)

        output = model(data)

        #output = output.view_as(target)

        #### manual calculation of the dice loss ####
        # L = 1 - 2 * precision * recall / (precision + recall)
        # recall = tp/(tp+fn) 
        # precision = tp/(tp+fp)
        # def.: pos = 1 
        #tp = torch.sum(output[target==1.] >= .5)
        #tn = torch.sum(output[target==0.] < .5)
        #fn = torch.sum(output[target==1.] < .5)
        #fp = torch.sum(output[target==0.] >= .5)
        #loss = 2 * tp / (2*fp + fn + fp) 

        loss = loss_fn(output, target)

        #loss = loss_fn(output, target)

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
        print(
            "Train Epoch: {} \tLoss: {:.6f}".format(
                epoch,
                np.mean(avg_loss),
            )
        )
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
    return tabulate(
        rows,
        headers=headers,
        tablefmt="fancy_grid",
    )


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

    # Segmentation - TEMPORARY 
    test_accs = []
    test_dices = []

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
            # TODO: ONLY MSD DATASET (NOT PREPROCESSED)
            #res = data.shape[-1]
            #data, target = data.view(-1, 1, res, res).to(device), target.view(-1, res, res).to(device)
            data, target = data.to(device), target.unsqueeze(dim=1).to(device)
            #dim = 256*256
            #dim_2 = int(dim/2)
            #model = nn.Sequential(
            #                    nn.Flatten(),
            #                    nn.Linear(dim, 100), 
            #                    nn.ReLU(), 
            #                    nn.Linear(100, dim), 
            #                    ).to(device)
            num_classes = 1
            model = smp.Unet("resnet18", classes=num_classes, activation="sigmoid")
            inpt_channels = 1
            if inpt_channels != 3:
                new_encoder = [nn.Conv2d(inpt_channels, 3, 1), model.encoder.conv1]
                model.encoder.conv1 = nn.Sequential(*new_encoder)
            model = model.to(device)
            output = model(data)

            #output = output.view_as(target)

            #output, target = output.cpu(), target.cpu() # for loss_fn
            #loss = loss_fn(output, oh_converter(target if oh_converter else target)

            #tp = torch.sum(output[target==1.] >= .5)
            #tn = torch.sum(output[target==0.] < .5)
            #fn = torch.sum(output[target==1.] < .5)
            #fp = torch.sum(output[target==0.] >= .5)
            #loss = 2 * tp / (2*fp + fn + fp)
            loss = loss_fn(output, target)

            test_loss += loss
            #test_loss += loss.item()  # sum up batch loss
            # Segmentation 
            # TODO: Adapt for all use-cases (train, inference, encrypted, uncencrypted)
            if args.data_dir == "seg_data": 
                # As for normal classification consider the most probable class (for every pixel)
                # the second dimension in model output is again the class-dimension
                # that's why the max should be taken over that dimension
                #_, pred = torch.max(output, 1)
                #pred = torch.round(output).type(torch.LongTensor).view(-1).cpu().numpy()

                # TODO: Mask only for MSRC
                # Only allow images/pixels with label >= 0 e.g. for segmentation 
                # (because of unlabeled datapoints with label: -1)
                #targets_mask = target >= 0
                #test_acc = np.mean((pred == target)[targets_mask].data.cpu().numpy())
                #_, target_pred = torch.max(target, 1)
                #target_pred = target.type(torch.LongTensor).view(-1).cpu().numpy()
               
                #test_acc = np.mean((pred==target_pred))
                #test_accs.append(test_acc)

                # f1-score 
                #test_dice = np.mean([mt.f1_score(tar, pred) for tar, pred in zip(target_pred.numpy(), pred.numpy())])
                #test_dice = []






                #print(pred[pred!=0])
                #print(pred.shape)
  





                #test_dice = mt.f1_score(target_pred, pred)

                ### Too inefficient ###
                #for i in range(target_pred.shape[1]): 
                #    for j in range(target_pred.shape[2]): 
                        # all classifcations for one pixel 
                #        test_dice.append(mt.f1_score(target_pred[:, i, j], pred[:, i, j]))

               #test_dices.append(test_dice)

                # Added from above (TO BE EXTENDED)
                #total_pred.append(pred)
                #total_target.append(target)

                # Make segmentation compatible with classification eval pipeline
                output = output.view(output.shape[0], -1)
                total_scores.append(output)
                target = target.view(-1).type(torch.LongTensor)
                pred = output.view(-1).round().type(torch.LongTensor)
                tgts = target
                total_pred.append(pred)
                total_target.append(tgts)
            else: 
                total_scores.append(output)
                pred = output.argmax(dim=1)
                tgts = target.view_as(pred)
                total_pred.append(pred)
                total_target.append(tgts)
                #print(output.shape)
                #print(pred.shape)
                #print(tgts.shape)
            equal = pred.eq(tgts)
            TP += (
                equal.sum().copy().get().float_precision().long().item()
                if args.encrypted_inference
                else equal.sum().item()
            )
    test_loss /= len(val_loader)
    
    #if args.data_dir == "seg_data": 
        # Segmentation - TEMPORARY 
    #    print(f"VALIDATION: Epoch: {epoch}, Val-Loss: {test_loss}, \
    #        Val-Acc.: {np.mean(test_accs)}, Dice: {np.mean(test_dices)}")  

    if args.encrypted_inference:
        objective = 100.0 * TP / (len(val_loader) * args.test_batch_size)
        L = len(val_loader.dataset)
        if verbose:
            print(
                "Test set: Epoch: {:d} Average loss: {:.4f}, Recall: {}/{} ({:.0f}%)\n".format(
                    epoch,
                    test_loss,
                    TP,
                    L,
                    objective,
                ),
                # end="",
            )
    else:
        # Segmentation: TEMPORARY
        #if args.data_dir == "seg_data": 
        #    matthews_coeff = 0
            # for now set objective to test_acc
            #objective = np.mean(test_accs)
            # for now set objective to F1-score 
        #    objective = np.mean(test_dices)
        if True: 
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
                warn(
                    "ROC AUC score could not be calculated and was set to zero.",
                    category=UserWarning,
                )
                roc_auc = 0.0
        
            #print((total_target==-1).sum())
            #print((total_target==1).sum())
            #print((total_target==0).sum())
            #print(((total_target!=0).any() and (total_target!=1).any()).sum())

            #print((total_pred==-1).sum())
            #print((total_pred==1).sum())
            #print((total_pred==0).sum())

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
