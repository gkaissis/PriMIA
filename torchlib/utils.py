import torch
import tqdm
import numpy as np
import pandas as pd
import multiprocessing as mp
from time import sleep
from random import random
from syft.frameworks.torch.fl.utils import scale_model, add_model
from os.path import isfile
from tabulate import tabulate
from sklearn import metrics as mt


"""
Available schedule plans:
log_linear : Linear interpolation with log learning rate scale
log_cosine : Cosine interpolation with log learning rate scale
"""


class LearningRateScheduler:
    def __init__(
        self,
        total_epochs,
        log_start_lr,
        log_end_lr,
        schedule_plan="log_linear",
        restarts=None,
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

    def get_lr(self, epoch):
        epoch = epoch % self.total_epochs
        if (type(epoch) is int and epoch > self.total_epochs) or (
            type(epoch) is np.ndarray and np.max(epoch) > self.total_epochs
        ):
            raise AssertionError("Requested epoch out of precalculated schedule")
        return self.calc_lr(epoch)

    def adjust_learning_rate(self, optimizer, epoch):
        new_lr = self.get_lr(epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr
        return new_lr


class Arguments:
    def __init__(self, cmd_args, config, mode="train", verbose=True):
        assert mode in ["train", "inference"], "no other mode known"
        self.batch_size = config.getint("config", "batch_size", fallback=1)
        self.test_batch_size = config.getint("config", "test_batch_size", fallback=1)
        self.train_resolution = config.getint(
            "config", "train_resolution", fallback=224
        )
        self.inference_resolution = config.getint(
            "config", "inference_resolution", fallback=self.train_resolution
        )
        self.val_split = config.getint("config", "validation_split", fallback=10)
        self.epochs = config.getint("config", "epochs", fallback=1)
        self.lr = config.getfloat("config", "lr", fallback=1e-3)
        self.end_lr = config.getfloat("config", "end_lr", fallback=self.lr)
        self.restarts = config.getint("config", "restarts", fallback=None)
        self.momentum = config.getfloat("config", "momentum", fallback=0.5)
        self.seed = config.getint("config", "seed", fallback=1)
        self.test_interval = config.getint("config", "test_interval", fallback=1)
        self.log_interval = config.getint("config", "log_interval", fallback=10)
        # self.save_interval = config.getint("config", "save_interval", fallback=10)
        # self.save_model = config.getboolean("config", "save_model", fallback=False)
        self.optimizer = config.get("config", "optimizer", fallback="SGD")
        assert self.optimizer in ["SGD", "Adam"], "Unknown optimizer"
        if self.optimizer == "Adam":
            self.beta1 = config.getfloat("config", "beta1", fallback=0.9)
            self.beta2 = config.getfloat("config", "beta2", fallback=0.999)
        self.model = config.get("config", "model", fallback="simpleconv")
        assert self.model in ["simpleconv", "resnet-18", "vgg16"]
        self.pretrained = config.getboolean("config", "pretrained", fallback=False)
        self.weight_decay = config.getfloat("config", "weight_decay", fallback=0.0)
        self.class_weights = config.getboolean(
            "config", "weight_classes", fallback=False
        )
        self.vertical_flip_prob = config.getfloat(
            "augmentation", "vertical_flip_prob", fallback=0.0
        )
        self.rotation = config.getfloat("augmentation", "rotation", fallback=0.0)
        self.translate = config.getfloat("augmentation", "translate", fallback=0.0)
        self.scale = config.getfloat("augmentation", "scale", fallback=0.0)
        self.shear = config.getfloat("augmentation", "shear", fallback=0.0)
        self.noise_std = config.getfloat("augmentation", "noise_std", fallback=1.0)
        self.noise_prob = config.getfloat("augmentation", "noise_prob", fallback=0.0)
        self.train_federated = cmd_args.train_federated if mode == "train" else False
        if self.train_federated:
            self.sync_every_n_batch = config.getint(
                "federated", "sync_every_n_batch", fallback=10
            )
            self.wait_interval = config.getfloat(
                "federated", "wait_interval", fallback=0.1
            )
            self.keep_optim_dict = config.getboolean(
                "federated", "keep_optim_dict", fallback=False
            )
            self.repetitions_dataset = config.getint(
                "federated", "repetitions_dataset", fallback=1
            )
            if self.repetitions_dataset > 1:
                self.epochs = int(self.epochs / self.repetitions_dataset)
                if verbose:
                    print(
                        "Number of epochs was decreased to "
                        "{:d} because of {:d} repetitions of dataset".format(
                            self.epochs, self.repetitions_dataset
                        )
                    )

        self.visdom = cmd_args.no_visdom if mode == "train" else False
        self.encrypted_inference = (
            cmd_args.encrypted_inference if mode == "inference" else False
        )
        self.dataset = cmd_args.dataset  # options: ['pneumonia', 'mnist']
        self.no_cuda = cmd_args.no_cuda
        self.websockets = cmd_args.websockets if mode == "train" else False
        if self.websockets:
            assert self.train_federated, "If you use websockets it must be federated"

    def from_previous_checkpoint(self, cmd_args):
        self.visdom = False
        self.encrypted_inference = cmd_args.encrypted_inference
        self.no_cuda = cmd_args.no_cuda
        self.websockets = (
            cmd_args.websockets  # currently not implemented for inference
            if self.encrypted_inference
            else False
        )

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
    def __init__(self, mean=0.0, std=1.0, p=None):
        super(AddGaussianNoise, self).__init__()
        self.std = std
        self.mean = mean
        self.p = p

    def forward(self, tensor):
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


def save_config_results(args, accuracy, timestamp, table):
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
    new_row["best_validation_accuracy"] = accuracy
    df = df.append(new_row, ignore_index=True)
    df.to_csv(table, index=False)


## Adaption of federated averaging from syft with option of weights
def federated_avg(models, weights: torch.Tensor = None):
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
        for id, partial_model in models.items():
            scaled_model = scale_model(partial_model, weights[id])
            if model:
                model = add_model(model, scaled_model)
            else:
                model = scaled_model
    else:
        nr_models = len(models)
        model_list = list(models.values())
        model = model_list[0]
        for i in range(1, nr_models):
            model = add_model(model, model_list[i])
        model = scale_model(model, (1.0 / nr_models))
    return model


def training_animation(done, message="training"):
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


## Assuming train loaders is dictionary with {worker : train_loader}
def train_federated(
    args, model, device, train_loaders, optimizer, epoch, loss_fn, vis_params=None,
):
    model.train()
    mng = mp.Manager()
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
        waiting_for_sync_dict[worker.id] = False
        progress_dict[worker.id] = (0, len(train_loaders[worker]))

    total_batches = 0
    weights = []
    w_dict = {}
    for id, (_, batches) in progress_dict.items():
        total_batches += batches
        weights.append(batches)
    weights = np.array(weights) / total_batches
    for weight, id in zip(weights, progress_dict.keys()):
        w_dict[id] = weight
    jobs = [
        mp.Process(
            target=train_on_server,
            args=(
                args,
                model,
                worker,
                device,
                train_loader,
                optimizer,
                epoch,
                loss_fn,
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
        target=synchronizer,
        args=(
            result_dict,
            waiting_for_sync_dict,
            sync_dict,
            stop_sync,
            sync_completed,
            w_dict,
        ),
        kwargs={"wait_interval": args.wait_interval},
    )
    synchronize.start()
    done = mng.Value("i", False)
    animate = mp.Process(target=progress_animation, args=(done, progress_dict))
    animate.start()
    for j in jobs:
        j.join()
    stop_sync.value = True
    synchronize.join()
    done.value = True
    animate.join()

    model = sync_dict["model"]
    avg_loss = np.average(loss_dict.values(), weights=weights)

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
        print("Train Epoch: {} \tLoss: {:.6f}".format(epoch, avg_loss,))
    return model


def synchronizer(
    result_dict,
    waiting_for_sync_dict,
    sync_dict,
    stop,
    sync_completed,
    weights,
    wait_interval=0.1,
):
    while not stop.value:
        while not all(waiting_for_sync_dict.values()) and not stop.value:
            sleep(0.1)
        # print("synchronizing: models from {:s}".format(str(result_dict.keys())))
        if len(result_dict) == 1:
            for model in result_dict.values():
                sync_dict["model"] = model
        elif len(result_dict) == 0:
            return None
        else:
            models = {}
            for id, worker_model in result_dict.items():
                models[id] = worker_model
            ## could be weighted here
            ## just add weights=weights
            avg_model = federated_avg(models)
            sync_dict["model"] = avg_model
        for k in waiting_for_sync_dict.keys():
            waiting_for_sync_dict[k] = False
        ## In theory we should clear the models here
        ## However, if one worker has more samples than any other worker,
        ## but has imbalanced data this destroys our training.
        ## By keeping the last model of each worker in the dict,
        ## we still assure that it's training is not lost
        # result_dict.clear()
        sync_completed.value = True


def train_on_server(
    args,
    model,
    worker,
    device,
    train_loader,
    optim,
    epoch,
    loss_fn,
    result_dict,
    waiting_for_sync_dict,
    sync_dict,
    sync_completed,
    progress_dict,
    loss_dict,
):
    optimizer = optim.get_optim(worker.id)
    avg_loss = []
    model.send(worker)
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
    result_dict[worker.id] = model
    loss_dict[worker.id] = np.mean(avg_loss)
    progress_dict[worker.id] = (batch_idx + 1, L)
    del waiting_for_sync_dict[worker.id]
    optim.optimizers[worker.id] = optimizer


def train(
    args, model, device, train_loader, optimizer, epoch, loss_fn, vis_params=None
):
    model.train()
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
    if not args.visdom:
        print("Train Epoch: {} \tLoss: {:.6f}".format(epoch, np.mean(avg_loss),))
    return model


def test(
    args,
    model,
    device,
    val_loader,
    epoch,
    loss_fn,
    num_classes,
    vis_params=None,
    class_names=None,
):
    model.eval()
    test_loss = 0
    TP = 0
    tp_per_class = {}
    fn_per_class = {}
    fp_per_class = {}
    total_pred, total_target = [], []
    for i in range(num_classes):
        tp_per_class[i] = 0
        fp_per_class[i] = 0
        fn_per_class[i] = 0
    with torch.no_grad():
        for data, target in tqdm.tqdm(
            val_loader,
            total=len(val_loader),
            desc="testing epoch {:d}".format(epoch),
            leave=False,
        ):
            if not args.encrypted_inference:
                data = data.to(device)
                target = target.to(device)
            output = model(data)
            loss = loss_fn(output, target)
            test_loss += loss.item()  # sum up batch loss
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
    objective = 100.0 * TP / (len(val_loader) * args.test_batch_size)
    L = len(val_loader.dataset)
    print(
        "Test set: Epoch: {:d} Average loss: {:.4f}, Recall: {}/{} ({:.0f}%)\n".format(
            epoch, test_loss, TP, L, objective,
        ),
        # end="",
    )
    if not args.encrypted_inference:
        total_pred = torch.cat(total_pred).cpu().numpy()  # pylint: disable=no-member
        total_target = (
            torch.cat(total_target).cpu().numpy()  # pylint: disable=no-member
        )
        conf_matrix = mt.confusion_matrix(total_target, total_pred)
        report = mt.classification_report(
            total_target, total_pred, output_dict=True, zero_division=0
        )
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
        rows.append(
            ["Micro accuracy", "{:.1f} %".format(report["accuracy"] * 100.0),]
        )
        objective = report["accuracy"]
        headers = [
            "Class",
            "Recall",
            "Precision",
            "F1 score",
            "n total",
        ]
        headers.extend(
            [class_names[i] if class_names else i for i in range(conf_matrix.shape[0])]
        )
        print(tabulate(rows, headers=headers, tablefmt="fancy_grid",))
        if args.visdom:
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
                name="accuracy",
                update="append",
                env=vis_params["vis_env"],
            )

    return test_loss, objective


def save_model(model, optim, path, args, epoch):
    opt_state_dict = (
        {name: optim.get_optim(name).state_dict() for name in optim.workers}
        if args.train_federated
        else optim.state_dict()
    )
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optim_state_dict": opt_state_dict,
            "args": args,
        },
        path,
    )
