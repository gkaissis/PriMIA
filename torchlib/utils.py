import torch
import tqdm
import numpy as np
import pandas as pd
from os.path import isfile
from tabulate import tabulate


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
    def __init__(self, cmd_args, config, mode="train"):
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
        self.visdom = cmd_args.no_visdom if mode == "train" else False
        self.encrypted_inference = (
            cmd_args.encrypted_inference if mode == "inference" else False
        )
        self.dataset = cmd_args.dataset  # options: ['pneumonia', 'mnist']
        self.no_cuda = cmd_args.no_cuda

    def from_previous_checkpoint(self, cmd_args):
        self.visdom = False
        self.encrypted_inference = cmd_args.encrypted_inference
        self.no_cuda = cmd_args.no_cuda

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


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return (
            tensor
            + torch.randn(tensor.size()) * self.std  # pylint: disable=no-member
            + self.mean
        )

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
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


def train(
    args, model, device, train_loader, optimizer, epoch, loss_fn, vis_params=None
):
    model.train()
    L = len(train_loader)
    div = 1.0 / float(L)
    if not args.train_federated:
        opt = optimizer

    avg_loss = []
    for batch_idx, (data, target) in tqdm.tqdm(
        enumerate(train_loader), leave=False, desc="training", total=L
    ):  # <-- now it is a distributed dataset
        if args.train_federated:
            model.send(data.location)  # <-- NEW: send the model to the right location
            opt = optimizer.get_optim(data.location.id)
            # print("data location: {:s}".format(str(data.location)))
            # print("target location: {:s}".format(str(target.location)))
            # print("model location: {:s}".format(str(model.location)))
            # model = model.to(device)

        data, target = data.to(device), target.to(device)
        opt.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        opt.step()
        if args.train_federated:
            model.get()  # <-- NEW: get the model back
        if batch_idx % args.log_interval == 0:
            if args.train_federated:
                loss = loss.get()  # <-- NEW: get the loss back

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


def test(
    args,
    model,
    device,
    test_loader,
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
    for i in range(num_classes):
        tp_per_class[i] = 0
        fp_per_class[i] = 0
        fn_per_class[i] = 0
    with torch.no_grad():
        for data, target in tqdm.tqdm(
            test_loader, total=len(test_loader), desc="testing", leave=False
        ):
            if not args.encrypted_inference:
                data = data.to(device)
                target = target.to(device)
            # print(model)
            # exit()
            output = model(data)
            test_loss += loss_fn(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1)
            tgts = target.view_as(pred)
            equal = pred.eq(tgts)
            if args.encrypted_inference:
                TP += equal.sum().copy().get().float_precision().long().item()
            else:
                for i, t in enumerate(tgts):
                    t = t.item()
                    if equal[i]:
                        tp_per_class[t] += 1
                    else:
                        fn_per_class[t] += 1
                for i, p in enumerate(pred):
                    p = p.item()
                    if not equal[i]:
                        fp_per_class[p] += 1
                TP += equal.sum().item()

    test_loss /= len(test_loader)
    objective = 100.0 * TP / (len(test_loader) * args.test_batch_size)
    print(
        "Test set: Epoch: {:d} Average loss: {:.4f}, Recall: {}/{} ({:.0f}%)\n".format(
            epoch, test_loss, TP, len(test_loader.dataset), objective,
        ),
        # end="",
    )
    if not args.encrypted_inference:
        rows = []
        tn_per_class = {}
        accs, recs, precs, f1s, total_per_class = [], [], [], [], []
        total_items = len(test_loader.dataset)
        for i, fp in fp_per_class.items():
            total = tp_per_class[i] + fn_per_class[i]
            total_per_class.append(total)
            tn_per_class[i] = total_items - total - fp
        # dataset = test_loader.dataset
        for i, tp in tp_per_class.items():
            total = tp + fn_per_class[i]
            acc = 100.0 * (tp + tn_per_class[i]) / float(total_items)
            rec = 100.0 * (tp / float(total))
            prec = (
                100.0 * (tp / float(tp + fp_per_class[i]))
                if tp + fp_per_class[i]
                else float("NaN")
            )
            f1_score = (2 * prec * rec) / (prec + rec) if prec + rec > 0 else 0
            accs.append(acc)
            recs.append(rec)
            precs.append(prec)
            f1s.append(f1_score)
            rows.append(
                [
                    class_names[i] if class_names else i,
                    "{:.1f} %".format(acc),
                    "{:.1f} %".format(rec),
                    "{:.1f} %".format(prec),
                    "{:.1f} %".format(f1_score),
                    total,
                    tp,
                    tn_per_class[i],
                    fp_per_class[i],
                    fn_per_class[i],
                ]
            )
        """
        Measures can be weighted s.t. larger classes have higher impact.
        It is currently not used as it makes sense to try to have all classes evenly good not depending on the class distribution
        weights = np.array(total_per_class, dtype=np.float64)
        weights *= 1./np.sum(weights)
        """
        precs = np.array(precs)
        # indices = ~np.isnan(precs)
        rec = np.average(recs)
        prec = np.average(precs)  # [indices])
        f1_score = np.average(f1s)
        rows.append(
            [
                "Mean / Total",
                "{:.1f} %".format(
                    np.average(accs)  # , weights=weights)
                ),  # "{:.1f} %".format(
                # 100.0 * ((sum(tp_per_class.values()) + sum(tn_per_class.values()))
                # / total_items)
                # ),
                "{:.1f} %".format(rec),  # weights=weights)),  # recall),
                "{:.1f} %".format(prec),  # weights=weights[indices])),
                # 100.0 * (TP / float(TP + sum(fp_per_class.values())))
                # if TP + sum(fp_per_class.values())
                # else float("NaN")
                # ),"""
                "{:.1f} %".format(f1_score),
                total_items,
                TP,
                "N/A",  # "{:d}".format(sum(tn_per_class.values())),
                "{:d}".format(sum(fp_per_class.values())),
                "{:d}".format(sum(fn_per_class.values())),
            ]
        )
        objective = np.average(accs)  # , weights=weights)
        print(
            tabulate(
                rows,
                headers=[
                    "Class",
                    "Accuracy",
                    "Recall",
                    "Precision",
                    "F1 score",
                    "n total",
                    "TP",
                    "TN",
                    "FP",
                    "FN",
                ],
                tablefmt="fancy_grid",
            )
        )
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
