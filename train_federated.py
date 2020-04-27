import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import configparser
import argparse
import visdom
import tqdm
import numpy as np
from tabulate import tabulate
from torchvision import datasets, transforms, models
from common.dataloader import PPPP
from common.models import vgg16, Net
from common.utils import LearningRateScheduler


def train(args, model, device, train_loader, optimizer, epoch, class_weights=None):
    model.train()
    avg_loss = []
    for batch_idx, (data, target) in tqdm.tqdm(
        enumerate(train_loader), leave=False, desc="training", total=L
    ):  # <-- now it is a distributed dataset
        if args.train_federated:
            model.send(data.location)  # <-- NEW: send the model to the right location
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target, weight=class_weights)
        loss.backward()
        optimizer.step()
        if args.train_federated:
            model.get()  # <-- NEW: get the model back
        if batch_idx % args.log_interval == 0:
            if args.train_federated:
                loss = loss.get()  # <-- NEW: get the loss back

            if args.visdom:
                vis.line(
                    X=np.asarray([epoch + float(batch_idx) * div - 1]),
                    Y=np.asarray([loss.item()]),
                    win="loss_win",
                    name="train_loss",
                    update="append",
                    env=vis_env,
                )
            else:
                avg_loss.append(loss.item)
    if not args.visdom:
        print(
            "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch,
                batch_idx * args.batch_size,
                len(train_loader)
                * args.batch_size,  # batch_idx * len(data), len(train_loader.dataset),
                100.0 * batch_idx / len(train_loader),
                np.mean(avg_loss),
            )
        )


def test(args, model, device, test_loader, epoch, num_classes):
    model.eval()
    test_loss = 0
    correct = 0
    correct_per_class = {}
    incorrect_per_class = {}
    for i in range(num_classes):
        correct_per_class[i] = 0
        incorrect_per_class[i] = 0
    with torch.no_grad():
        for data, target in tqdm.tqdm(
            test_loader, total=len(test_loader), desc="testing", leave=False
        ):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                1, keepdim=True
            )  # get the index of the max log-probability
            tgts = target.view_as(pred)
            equal = pred.eq(tgts)
            for i, t in enumerate(tgts):
                t = t.item()
                if equal[i]:
                    correct_per_class[t] += 1
                else:
                    incorrect_per_class[t] += 1
            correct += equal.sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Epoch: {:d} Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            epoch,
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        ),
        end="",
    )
    rows = []
    dataset = test_loader.dataset
    for i, v in correct_per_class.items():
        total = v + incorrect_per_class[i]
        rows.append(
            [
                dataset.get_class_name(i) if hasattr(dataset, "get_class_name") else i,
                "{:.1f} %".format(100.0 * (v / float(total))),
                v,
                total,
            ]
        )
    rows.append(
        [
            "Total",
            "{:.1f} %".format(100.0 * correct / len(test_loader.dataset)),
            correct,
            len(test_loader.dataset),
        ]
    )
    print(tabulate(rows, headers=["Class", "Accuracy", "n correct", "n total"]))
    if args.visdom:
        vis.line(
            X=np.asarray([epoch]),
            Y=np.asarray([test_loss]),
            win="loss_win",
            name="val_loss",
            update="append",
            env=vis_env,
        )


def save_model(model, optim, path):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optim.state_dict(),
        },
        path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        default="configs/pneumonia.ini",
        help="Path to config",
    )
    parser.add_argument(
        "--train_federated", action="store_true", help="Train in federated setting"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="pneumonia",
        choices=["pneumonia", "mnist"],
        help="which dataset?",
    )
    parser.add_argument(
        "--no_visdom", action="store_false", help="dont use a visdom server"
    )
    parser.add_argument(
        "--start_at_epoch",
        type=int,
        default=1,
        help="At which epoch should the training start?",
    )
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Start training from older model checkpoint",
    )
    cmd_args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(cmd_args.config)

    class Arguments:
        def __init__(self, cmd_args, config):
            self.batch_size = config.getint("config", "batch_size", fallback=1)
            self.test_batch_size = config.getint(
                "config", "test_batch_size", fallback=1
            )
            self.resolution = config.getint("config", "resolution", fallback=224)
            self.epochs = config.getint("config", "epochs", fallback=1)
            self.lr = config.getfloat("config", "lr", fallback=1e-3)
            self.end_lr = config.getfloat("config", "end_lr", fallback=self.lr)
            self.momentum = config.getfloat("config", "momentum", fallback=0.5)
            self.no_cuda = config.getboolean("config", "no_cuda", fallback=False)
            self.seed = config.getint("config", "seed", fallback=1)
            self.test_interval = config.getint("config", "test_interval", fallback=1)
            self.log_interval = config.getint("config", "log_interval", fallback=10)
            self.save_interval = config.getint("config", "save_interval", fallback=10)
            self.save_model = config.getboolean("config", "save_model", fallback=False)
            self.train_federated = cmd_args.train_federated
            # print('Train federated: {0}'.format(self.train_federated))
            self.dataset = cmd_args.dataset  # options: ['pneumonia', 'mnist']
            self.visdom = cmd_args.no_visdom
            self.optimizer = config.get("config", "optimizer", fallback="SGD")
            assert self.optimizer in ["SGD", "Adam"], "Unknown optimizer"
            if self.optimizer == "Adam":
                self.beta1 = config.getfloat("config", "beta1", fallback=0.9)
                self.beta2 = config.getfloat("config", "beta2", fallback=0.999)
            self.weight_decay = config.getfloat("config", "weight_decay", fallback=0.0)
            self.class_weights = config.getboolean(
                "config", "weight_classes", fallback=False
            )

    args = Arguments(cmd_args, config)

    if args.train_federated:
        import syft as sy

        hook = sy.TorchHook(
            torch
        )  # <-- NEW: hook PyTorch ie add extra functionalities to support Federated Learning
        bob = sy.VirtualWorker(hook, id="bob")  # <-- NEW: define remote worker bob
        alice = sy.VirtualWorker(hook, id="alice")  # <-- NEW: and alice
        charlie = sy.VirtualWorker(hook, id="charlie")  # <-- NEW: and alice

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")  # pylint: disable=no-member

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    if args.dataset == "mnist":
        num_classes = 10
        dataset = datasets.MNIST(
            "../data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Resize(56),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        )
        testset = datasets.MNIST(
            "../data",
            train=False,
            transform=transforms.Compose(
                [
                    transforms.Resize(56),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        )
    elif args.dataset == "pneumonia":
        num_classes = 3
        tf = transforms.Compose(
            [
                transforms.Resize(args.resolution),
                transforms.CenterCrop(args.resolution),
                transforms.ToTensor(),
            ]
        )  # TODO: Add normalization
        dataset = PPPP("Labels.csv", train=True, transform=tf)
        testset = PPPP("Labels.csv", train=False, transform=tf)
    else:
        raise NotImplementedError("dataset not implemented")

    if args.train_federated:
        train_loader = sy.FederatedDataLoader(
            dataset.federate((bob, alice, charlie)),
            batch_size=args.batch_size,
            shuffle=True,
            **kwargs
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, **kwargs
        )

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size, shuffle=True, **kwargs
    )
    L = len(train_loader)
    cw = None
    if args.class_weights:
        occurances = {}
        if hasattr(dataset, "get_class_occurances"):
            occurances = dataset.get_class_occurances()
        else:
            for _, c in tqdm.tqdm(
                dataset, total=len(dataset), leave=False, desc="calc class weights"
            ):
                if c.item() in occurances:
                    occurances[c] += 1
                else:
                    occurances[c] = 1
        cw = torch.zeros((len(occurances)))  # pylint: disable=no-member
        for c, n in occurances.items():
            cw[c] = 1.0 / float(n)
        cw /= torch.sum(cw)  # pylint: disable=no-member

    scheduler = LearningRateScheduler(
        args.epochs, np.log10(args.lr), np.log10(args.end_lr)
    )

    ## visdom
    if args.visdom:
        vis = visdom.Visdom()
        assert vis.check_connection(
            timeout_seconds=3
        ), "No connection could be formed quickly"
        vis_env = "{:s}/{:s}".format(
            "federated" if args.train_federated else "vanilla", args.dataset
        )
        plt_dict = dict(
            name="training loss",
            ytickmax=10,
            xlabel="epoch",
            ylabel="loss",
            legend=["train_loss"],
        )
        vis.line(
            X=np.zeros((1, 2)),
            Y=np.zeros((1, 2)),
            win="loss_win",
            opts={
                "legend": ["train_loss", "val_loss"],
                "xlabel": "epochs",
                "ylabel": "loss",
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
        div = 1.0 / float(L)
    # model = Net().to(device)
    model = vgg16(pretrained=False, num_classes=num_classes, in_channels=1)
    # model = models.vgg16(pretrained=False, num_classes=3)
    # model.classifier = vggclassifier()
    model.to(device)
    if args.optimizer == "SGD":
        optimizer = optim.SGD(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        )  # TODO momentum is not supported at the moment
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )
    else:
        raise NotImplementedError("optimization not implemented")

    if cmd_args.resume_checkpoint:
        state = torch.load(cmd_args.resume_checkpoint)
        if "optim_state_dict" in state:
            optimizer.load_state_dict(state["optim_state_dict"])
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
    test(
        args,
        model,
        device,
        test_loader,
        cmd_args.start_at_epoch - 1,
        num_classes=num_classes,
    )
    for epoch in range(cmd_args.start_at_epoch, args.epochs + 1):
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
        train(args, model, device, train_loader, optimizer, epoch)
        if (epoch % args.test_interval) == 0:
            test(args, model, device, test_loader, epoch, num_classes=num_classes)

        if args.save_model and (epoch % args.save_interval) == 0:
            save_model(
                model,
                optimizer,
                "model_weights/{:s}_chestxray_epoch_{:03d}.pt".format(
                    "federated" if args.train_federated else "vanilla", epoch
                ),
            )

    if args.save_model:
        save_model(
            model,
            optimizer,
            "model_weights/{:s}_chestxray_final.pt".format(
                "federated" if args.train_federated else "vanilla"
            ),
        )
