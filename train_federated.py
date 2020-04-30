import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import configparser
import argparse
import visdom
import tqdm
from os import path
import numpy as np
from tabulate import tabulate
from torchvision import datasets, transforms, models
from torchlib.dataloader import PPPP
from torchlib.models import vgg16, resnet18, conv_at_resolution
from torchlib.utils import LearningRateScheduler, Arguments, train, test, save_model


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
        "--no_cuda", action="store_true", help="dont use a visdom server"
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
    assert path.isfile(cmd_args.config), "config file not found"
    config.read(cmd_args.config)

    args = Arguments(cmd_args, config, mode='train')

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
                    transforms.Resize(args.train_resolution),
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
                    transforms.Resize(args.inference_resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        )
    elif args.dataset == "pneumonia":
        num_classes = 3
        train_tf = transforms.Compose(
            [
                transforms.Resize(args.inference_resolution),
                transforms.RandomCrop(args.train_resolution),
                transforms.ToTensor(),
            ]
        )  # TODO: Add normalization
        test_tf = transforms.Compose(
            [
                transforms.Resize(args.inference_resolution),
                transforms.CenterCrop(args.inference_resolution),
                transforms.ToTensor(),
            ]
        )
        dataset = PPPP("data/Labels.csv", train=True, transform=train_tf)
        testset = PPPP("data/Labels.csv", train=False, transform=test_tf)
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
            X=np.zeros((1, 3)),
            Y=np.zeros((1, 3)),
            win="loss_win",
            opts={
                "legend": ["train_loss", "val_loss", "accuracy"],
                "xlabel": "epochs",
                "ylabel": "loss / accuracy [%]",
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
    # model = Net().to(device)
    if args.model == 'vgg16':
        model = vgg16(pretrained=False, num_classes=num_classes, in_channels=1, adptpool=False)
    elif args.model == 'simpleconv':
        model = conv_at_resolution[args.train_resolution](num_classes=num_classes)
    elif args.model == 'resnet-18':
        model = resnet18(pretrained=False, num_classes=num_classes, in_channels=1, adptpool=False)
    else:
        raise NotImplementedError('model unknown')
    # model = resnet18(pretrained=False, num_classes=num_classes, in_channels=1)
    # model = models.vgg16(pretrained=False, num_classes=3)
    # model.classifier = vggclassifier()
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
    loss_fn = nn.CrossEntropyLoss(weight=cw, reduction="mean")

    if cmd_args.resume_checkpoint:
        state = torch.load(cmd_args.resume_checkpoint, map_location=device)
        if "optim_state_dict" in state:
            optimizer.load_state_dict(state["optim_state_dict"])
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
    model.to(device)
        
    test(
        args,
        model,
        device,
        test_loader,
        cmd_args.start_at_epoch - 1,
        loss_fn,
        num_classes,
        vis_params=vis_params,
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
        train(
            args,
            model,
            device,
            train_loader,
            optimizer,
            epoch,
            loss_fn,
            vis_params=vis_params,
        )
        if (epoch % args.test_interval) == 0:
            test(
                args,
                model,
                device,
                test_loader,
                epoch,
                loss_fn,
                num_classes=num_classes,
                vis_params=vis_params,
            )

        if args.save_model and (epoch % args.save_interval) == 0:
            save_model(
                model,
                optimizer,
                "model_weights/{:s}_{:s}_epoch_{:03d}.pt".format(
                    "federated" if args.train_federated else "vanilla",
                    args.dataset,
                    epoch,
                ),
            )

    if args.save_model:
        save_model(
            model,
            optimizer,
            "model_weights/{:s}_{:s}_final.pt".format(
                "federated" if args.train_federated else "vanilla", args.dataset
            ),
        )
