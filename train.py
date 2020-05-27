import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import configparser
import argparse
import visdom
import tqdm
import shutil
import random
import numpy as np
from os import path, remove
from warnings import warn
from datetime import datetime
from tabulate import tabulate
from torchvision import datasets, transforms, models
from torchlib.dataloader import PPPP
from torchlib.models import vgg16, resnet18, conv_at_resolution
from torchlib.utils import (
    LearningRateScheduler,
    Arguments,
    train,
    test,
    save_model,
    save_config_results,
    AddGaussianNoise,
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
        "--no_cuda", action="store_true", help="dont use a visdom server"
    )
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Start training from older model checkpoint",
    )
    parser.add_argument(
        "--websockets", action="store_true", help="train on websocket config"
    )
    cmd_args = parser.parse_args()

    config = configparser.ConfigParser()
    assert path.isfile(cmd_args.config), "config file not found"
    config.read(cmd_args.config)

    args = Arguments(cmd_args, config, mode="train")
    print(str(args))

    if args.train_federated:
        import syft as sy
        from torchlib.websocket_utils import read_websocket_config

        hook = sy.TorchHook(torch)
        hook.local_worker.is_client_worker = False
        server = hook.local_worker
        worker_dict = read_websocket_config("configs/websetting/config.csv")
        worker_names = [id_dict["id"] for _, id_dict in worker_dict.items()]

        if args.websockets:
            workers = [
                sy.workers.websocket_client.WebsocketClientWorker(
                    hook=hook,
                    id=worker["id"],
                    port=worker["port"],
                    host=worker["host"],
                )
                for row, worker in worker_dict.items()
            ]
            fed_datasets = sy.FederatedDataset(
                [
                    sy.local_worker.request_search(args.dataset, location=worker)[0]
                    for worker in workers
                ]
            )
            train_loader = sy.FederatedDataLoader(fed_datasets)
        else:
            workers = [
                sy.VirtualWorker(hook, id=id_dict["id"])
                for row, id_dict in worker_dict.items()
            ]

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if use_cuda else "cpu")  # pylint: disable=no-member

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    exp_name = "{:s}_{:s}_{:s}".format(
        "federated" if args.train_federated else "vanilla", args.dataset, timestamp
    )

    """
    Dataset creation and definition
    """
    class_names = None
    if args.dataset == "mnist":
        num_classes = 10
        train_tf = [
            transforms.Resize(args.train_resolution),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
        test_tf = [
            transforms.Resize(args.inference_resolution),
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
            test_tf.append(repeat)
        dataset = datasets.MNIST(
            "../data",
            train=True,
            download=True,
            transform=transforms.Compose(train_tf),
        )

        testset = datasets.MNIST(
            "../data", train=False, transform=transforms.Compose(test_tf),
        )
    elif args.dataset == "pneumonia":
        num_classes = 3
        """
        Different train and inference resolution only works with adaptive
        pooling in model activated
        """
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
            transforms.Normalize((0.57282609,), (0.17427578,)),
            transforms.RandomApply(
                [AddGaussianNoise(mean=0.0, std=args.noise_std)], p=args.noise_prob
            ),
        ]
        # TODO: Add normalization
        test_tf = [
            transforms.Resize(args.inference_resolution),
            transforms.CenterCrop(args.inference_resolution),
            transforms.ToTensor(),
            transforms.Normalize((0.57282609,), (0.17427578,)),
        ]

        """
        Duplicate grayscale one channel image into 3 channels
        """
        if args.pretrained or args.websockets:
            repeat = transforms.Lambda(
                lambda x: torch.repeat_interleave(  # pylint: disable=no-member
                    x, 3, dim=0
                )
            )
            train_tf.append(repeat)
            test_tf.append(repeat)
        dataset = PPPP(
            "data/Labels.csv",
            train=True,
            transform=transforms.Compose(train_tf),
            seed=args.seed,
        )
        """ Removed because bad practice
        testset = PPPP(
            "data/Labels.csv",
            train=False,
            transform=transforms.Compose(test_tf),
            seed=args.seed,
        )"""
        class_names = {0: "normal", 1: "bacterial pneumonia", 2: "viral pneumonia"}
        occurances = dataset.get_class_occurances()
    else:
        raise NotImplementedError("dataset not implemented")

    total_L = len(dataset)
    fraction = 1.0 / args.val_split
    dataset, valset = torch.utils.data.random_split(
        dataset,
        [int(round(total_L * (1.0 - fraction))), int(round(total_L * fraction))],
    )
    del total_L, fraction

    if args.train_federated:
        if not args.websockets:
            train_loader = sy.FederatedDataLoader(
                dataset.federate(tuple(workers)),
                batch_size=args.batch_size,
                shuffle=True,
                **kwargs,
            )
        """val_loader = sy.FederatedDataLoader(
            valset.federate((bob, alice, charlie)),
            batch_size=args.test_batch_size,
            shuffle=True,
            **kwargs
        )"""
    else:
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, **kwargs
        )
    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=args.test_batch_size, shuffle=False, **kwargs
    )

    """ Removed because bad practice
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size, shuffle=True, **kwargs
    )"""
    cw = None
    if args.class_weights:
        if not occurances:
            occurances = {}
            # if hasattr(dataset, "get_class_occurances"):
            #    occurances = dataset.get_class_occurances()
            # else:
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
    if args.model == "vgg16":
        model = vgg16(
            pretrained=args.pretrained,
            num_classes=num_classes,
            in_channels=3 if args.dataset == "pneumonia" else 1,
            adptpool=False,
            input_size=args.inference_resolution,
        )
    elif args.model == "simpleconv":
        if args.pretrained:
            warn("No pretrained version available")

        model = conv_at_resolution[args.train_resolution](
            num_classes=num_classes, in_channels=3 if args.dataset == "pneumonia" else 1
        )
        """if args.train_federated:
            
            data_shape = torch.ones(  # pylint: disable=no-member
                (
                    args.batch_size,
                    3 if args.dataset == "pneumonia" else 1,
                    args.train_resolution,
                    args.train_resolution,
                ),
                device=device,
            )
            print(data_shape.size())
            model.build(data_shape)"""
    elif args.model == "resnet-18":
        model = resnet18(
            pretrained=args.pretrained,
            num_classes=num_classes,
            in_channels=3 if args.dataset == "pneumonia" else 1,
            adptpool=False,
            input_size=args.inference_resolution,
        )
    else:
        raise NotImplementedError("model unknown")

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
    if args.train_federated:
        from syft.federated.floptimizer import Optims

        optimizer = Optims(worker_names, optimizer)
    loss_fn = nn.CrossEntropyLoss(weight=cw, reduction="mean")

    start_at_epoch = 1
    if cmd_args.resume_checkpoint:
        print("resuming checkpoint - args will be overwritten")
        state = torch.load(cmd_args.resume_checkpoint, map_location=device)
        start_at_epoch = state["epoch"]
        args = state["args"]
        if args.train_federated:
            opt_state_dict = state["optim_state_dict"]
            for w in worker_names:
                optimizer.get_optim(w).load_state_dict(opt_state_dict[w])
        else:
            optimizer.load_state_dict(state["optim_state_dict"])
        model.load_state_dict(state["model_state_dict"])
    model.to(device)

    test(
        args,
        model,
        device,
        val_loader,
        start_at_epoch - 1,
        loss_fn,
        num_classes,
        vis_params=vis_params,
        class_names=class_names,
    )
    accuracies = []
    model_paths = []
    for epoch in range(start_at_epoch, args.epochs + 1):
        if args.train_federated:
            for w in worker_names:
                new_lr = scheduler.adjust_learning_rate(
                    optimizer.get_optim(w), epoch - 1
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
            _, acc = test(
                args,
                model,
                device,
                val_loader,
                epoch,
                loss_fn,
                num_classes=num_classes,
                vis_params=vis_params,
                class_names=class_names,
            )
            model_path = "model_weights/{:s}_epoch_{:03d}.pt".format(exp_name, epoch,)

            save_model(model, optimizer, model_path, args, epoch)
            accuracies.append(acc)
            model_paths.append(model_path)
    # reversal and formula because we want last occurance of highest value
    accuracies = np.array(accuracies)[::-1]
    am = np.argmax(accuracies)
    highest_acc = len(accuracies) - am - 1
    best_epoch = highest_acc * args.test_interval
    best_model_file = model_paths[highest_acc]
    print(
        "Highest accuracy was {:.1f}% in epoch {:d}".format(accuracies[am], best_epoch)
    )
    # load best model on val set
    state = torch.load(best_model_file, map_location=device)
    model.load_state_dict(state["model_state_dict"])

    """_, result = test(
        args,
        model,
        device,
        val_loader,
        args.epochs + 1,
        loss_fn,
        num_classes=num_classes,
        vis_params=vis_params,
        class_names=class_names,
    )
    print('result: {:.1f} - best accuracy: {:.1f}'.format(result, accuracies[am]))"""
    shutil.copyfile(
        best_model_file, "model_weights/final_{:s}.pt".format(exp_name),
    )
    save_config_results(
        args, accuracies[am], timestamp, "model_weights/completed_trainings.csv"
    )

    # delete old model weights
    for model_file in model_paths:
        remove(model_file)
    """save_model(
        model,
        optimizer,
        "model_weights/{:s}_{:s}_final.pt".format(
            "federated" if args.train_federated else "vanilla", args.dataset
        ),
    )"""
    if args.train_federated:
        for w in workers:
            w.close()
