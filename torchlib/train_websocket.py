import logging
import configparser
import argparse
import sys
import asyncio
import visdom
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from random import seed
from datetime import datetime

import syft as sy
from syft.workers import websocket_client
from syft.frameworks.torch.fl import utils

from os import path

sys.path.append(path.abspath(path.join(path.dirname(__file__), path.pardir)))
from torchlib.websocket_utils import read_websocket_config
from torchlib.dataloader import PPPP
from torchlib.models import vgg16, resnet18, conv_at_resolution
from torchlib.utils import (
    LearningRateScheduler,
    Arguments,
    test,
    save_model,
    save_config_results,
)

LOG_INTERVAL = 25
logger = logging.getLogger("run_websocket_client")


# Loss function
"""@torch.jit.script
def loss_fn(pred, target):
    log_probabilities = torch.log(  # pylint: disable=no-member
        pred.exp() / (pred.exp().sum(-1)).unsqueeze(-1)
    
    return torch.mean(  # pylint: disable=no-member
        -torch.ones((10,)).index_select(0, target)  # pylint: disable=no-member
        * log_probabilities.index_select(-1, target).diag()
    )
    #return F.nll_loss(input=log_probabilities, target=target)"""


@torch.jit.script
def loss_fn(pred, target):
    # return F.nll_loss(input=pred, target=target)
    return F.cross_entropy(input=pred, target=target)


def define_and_get_arguments(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        default="configs/pneumonia.ini",
        help="Path to config",
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
    cmd_args = parser.parse_args()
    cmd_args_dict = vars(cmd_args)
    cmd_args_dict["train_federated"] = True

    config = configparser.ConfigParser()
    assert path.isfile(cmd_args.config), "config file not found"
    config.read(cmd_args.config)

    args = Arguments(cmd_args, config, mode="train")
    return args


async def fit_model_on_worker(
    worker: websocket_client.WebsocketClientWorker,
    traced_model: torch.jit.ScriptModule,
    batch_size: int,
    curr_round: int,
    max_nr_batches: int,
    lr: float,
    args: Arguments,
    device: str,
):
    """Send the model to the worker and fit the model on the worker's training data.

    Args:
        worker: Remote location, where the model shall be trained.
        traced_model: Model which shall be trained.
        batch_size: Batch size of each training step.
        curr_round: Index of the current training round (for logging purposes).
        max_nr_batches: If > 0, training on worker will stop at min(max_nr_batches, nr_available_batches).
        lr: Learning rate of each training step.

    Returns:
        A tuple containing:
            * worker_id: Union[int, str], id of the worker.
            * improved model: torch.jit.ScriptModule, model after training at the worker.
            * loss: Loss on last training batch, torch.tensor.
    """
    opt_args = {"lr": lr}
    if args.optimizer == "Adam":
        opt_args["betas"] = (args.beta1, args.beta2)
    train_config = sy.TrainConfig(
        model=traced_model,
        loss_fn=loss_fn,
        batch_size=batch_size,
        shuffle=False,
        max_nr_batches=-1,
        epochs=1,
        optimizer=args.optimizer,
        optimizer_args=opt_args,
    )
    train_config.send(worker)
    loss = await worker.async_fit(
        dataset_key=args.dataset, return_ids=[0], device=device
    )
    model = train_config.model_ptr.get().obj
    return worker.id, model, loss


async def main():
    args = define_and_get_arguments()

    hook = sy.TorchHook(torch)

    kwargs_websocket = {
        "hook": hook,
        "verbose": True,
    }
    worker_dict = read_websocket_config("configs/websetting/config.csv")
    worker_instances = [
        websocket_client.WebsocketClientWorker(
            id=worker["id"],
            port=worker["port"],
            host=worker["host"],
            **kwargs_websocket
        )
        for row, worker in worker_dict.items()
    ]
    fed_datasets = sy.FederatedDataset([
        sy.local_worker.request_search("mnist", location=worker)[0]
        for worker in worker_instances
    ])
    dataloader = sy.FederatedDataLoader(fed_datasets)
    exit()

    for wcw in worker_instances:
        wcw.clear_objects_remote()

    # worker_instances = [alice, bob, charlie]

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")  # pylint:disable=no-member
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    exp_name = "{:s}_{:s}_{:s}".format(
        "federated" if args.train_federated else "vanilla", args.dataset, timestamp
    )
    scheduler = LearningRateScheduler(
        args.epochs, np.log10(args.lr), np.log10(args.end_lr), restarts=args.restarts
    )

    torch.manual_seed(args.seed)
    seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    """
    Dataset creation and definition
    """
    class_names = None
    if args.dataset == "mnist":
        num_classes = 10
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
            test_tf.append(repeat)

        testset = datasets.MNIST(
            "../data", train=False, transform=transforms.Compose(test_tf),
        )

        data_shape = torch.zeros(  # pylint:disable=no-member
            [1, 1, 28, 28], dtype=torch.float  # pylint:disable=no-member
        ).to(device)
    elif args.dataset == "pneumonia":
        num_classes = 3

        # Different train and inference resolution only works with adaptive
        # pooling in model activated

        test_tf = [
            transforms.Resize(args.inference_resolution),
            transforms.CenterCrop(args.inference_resolution),
            transforms.ToTensor(),
            transforms.Normalize((0.57282609,), (0.17427578,)),
        ]

        # Duplicate grayscale one channel image into 3 channels

        repeat = transforms.Lambda(
            lambda x: torch.repeat_interleave(x, 3, dim=0)  # pylint: disable=no-member
        )
        test_tf.append(repeat)
        testset = PPPP(
            "data/Labels.csv",
            train=False,
            transform=transforms.Compose(test_tf),
            seed=args.seed,
        )
        class_names = {0: "normal", 1: "bacterial pneumonia", 2: "viral pneumonia"}
        data_shape = torch.zeros(  # pylint:disable=no-member
            [1, 3, args.train_resolution, args.train_resolution,],
            dtype=torch.float,  # pylint:disable=no-member
        ).to(device)
    else:
        raise NotImplementedError("dataset not implemented")

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size, shuffle=False, **kwargs
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
            print("No pretrained version available")
        model = conv_at_resolution[args.train_resolution](
            num_classes=num_classes, in_channels=3 if args.dataset == "pneumonia" else 1
        )
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
    model.to(device)
    traced_model = torch.jit.trace(model, data_shape.to(device),)

    _, acc = test(
        args,
        traced_model,
        device,
        test_loader,
        0,
        loss_fn,
        num_classes=num_classes,
        vis_params=vis_params,
        class_names=class_names,
    )

    learning_rate = args.lr
    for curr_round in range(1, args.epochs + 1):
        logger.info("Training round %s/%s", curr_round, args.epochs)

        results = await asyncio.gather(
            *[
                fit_model_on_worker(
                    worker=worker,
                    traced_model=traced_model,
                    batch_size=args.batch_size,
                    curr_round=curr_round,
                    max_nr_batches=0,
                    lr=learning_rate,
                    args=args,
                    device=device,
                )
                for worker in worker_instances
            ]
        )
        if args.visdom:
            vis.line(
                X=np.asarray([curr_round - 1]),
                Y=np.asarray([learning_rate]),
                win="lr_win",
                name="learning_rate",
                update="append",
                env=vis_env,
            )
        learning_rate = scheduler.get_lr(curr_round - 1)
        models = {}
        loss_values = {}

        # Federate models (note that this will also change the model in models[0]
        for worker_id, worker_model, worker_loss in results:
            if worker_model is not None:
                models[worker_id] = worker_model
                loss_values[worker_id] = worker_loss
        traced_model = utils.federated_avg(models)
        if args.visdom:
            loss = torch.mean(  # pylint:disable=no-member
                torch.stack(tuple(loss_values.values()))  # pylint:disable=no-member
            )
            vis_params["vis"].line(
                X=np.asarray([curr_round]),
                Y=np.asarray([loss.item()]),
                win="loss_win",
                name="train_loss",
                update="append",
                env=vis_params["vis_env"],
            )

        test_models = curr_round % args.test_interval == 0 or curr_round == args.epochs
        if test_models:
            _, acc = test(
                args,
                traced_model,
                device,
                test_loader,
                curr_round,
                loss_fn,
                num_classes=num_classes,
                vis_params=vis_params,
                class_names=class_names,
            )

    # torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    # Logging setup
    FORMAT = "%(asctime)s | %(message)s"
    logging.basicConfig(format=FORMAT)
    logger.setLevel(level=logging.DEBUG)

    # Websockets setup
    websockets_logger = logging.getLogger("websockets")
    websockets_logger.setLevel(logging.INFO)
    websockets_logger.addHandler(logging.StreamHandler())

    # Run main
    asyncio.get_event_loop().run_until_complete(main())
