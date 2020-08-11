"""
This implementation is based on the pysyft tutorial:
https://github.com/OpenMined/PySyft/blob/master/examples/tutorials/Part%2011%20-%20Secure%20Deep%20Learning%20Classification.ipynb
"""


import torch
import configparser
import argparse
import syft as sy
import sys, os.path
from warnings import warn
from torchvision import datasets, transforms, models
from argparse import Namespace
from tqdm import tqdm

sys.path.insert(0, os.path.split(sys.path[0])[0])  # TODO: make prettier
from utils import test, Arguments  # pylint:disable=import-error
from torchlib.dataloader import PPPP, ImageFolderFromCSV
from torchlib.models import vgg16, resnet18, conv_at_resolution
from torchlib.websocket_utils import read_websocket_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        default="pneumonia",
        choices=["pneumonia", "mnist"],
        help="which dataset?",
    )
    parser.add_argument(
        "--model_weights",
        type=str,
        required=True,
        default=None,
        help="model weights to use",
    )
    parser.add_argument(
        "--encrypted_inference", action="store_true", help="Perform encrypted inference"
    )
    parser.add_argument(
        "--websockets_config",
        default=None,
        help="Give csv file where ip address and port of data_owner and "
        "crypto_provider are given"
        "\nNote: Names must be exactly like that"
        "\nFirst column consists of id, host and port"
        "\nIf not passed as argument virtual workers are used",
    )
    parser.add_argument("--adults", action="store_true", help="Use adult images")
    parser.add_argument("--no_cuda", action="store_true", help="dont use gpu")
    cmd_args = parser.parse_args()

    use_cuda = not cmd_args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")  # pylint: disable=no-member
    state = torch.load(cmd_args.model_weights, map_location=device)

    args = state["args"]
    if type(args) is Namespace:
        args = Arguments.from_namespace(args)
    args.from_previous_checkpoint(cmd_args)
    print(str(args))

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if cmd_args.encrypted_inference:
        hook = sy.TorchHook(torch)
        if cmd_args.websockets_config:
            worker_dict = read_websocket_config("configs/websetting/config.csv")
            worker_names = [id_dict["id"] for _, id_dict in worker_dict.items()]
            assert (
                "crypto_provider" in worker_names and "data_owner" in worker_names
            ), "No crypto_provider and data_owner in websockets config"
            data_owner = sy.workers.node_client.NodeClient(
                hook,
                "http://{:s}:{:s}".format(
                    worker_dict["data_owner"]["id"], worker_dict["data_owner"]["port"]
                ),
            )
            crypto_provider = sy.workers.node_client.NodeClient(
                hook,
                "http://{:s}:{:s}".format(
                    worker_dict["crypto_provider"]["id"],
                    worker_dict["crypto_provider"]["port"],
                ),
            )
        else:
            data_owner = sy.VirtualWorker(hook, id="data_owner")
            crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    class_names = None
    if args.dataset == "mnist":
        num_classes = 10
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
        tf = [
            transforms.Resize(args.inference_resolution),
            transforms.CenterCrop(args.inference_resolution),
            transforms.ToTensor(),
            transforms.Normalize((0.57282609,), (0.17427578,)),
        ]

        if args.pretrained:
            repeat = transforms.Lambda(
                lambda x: torch.repeat_interleave(  # pylint: disable=no-member
                    x, 3, dim=0
                )
            )
            tf.append(repeat)
        if cmd_args.adults:
            testset = ImageFolderFromCSV(
                "data/Chest_xray_Corona_Metadata.csv",
                "data/Adults",
                transform=transforms.Compose(tf),
            )

        else:
            testset = PPPP(
                "data/Labels.csv", train=False, transform=transforms.Compose(tf)
            )
        class_names = {0: "normal", 1: "bacterial pneumonia", 2: "viral pneumonia"}
    else:
        raise NotImplementedError("dataset not implemented")

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size, shuffle=True, **kwargs
    )
    if args.encrypted_inference:
        if not cmd_args.websockets_config:
            data, targets = [], []
            for d, t in tqdm(
                test_loader, total=len(test_loader), leave=False, desc="load data"
            ):
                data.append(d)
                targets.append(t)
            data = torch.stack(data)
            targets = torch.stack(targets)
            data.tag("#inference_data")
            targets.tag("#inference_targets")
            data_owner.load_data([data, targets])

        grid = sy.PrivateGridNetwork(data_owner)
        data = grid.search("#inference_data")
        targets = grid.search("#inference_targets")
        found_targets = len(targets) > 0
        if not found_targets:
            print("No targets found")
            targets = {
                worker: [torch.zeros_like(data[worker][0])] for worker in data.keys()
            }

        for worker in data.keys():
            dist_dataset = [  # TODO: in the future transform here would be nice but currently raise errors
                sy.BaseDataset(
                    data[worker][0], targets[worker][0],
                )  # transform=federated_tf
            ]
            fed_dataset = sy.FederatedDataset(dist_dataset)
            test_loader = sy.FederatedDataLoader(
                fed_dataset, batch_size=args.batch_size, shuffle=True
            )

    if args.model == "vgg16":
        model = vgg16(
            pretrained=args.pretrained,
            num_classes=num_classes,
            in_channels=3 if args.pretrained else 1,
            adptpool=False,
            input_size=args.inference_resolution,
            pooling=args.pooling_type,
        )
    elif args.model == "simpleconv":
        if args.pretrained:
            warn("No pretrained version available")
        model = conv_at_resolution[args.train_resolution](
            num_classes=num_classes,
            in_channels=3 if args.pretrained else 1,
            pooling=args.pooling_type,
        )
    elif args.model == "resnet-18":
        model = resnet18(
            pretrained=args.pretrained,
            num_classes=num_classes,
            in_channels=3 if args.pretrained else 1,
            adptpool=False,
            input_size=args.inference_resolution,
            pooling=args.pooling_type if hasattr(args, "pooling_type") else "avg",
        )
    else:
        raise NotImplementedError("model unknown")
    model.load_state_dict(state["model_state_dict"])
    # model = models.vgg16(pretrained=False, num_classes=3)
    # model.classifier = vggclassifier()
    model.to(device)
    # model = model.fix_prec().share(
    #     sy.local_worker, data_owner, crypto_provider=crypto_provider, protocol="fss"
    # )
    # test method
    model.eval()
    test_loss, TP = 0, 0
    total_pred, total_target, total_scores = [], [], []
    with torch.no_grad():
        for data, target in tqdm(
            test_loader,
            total=len(test_loader),
            desc="performing inference",
            leave=False,
        ):
            if args.encrypted_inference:
                data = data.copy()
                data = data.fix_prec()
                data = data.share(
                    sy.local_worker,
                    data_owner,
                    crypto_provider=crypto_provider,
                    protocol="fss",
                )
                data = data.get()
                data = data.squeeze()
            output = model(data)
            if args.encrypted_inference:
                output = output.get().float_prec()
            # test_loss += loss.item()  # sum up batch loss
            # total_scores.append(output)
            # pred = output.argmax(dim=1)
            # tgts = target.view_as(pred)
            # total_pred.append(pred)
            # total_target.append(tgts)
            # equal = pred.eq(tgts)
            # TP += (
            #     equal.sum().copy().get().float_precision().long().item()
            #     if args.encrypted_inference
            #     else equal.sum().item()
            # )
