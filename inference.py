"""
This implementation is based on the pysyft tutorial:
https://github.com/OpenMined/PySyft/blob/master/examples/tutorials/Part%2011%20-%20Secure%20Deep%20Learning%20Classification.ipynb
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import torch

torch.set_num_threads(1)
import configparser
import argparse
import syft as sy
import sys, os.path
from warnings import warn
from torchvision import datasets, transforms, models
from argparse import Namespace
from tqdm import tqdm
from sklearn import metrics as mt
from numpy import newaxis
from os import listdir
import json
import albumentations as a

from random import seed as rseed

from torchlib.utils import stats_table, Arguments  # pylint:disable=import-error
from torchlib.models import vgg16, resnet18, conv_at_resolution
from torchlib.websocket_utils import read_websocket_config
from torchlib.dicomtools import CombinedLoader
from torchlib.dataloader import AlbumentationsTorchTransform


class PathDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        transform=None,
        loader=CombinedLoader(),
        extensions=[
            ".jpg",
            ".jpeg",
            ".png",
            ".ppm",
            ".bmp",
            ".pgm",
            ".tif",
            ".tiff",
            ".webp",
            ".dcm",
            ".dicom",
        ],
    ):
        super(PathDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.loader = loader
        self.imgs = [
            f
            for f in listdir(root)
            if os.path.splitext(f)[1].lower() in extensions
            and not os.path.split(f)[1].lower().startswith("._")
        ]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = self.loader(os.path.join(self.root, img_path))
        if self.transform:
            img = self.transform(img)
        return img


class RemoteTensorDataset(torch.utils.data.Dataset):
    def __init__(self, tensor):
        self.tensor = tensor

    def __len__(self):
        return self.tensor.shape[0]

    def __getitem__(self, idx):
        return self.tensor[idx].copy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="data to classify")
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
    parser.add_argument("--cuda", action="store_true", help="Use CUDA acceleration.")
    cmd_args = parser.parse_args()

    use_cuda = cmd_args.cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")  # pylint: disable=no-member
    state = torch.load(cmd_args.model_weights, map_location=device)

    args = state["args"]
    if type(args) is Namespace:
        args = Arguments.from_namespace(args)
    args.from_previous_checkpoint(cmd_args)
    sys.stderr.write(str(args))

    torch.manual_seed(args.seed)
    rseed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if cmd_args.encrypted_inference:
        hook = sy.TorchHook(torch)
        if cmd_args.websockets_config:
            worker_dict = read_websocket_config(cmd_args.websocket_config)
            worker_names = [id_dict["id"] for _, id_dict in worker_dict.items()]
            assert (
                "crypto_provider" in worker_names and "data_owner" in worker_names
            ), "No crypto_provider and data_owner in websockets config"
            data_owner = sy.grid.clients.data_centric_fl_client.DataCentricFLClient(
                hook,
                "http://{:s}:{:s}".format(
                    worker_dict["data_owner"]["id"], worker_dict["data_owner"]["port"]
                ),
            )
            crypto_provider = sy.grid.clients.data_centric_fl_client.DataCentricFLClient(
                hook,
                "http://{:s}:{:s}".format(
                    worker_dict["crypto_provider"]["id"],
                    worker_dict["crypto_provider"]["port"],
                ),
            )
        else:
            data_owner = sy.VirtualWorker(hook, id="data_owner")
            crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")
        model_owner = sy.VirtualWorker(hook, id="model_owner")
        workers = [model_owner, data_owner]
        sy.local_worker.clients = [model_owner, data_owner]

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    class_names = None
    val_mean_std = (
        state["val_mean_std"]
        if "val_mean_std" in state.keys()
        else (
            torch.tensor([0.5]),  # pylint:disable=not-callable
            torch.tensor([0.2]),  # pylint:disable=not-callable
        )
        if args.pretrained
        else (
            torch.tensor([0.5, 0.5, 0.5]),  # pylint:disable=not-callable
            torch.tensor([0.2, 0.2, 0.2]),  # pylint:disable=not-callable
        )
    )
    mean, std = val_mean_std
    if args.data_dir == "mnist":
        num_classes = 10
        tf = transforms.Compose(
            [
                transforms.Resize(args.inference_resolution),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        num_classes = 3
        tf = [
            a.Resize(args.inference_resolution, args.inference_resolution),
            a.CenterCrop(args.inference_resolution, args.inference_resolution),
        ]
        if hasattr(args, "clahe") and args.clahe:
            tf.append(a.CLAHE(always_apply=True, clip_limit=(1, 1)))
        tf.extend(
            [
                a.ToFloat(max_value=255.0),
                a.Normalize(
                    mean.cpu().numpy()[None, None, :],
                    std.cpu().numpy()[None, None, :],
                    max_pixel_value=1.0,
                ),
            ]
        )
        tf = AlbumentationsTorchTransform(a.Compose(tf))

        class_names = {0: "normal", 1: "bacterial pneumonia", 2: "viral pneumonia"}

    loader = CombinedLoader()
    if not args.pretrained:
        loader.change_channels(1)
    if not cmd_args.websockets_config:
        dataset = PathDataset(cmd_args.data_dir, transform=tf, loader=loader,)
        if cmd_args.encrypted_inference:
            data = []
            for d in tqdm(dataset, total=len(dataset), leave=False, desc="load data"):
                data.append(d)
            data = torch.stack(data)
            data.tag("#inference_data")
            data_owner.load_data([data])
    if cmd_args.encrypted_inference:
        grid = sy.PrivateGridNetwork(data_owner)
        data_tensor = grid.search("#inference_data")["data_owner"][0]
        dataset = RemoteTensorDataset(data_tensor)

        # for worker in data.keys():
        #     dist_dataset = [  # TODO: in the future transform here would be nice but currently raise errors
        #         sy.BaseDataset(
        #             data[worker][0], torch.zeros_like(data[worker][0])
        #         )  # transform=federated_tf
        #     ]
        #     fed_dataset = sy.FederatedDataset(dist_dataset)
        #     test_loader = sy.FederatedDataLoader(
        #         fed_dataset, batch_size=1, shuffle=False
        #     )

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
    model.to(device)
    if args.encrypted_inference:
        fix_prec_kwargs = {"precision_fractional": 4, "dtype": "long"}
        share_kwags = {
            "crypto_provider": crypto_provider,
            "protocol": "fss",
            "requires_grad": False,
        }
        model.fix_precision(precision_fractional=4, dtype="long").share(
            *workers,
            crypto_provider=crypto_provider,
            protocol="fss",
            requires_grad=False
        )
    # test method
    model.eval()
    total_pred, total_target, total_scores = [], [], []
    if args.encrypted_inference:
        mean, std = mean.send(data_owner), std.send(data_owner)
    with torch.no_grad():
        for data in tqdm(
            dataset, total=len(dataset), desc="performing inference", leave=False,
        ):
            ## TODO: remove
            # if type(data) == tuple:
            #     if data[1] != 1:
            #         continue
            #     data = data[0]
            if len(data.shape) > 4:
                data = data.squeeze()
                if len(data.shape) > 4:
                    raise ValueError("need 4 dimensional tensor")
            while len(data.shape) < 4:
                data = data.unsqueeze(0)
            data = data.to(device)
            ## normalize data
            if args.encrypted_inference:
                data = (
                    data.fix_precision(precision_fractional=4, dtype="long")
                    .share(
                        *workers,
                        crypto_provider=crypto_provider,
                        protocol="fss",
                        requires_grad=False
                    )
                    .get()
                )
            output = model(data)
            if args.encrypted_inference:
                output = output.get().float_prec()
            pred = output.argmax(dim=1)
            total_pred.append(pred.detach().cpu().item())
    pred_dict = {"Inference Results": dict(enumerate(total_pred))}
    sys.stdout.write(json.dumps(pred_dict))
    from collections import Counter

    print("\n{:s}".format(str(Counter(total_pred))))

