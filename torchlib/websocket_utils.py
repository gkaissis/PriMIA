import torch


import syft as sy
import numpy as np
from tqdm import tqdm
from pandas import read_csv
from torchvision.datasets import MNIST
from torchvision import transforms
import sys
import os.path
from random import seed

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
from torchlib.dataloader import PPPP


KEEP_LABELS_DICT = {
    "alice": [0, 1, 2, 3],
    "bob": [4, 5, 6],
    "charlie": [7, 8, 9],
    None: list(range(10)),
}


def start_webserver(id: str, port: int, data_dir=None):
    hook = sy.TorchHook(torch)

    server = sy.workers.websocket_server.WebsocketServerWorker(
        id=id, host=None, port=port, hook=hook, verbose=True
    )
    torch.manual_seed(1)
    seed(1)
    np.random.seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if data_dir:
        if "mnist" in data_dir:
            dataset = MNIST(
                root="./data",
                train=True,
                download=True,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
                ),
            )
            if id in KEEP_LABELS_DICT:
                indices = np.isin(dataset.targets, KEEP_LABELS_DICT[id]).astype("uint8")
                selected_data = (
                    torch.native_masked_select(  # pylint:disable=no-member
                        dataset.data.transpose(0, 2),
                        torch.tensor(indices),  # pylint:disable=not-callable
                    )
                    .view(28, 28, -1)
                    .transpose(2, 0)
                )
                selected_targets = torch.native_masked_select(  # pylint:disable=no-member
                    dataset.targets,
                    torch.tensor(indices),  # pylint:disable=not-callable
                )
                dataset = sy.BaseDataset(
                    data=selected_data,
                    targets=selected_targets,
                    transform=dataset.transform,
                )
            dataset_name = "mnist"
        else:
            from torchvision.datasets import ImageFolder
            from utils import AddGaussianNoise  # pylint: disable=import-error

            train_tf = [
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomAffine(
                    degrees=30,
                    translate=(0, 0),
                    scale=(0.85, 1.15),
                    shear=10,
                    #    fillcolor=0.0,
                ),
                transforms.Resize(224),
                transforms.RandomCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.57282609,), (0.17427578,)),
                # transforms.RandomApply([AddGaussianNoise(mean=0.0, std=0.05)], p=0.5),
            ]
            """train_tf.append(
                transforms.Lambda(
                    lambda x: torch.repeat_interleave(  # pylint: disable=no-member
                        x, 3, dim=0
                    )
                )
            )"""
            target_dict_pneumonia = {0: 1, 1: 0, 2: 2}
            dataset = ImageFolder(
                data_dir,
                transform=transforms.Compose(train_tf),
                target_transform=lambda x: target_dict_pneumonia[x],
            )
            data, targets = [], []
            for d, t in tqdm(dataset, total=len(dataset)):
                data.append(d)
                targets.append(t)
            data = torch.stack(data)  # pylint:disable=no-member
            targets = torch.from_numpy(np.array(targets))  # pylint:disable=no-member
            dataset = sy.BaseDataset(data=data, targets=targets)
            """dataset = PPPP(
                "data/Labels.csv",
                train=True,
                transform=transforms.Compose(train_tf),
                seed=1
            )"""
            dataset_name = "pneumonia"
        print("added {:s} dataset".format(dataset_name))
        server.register_obj(dataset, obj_id=dataset_name)
    server.start()
    return server


def read_websocket_config(path: str):
    df = read_csv(path, header=None, index_col=0)
    return df.to_dict()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--id", type=str, required=True, help="id of worker")
    parser.add_argument("--port", type=int, required=True, help="Port to be opened")
    parser.add_argument(
        "--data_directory",
        type=str,
        default=None,
        required=True,
        help="Directory where data is stored",
    )
    args = parser.parse_args()

    start_webserver(args.id, args.port, args.data_directory)
