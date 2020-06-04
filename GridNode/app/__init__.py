from flask import Flask
from flask_sockets import Sockets

import syft as sy
import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.datasets import ImageFolder
from tqdm import tqdm


# from utils import AddGaussianNoise  # pylint: disable=import-error

KEEP_LABELS_DICT = {
    "alice": [0, 1, 2, 3],
    "bob": [4, 5, 6],
    "charlie": [7, 8, 9],
    None: list(range(10)),
}


def create_app(node_id, debug=False, database_url=None, data_dir: str = None):
    """ Create / Configure flask socket application instance.
        
        Args:
            node_id (str) : ID of Grid Node.
            debug (bool) : debug flag.
            test_config (bool) : Mock database environment.
        Returns:
            app : Flask application instance.
    """
    app = Flask(__name__)
    app.debug = debug

    app.config["SECRET_KEY"] = "justasecretkeythatishouldputhere"

    # Enable persistent mode
    # Overwrite syft.object_storage methods to work in a persistent way
    # Persist models / tensors
    if database_url:
        app.config["REDISCLOUD_URL"] = database_url
        from .main.persistence import database, object_storage

        db_instance = database.set_db_instance(database_url)
        object_storage.set_persistent_mode(db_instance)

    from .main import html, ws, hook, local_worker, auth

    # Global socket handler
    sockets = Sockets(app)

    # set_node_id(id)
    local_worker.id = node_id
    hook.local_worker._known_workers[node_id] = local_worker
    local_worker.add_worker(hook.local_worker)

    # add data
    if data_dir:
        print("register data")
        if "mnist" in data_dir.lower():
            dataset = MNIST(
                root="./data",
                train=True,
                download=True,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
                ),
            )
            selected_data = dataset.data
            selected_targets = dataset.targets
            if node_id in KEEP_LABELS_DICT:
                indices = np.isin(dataset.targets, KEEP_LABELS_DICT[node_id]).astype(
                    "uint8"
                )
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
            selected_data = torch.stack(data)  # pylint:disable=no-member
            selected_targets = torch.from_numpy(
                np.array(targets)
            )  # pylint:disable=no-member
            del data, targets
            dataset = sy.BaseDataset(data=selected_data, targets=selected_targets)
            """dataset = PPPP(
                "data/Labels.csv",
                train=True,
                transform=transforms.Compose(train_tf),
                seed=1
            )"""
            dataset_name = "pneumonia"
        #local_worker.register_obj(dataset, obj_id=dataset_name)
        dataset.tag(dataset_name)
        dataset.send(local_worker)
        # local_worker.register_obj(selected_data, obj_id="{:s}_data".format(dataset_name))
        # local_worker.register_obj(selected_targets, obj_id="{:s}_targets".format(dataset_name))

        print(
            "registered {:d} samples of {:s} data".format(
                selected_data.size(0), dataset_name
            )
        )

        # print(local_worker.request_search(dataset, location=local_worker))

    # Register app blueprints
    app.register_blueprint(html, url_prefix=r"/")
    sockets.register_blueprint(ws, url_prefix=r"/")

    # Set Authentication configs
    app = auth.set_auth_configs(app)

    return app
