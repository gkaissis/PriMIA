from flask import Flask
from flask_sockets import Sockets

import syft as sy
import numpy as np
import torch
from torchvision import transforms

from torchvision.datasets import ImageFolder
from tqdm import tqdm
import sys
import os.path

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)
    )
)
from torchlib.utils import AddGaussianNoise
from torchlib.dataloader import LabelMNIST


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

            # selected_data = dataset.data.unsqueeze(1)
            # selected_targets = dataset.targets
            dataset = LabelMNIST(
                labels=KEEP_LABELS_DICT[node_id]
                if node_id in KEEP_LABELS_DICT
                else KEEP_LABELS_DICT[None],
                root="./data",
                train=True,
                download=True,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)),]
                ),
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
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.57282609,), (0.17427578,)),
                transforms.RandomApply([AddGaussianNoise(mean=0.0, std=0.05)], p=0.5),
            ]
            target_dict_pneumonia = {0: 1, 1: 0, 2: 2}
            dataset = ImageFolder(
                data_dir,
                transform=transforms.Compose(train_tf),
                target_transform=lambda x: target_dict_pneumonia[x],
            )
            dataset_name = "pneumonia"

        data, targets = [], []
        for d, t in tqdm(dataset, total=len(dataset)):
            data.append(d)
            targets.append(t)
        selected_data = torch.stack(data)  # pylint:disable=no-member
        selected_targets = torch.tensor(targets) # pylint:disable=not-callable
        del data, targets
        selected_data.tag(dataset_name, "#data")
        selected_targets.tag(dataset_name, "#target")
        """selected_data.send(local_worker)
        selected_targets.send(local_worker)"""
        local_worker.load_data([selected_data, selected_targets])

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
