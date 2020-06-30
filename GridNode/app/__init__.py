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


def create_app(
    node_id,
    debug=False,
    database_url=None,
    data_dir: str = None,
    config_file: str = None,
):
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
        import configparser
        from torchlib.utils import Arguments
        from os import path
        from argparse import Namespace
        from random import seed as r_seed

        config = configparser.ConfigParser()
        assert path.isfile(config_file), "config file not found"
        config.read(config_file)
        cmd_args = Namespace(
            dataset="mnist" if "mnist" in data_dir.lower() else "pneumonia",
            no_visdom=False,
            no_cuda=False,
            train_federated=True,
            websockets=True,
            verbose=True,
        )
        args = Arguments(cmd_args, config, mode="train", verbose=False)
        torch.manual_seed(args.seed)
        r_seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if args.dataset == "mnist":
            node_id = local_worker.id
            KEEP_LABELS_DICT = {
                "alice": [0, 1, 2, 3],
                "bob": [4, 5, 6],
                "charlie": [7, 8, 9],
                None: list(range(10)),
            }
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
        else:
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
                AddGaussianNoise(mean=0.0, std=args.noise_std, p=args.noise_prob),
            ]
            target_dict_pneumonia = {0: 1, 1: 0, 2: 2}
            dataset = ImageFolder(
                data_dir,
                transform=transforms.Compose(train_tf),
                target_transform=lambda x: target_dict_pneumonia[x],
            )

        data, targets = [], []
        for j in tqdm(
            range(args.repetitions_dataset),
            total=args.repetitions_dataset,
            leave=False,
            desc="register data on {:s}".format(local_worker.id),
        ):
            for d, t in tqdm(
                dataset,
                total=len(dataset),
                leave=False,
                desc="register data {:d}. time".format(j),
            ):
                data.append(d)
                targets.append(t)
        selected_data = torch.stack(data)  # pylint:disable=no-member
        selected_targets = torch.tensor(targets)  # pylint:disable=not-callable
        del data, targets
        selected_data.tag(args.dataset, "#traindata")
        selected_targets.tag(args.dataset, "#traintargets")
        local_worker.load_data([selected_data, selected_targets])

        print(
            "registered {:d} samples of {:s} data".format(
                selected_data.size(0), args.dataset
            )
        )
        del selected_data, selected_targets

        # print(local_worker.request_search(dataset, location=local_worker))

    # Register app blueprints
    app.register_blueprint(html, url_prefix=r"/")
    sockets.register_blueprint(ws, url_prefix=r"/")

    # Set Authentication configs
    app = auth.set_auth_configs(app)

    return app
