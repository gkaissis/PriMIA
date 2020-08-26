import logging
import os

# TODO: Tensorflow catch

import sys
import os.path

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)
    )
)

import syft as sy
import numpy as np
import torch

torch.set_num_threads(1)
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm


from .util import mask_payload_fast
from flask import Flask
from flask_cors import CORS
from flask_executor import Executor
from flask_migrate import Migrate
from flask_sockets import Sockets
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy_mixins import AllFeaturesMixin
from sqlalchemy_utils.functions import database_exists
from geventwebsocket.websocket import Header

# Monkey patch geventwebsocket.websocket.Header.mask_payload() and
# geventwebsocket.websocket.Header.unmask_payload(), for efficiency
Header.mask_payload = mask_payload_fast
Header.unmask_payload = mask_payload_fast

from .version import __version__

KEEP_LABELS_DICT = {
    "alice": [0, 1, 2, 3],
    "bob": [4, 5, 6],
    "charlie": [7, 8, 9],
    None: list(range(10)),
}
# Default secret key used only for testing / development
DEFAULT_SECRET_KEY = "justasecretkeythatishouldputhere"

db = SQLAlchemy()
executor = Executor()
logging.getLogger().setLevel(logging.INFO)


class BaseModel(db.Model, AllFeaturesMixin):
    __abstract__ = True
    pass


# Tables must be created after db has been created
from .main.users import Role


def set_database_config(app, test_config=None, verbose=False):
    """Set configs to use SQL Alchemy library.

    Args:
        app: Flask application.
        test_config : Dictionary containing SQLAlchemy configs for test purposes.
        verbose : Level of flask application verbosity.
    Returns:
        app: Flask application.
    Raises:
        RuntimeError : If DATABASE_URL or test_config didn't initialized, RuntimeError exception will be raised.
    """
    db_url = os.environ.get("DATABASE_URL")
    migrate = Migrate(app, db)
    if test_config is None:
        if db_url:
            app.config.from_mapping(
                SQLALCHEMY_DATABASE_URI=db_url, SQLALCHEMY_TRACK_MODIFICATIONS=False
            )
        else:
            raise RuntimeError(
                "Invalid database address : Set DATABASE_URL environment var or add test_config parameter at create_app method."
            )
    else:
        app.config["SQLALCHEMY_DATABASE_URI"] = test_config["SQLALCHEMY_DATABASE_URI"]
        app.config["TESTING"] = (
            test_config["TESTING"] if test_config.get("TESTING") else True
        )
        app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = (
            test_config["SQLALCHEMY_TRACK_MODIFICATIONS"]
            if test_config.get("SQLALCHEMY_TRACK_MODIFICATIONS")
            else False
        )
    app.config["VERBOSE"] = verbose
    db.init_app(app)


def seed_db():
    global db

    new_role = Role(
        name="User",
        can_triage_jobs=False,
        can_edit_settings=False,
        can_create_users=False,
        can_create_groups=False,
        can_edit_roles=False,
        can_manage_infrastructure=False,
    )
    db.session.add(new_role)

    new_role = Role(
        name="Compliance Officer",
        can_triage_jobs=True,
        can_edit_settings=False,
        can_create_users=False,
        can_create_groups=False,
        can_edit_roles=False,
        can_manage_infrastructure=False,
    )
    db.session.add(new_role)

    new_role = Role(
        name="Administrator",
        can_triage_jobs=True,
        can_edit_settings=True,
        can_create_users=True,
        can_create_groups=True,
        can_edit_roles=False,
        can_manage_infrastructure=False,
    )
    db.session.add(new_role)

    new_role = Role(
        name="Owner",
        can_triage_jobs=True,
        can_edit_settings=True,
        can_create_users=True,
        can_create_groups=True,
        can_edit_roles=True,
        can_manage_infrastructure=True,
    )
    db.session.add(new_role)


def create_app(
    node_id: str,
    debug=False,
    n_replica=None,
    test_config=None,
    data_dir=None,
    config_file=None,
    mean_std_file: str = None,
) -> Flask:
    """Create flask application.

    Args:
         node_id: ID used to identify this node.
         debug: debug mode flag.
         n_replica: Number of model replicas used for fault tolerance purposes.
         test_config: database test settings.
    Returns:
         app : Flask App instance.
    """
    app = Flask(__name__)
    app.debug = debug

    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", None)

    if app.config["SECRET_KEY"] is None:
        app.config["SECRET_KEY"] = DEFAULT_SECRET_KEY
        logging.warning(
            "Using default secrect key, this is not safe and should be used only for testing and development. To define a secrete key please define the environment variable SECRET_KEY."
        )

    app.config["N_REPLICA"] = n_replica
    sockets = Sockets(app)

    # Register app blueprints
    from .main import (
        auth,
        data_centric_routes,
        hook,
        local_worker,
        main_routes,
        model_centric_routes,
        ws,
    )

    # set_node_id(id)
    local_worker.id = node_id
    hook.local_worker._known_workers[node_id] = local_worker
    local_worker.add_worker(hook.local_worker)

    # add data
    if data_dir:
        import configparser
        import albumentations as a
        from inference import PathDataset
        from torchlib.utils import Arguments
        from torchlib.dicomtools import CombinedLoader
        from torchlib.dataloader import AlbumentationsTorchTransform
        from os import path
        from argparse import Namespace
        from random import seed as r_seed
        from torchlib.utils import AddGaussianNoise, To_one_hot, MixUp
        from torchlib.dataloader import LabelMNIST, calc_mean_std
        from torchlib.dicomtools import CombinedLoader
        from train import create_albu_transform

        loader = CombinedLoader()
        config = configparser.ConfigParser()
        assert path.isfile(config_file), "config file not found"
        config.read(config_file)
        cmd_args = Namespace(
            data_dir=data_dir,
            visdom=False,
            cuda=False,
            train_federated=True,
            websockets=True,
            verbose=True,
            unencrypted_aggregation=False,
        )
        args = Arguments(cmd_args, config, mode="train", verbose=False)
        # torch.manual_seed(args.seed)
        # r_seed(args.seed)
        # np.random.seed(args.seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        if node_id == "data_owner":
            print("setting up an data to be remotely classified.")
            tf = [
                a.Resize(args.inference_resolution, args.inference_resolution),
                a.CenterCrop(args.inference_resolution, args.inference_resolution),
            ]
            if hasattr(args, "clahe") and args.clahe:
                tf.append(a.CLAHE(always_apply=True, clip_limit=(1, 1)))
            if mean_std_file:
                mean_std = torch.load(mean_std_file)
                if type(mean_std) == dict and "val_mean_std" in mean_std:
                    mean_std = mean_std["val_mean_std"]
                mean, std = mean_std
            else:
                raise RuntimeError(
                    "To set up a data owner for inference we need a file which tells"
                    " us how to normalize the data."
                )
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
            loader = CombinedLoader()

            dataset = PathDataset(data_dir, transform=tf, loader=loader,)
            data = []
            for d in tqdm(dataset, total=len(dataset), leave=False, desc="load data"):
                data.append(d)
            data = torch.stack(data)  # pylint:disable=no-member
            data.tag("#inference_data")
            local_worker.load_data([data])
            print("Loaded {:d} samples as inference data".format(data.shape[0]))
        else:
            if args.data_dir == "mnist":
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
                        [
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,)),
                        ]
                    ),
                )
            else:
                stats_dataset = ImageFolder(
                    data_dir,
                    loader=loader,
                    transform=transforms.Compose(
                        [
                            transforms.Resize(args.train_resolution),
                            transforms.CenterCrop(args.train_resolution),
                            transforms.ToTensor(),
                        ]
                    ),
                )
                assert (
                    len(stats_dataset.classes) == 3
                ), "We can only handle data that has 3 classes: normal, bacterial and viral"
                mean, std = calc_mean_std(stats_dataset, save_folder=data_dir,)
                del stats_dataset
                target_tf = None
                if args.mixup or args.weight_classes:
                    target_tf = [
                        lambda x: torch.tensor(x),  # pylint:disable=not-callable
                        To_one_hot(3),
                    ]
                dataset = ImageFolder(
                    # path.join("data/server_simulation/", "validation")
                    # if worker.id == "validation"
                    # else
                    data_dir,
                    loader=loader,
                    transform=create_albu_transform(args, mean, std),
                    target_transform=transforms.Compose(target_tf)
                    if target_tf
                    else None,
                )
                assert (
                    len(dataset.classes) == 3
                ), "We can only handle data that has 3 classes: normal, bacterial and viral"
                mean.tag("#datamean")
                std.tag("#datastd")
                local_worker.load_data([mean, std])
            data, targets = [], []
            # repetitions = 1 if worker.id == "validation" else args.repetitions_dataset
            if args.mixup:
                dataset = torch.utils.data.DataLoader(
                    dataset, batch_size=1, shuffle=True
                )
                mixup = MixUp(λ=args.mixup_lambda, p=args.mixup_prob)
                last_set = None
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
                    desc="register data {:d}. time".format(j + 1),
                ):
                    if args.mixup:
                        original_set = (d, t)
                        if last_set:
                            # pylint:disable=unsubscriptable-object
                            d, t = mixup(((d, last_set[0]), (t, last_set[1])))
                        last_set = original_set
                    data.append(d)
                    targets.append(t)
            selected_data = torch.stack(data)  # pylint:disable=no-member
            selected_targets = (
                torch.stack(targets)  # pylint:disable=no-member
                if args.mixup or args.weight_classes
                else torch.tensor(targets)  # pylint:disable=not-callable
            )
            if args.mixup:
                selected_data = selected_data.squeeze(1)
                selected_targets = selected_targets.squeeze(1)
            del data, targets
            selected_data.tag(
                "#traindata",
            )  # "#valdata" if worker.id == "validation" else
            selected_targets.tag(
                # "#valtargets" if worker.id == "validation" else
                "#traintargets",
            )
            local_worker.load_data([selected_data, selected_targets])
            print(
                "registered {:d} samples of {:s} data".format(
                    selected_data.size(0), args.data_dir
                )
            )
            del selected_data, selected_targets

    # Register app blueprints
    app.register_blueprint(main_routes, url_prefix=r"/")
    app.register_blueprint(model_centric_routes, url_prefix=r"/model-centric")
    app.register_blueprint(data_centric_routes, url_prefix=r"/data-centric")

    sockets.register_blueprint(ws, url_prefix=r"/")

    # Set SQLAlchemy configs
    set_database_config(app, test_config=test_config)
    s = app.app_context().push()

    if database_exists(db.engine.url):
        db.create_all()
    else:
        db.create_all()
        seed_db()

    db.session.commit()

    # Set Authentication configs
    app = auth.set_auth_configs(app)

    CORS(app)

    # Threads
    executor.init_app(app)
    app.config["EXECUTOR_PROPAGATE_EXCEPTIONS"] = True
    app.config["EXECUTOR_TYPE"] = "thread"

    return app
