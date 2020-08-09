import random
from os import path, remove
from argparse import Namespace
import numpy as np
import syft as sy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import visdom
from tabulate import tabulate
from torchvision import datasets, models, transforms
from sklearn import metrics as mt

from torchlib.dataloader import (
    PPPP,
    calc_mean_std,
)  # pylint:disable=import-error
from torchlib.models import resnet18
from torchlib.utils import (
    AddGaussianNoise,  # pylint:disable=import-error
    Cross_entropy_one_hot,
    LearningRateScheduler,
    MixUp,
    To_one_hot,
    stats_table,
    send_new_models,
    secure_aggregation,
)


def test(
    model, val_loader, epoch, num_classes, class_names=None,
):
    model.eval()
    test_loss = 0
    total_pred, total_target, total_scores = [], [], []
    with torch.no_grad():
        for data, target in tqdm.tqdm(
            val_loader,
            total=len(val_loader),
            desc="testing epoch {:d}".format(epoch),
            leave=False,
        ):
            output = model(data)
            total_scores.append(output)
            pred = output.argmax(dim=1)
            tgts = target.view_as(pred)
            total_pred.append(pred)
            total_target.append(tgts)
    test_loss /= len(val_loader)

    total_pred = torch.cat(total_pred).cpu().numpy()  # pylint: disable=no-member
    total_target = torch.cat(total_target).cpu().numpy()  # pylint: disable=no-member
    total_scores = torch.cat(total_scores).cpu().numpy()  # pylint: disable=no-member
    total_scores -= total_scores.min(axis=1)[:, np.newaxis]
    total_scores = total_scores / total_scores.sum(axis=1)[:, np.newaxis]

    roc_auc = mt.roc_auc_score(total_target, total_scores, multi_class="ovo")
    objective = 100.0 * roc_auc
    conf_matrix = mt.confusion_matrix(total_target, total_pred)
    report = mt.classification_report(
        total_target, total_pred, output_dict=True, zero_division=0
    )
    print(
        stats_table(
            conf_matrix,
            report,
            roc_auc=roc_auc,
            matthews_coeff=mt.matthews_corrcoef(total_target, total_pred),
            class_names=class_names,
            epoch=epoch,
        )
    )

    return test_loss, objective


if __name__ == "__main__":
    args = Namespace(
        batch_size=4,  # debug mode has 48 images, just so we make sure we sync a few times
        train_resolution=224,
        test_batch_size=4,
        test_interval=1,
        validation_split=10,
        epochs=1,
        lr=1e-4,
        end_lr=1e-5,
        restarts=0,
        beta1=0.5,
        beta2=0.99,
        weight_decay=5e-4,
        seed=1,
        log_interval=10,
        pretrained=True,
        pooling_type="max",
        vertical_flip_prob=0.5,
        rotation=30,
        scale=0.15,
        shear=10,
        translate=0,
        noise_std=0.05,
        noise_prob=0.5,
        mixup_prob=0.9,
        mixup_lambda=None,
        sync_every_n_batch=4,
        wait_interval=0.1,
        repetitions_dataset=1,
    )

    use_cuda = False

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = "cpu"

    num_classes = 3
    class_names = {0: "normal", 1: "bacterial pneumonia", 2: "viral pneumonia"}

    hook = sy.TorchHook(torch)
    from torchlib.websocket_utils import (  # pylint:disable=import-error
        read_websocket_config,
    )

    worker_dict = read_websocket_config("configs/websetting/config.csv")
    worker_names = [id_dict["id"] for _, id_dict in worker_dict.items()]

    crypto_in_config = "crypto_provider" in worker_names
    crypto_provider = None
    assert crypto_in_config, "No crypto provider in configuration"
    worker_names.remove("crypto_provider")
    cp_key = [
        key for key, worker in worker_dict.items() if worker["id"] == "crypto_provider"
    ]
    assert len(cp_key) == 1
    cp_key = cp_key[0]
    crypto_provider_data = worker_dict[cp_key]
    worker_dict.pop(cp_key)

    workers = {
        worker["id"]: sy.VirtualWorker(hook, id=worker["id"], verbose=False)
        for _, worker in worker_dict.items()
    }
    crypto_provider = sy.VirtualWorker(hook, id="crypto_provider", verbose=False)
    train_loader = None
    for i, worker in tqdm.tqdm(
        enumerate(workers.values()),
        total=len(workers.keys()),
        leave=False,
        desc="load data",
    ):
        data_dir = path.join("data/server_simulation/", "worker{:d}".format(i + 1))
        stats_dataset = datasets.ImageFolder(
            data_dir,
            transform=transforms.Compose(
                [
                    transforms.Resize(args.train_resolution),
                    transforms.CenterCrop(args.train_resolution),
                    transforms.ToTensor(),
                ]
            ),
        )
        mean, std = calc_mean_std(stats_dataset, save_folder=data_dir,)
        del stats_dataset
        train_tf = [
            transforms.RandomVerticalFlip(p=args.vertical_flip_prob),
            transforms.RandomAffine(
                degrees=args.rotation,
                translate=(args.translate, args.translate),
                scale=(1.0 - args.scale, 1.0 + args.scale),
                shear=args.shear,
                #    fillcolor=0,
            ),
            transforms.Resize(args.train_resolution),
            transforms.RandomCrop(args.train_resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            AddGaussianNoise(mean=0.0, std=args.noise_std, p=args.noise_prob),
        ]
        target_dict_pneumonia = {0: 1, 1: 0, 2: 2}
        target_tf = [lambda x: target_dict_pneumonia[x]]
        target_tf.append(lambda x: torch.tensor(x))  # pylint:disable=not-callable
        target_tf.append(To_one_hot(3))
        dataset = datasets.ImageFolder(
            data_dir,
            transform=transforms.Compose(train_tf),
            target_transform=transforms.Compose(target_tf),
        )
        mean.tag("#datamean")
        std.tag("#datastd")
        worker.load_data([mean, std])

        data, targets = [], []
        dataset = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
        mixup = MixUp(λ=args.mixup_lambda, p=args.mixup_prob)
        last_set = None
        for j in tqdm.tqdm(
            range(args.repetitions_dataset),
            total=args.repetitions_dataset,
            leave=False,
            desc="register data on {:s}".format(worker.id),
        ):
            for d, t in tqdm.tqdm(
                dataset,
                total=len(dataset),
                leave=False,
                desc="register data {:d}. time".format(j + 1),
            ):
                original_set = (d, t)
                if last_set:
                    # pylint:disable=unsubscriptable-object
                    d, t = mixup(((d, last_set[0]), (t, last_set[1])))
                last_set = original_set
                data.append(d)
                targets.append(t)
        selected_data = torch.stack(data)  # pylint:disable=no-member
        selected_targets = torch.stack(targets)  # pylint:disable=no-member
        selected_data = selected_data.squeeze(1)
        selected_targets = selected_targets.squeeze(1)
        del data, targets
        selected_data.tag(
            "pneumonia", "#traindata",  # "#valdata" if worker.id == "validation" else
        )
        selected_targets.tag(
            "pneumonia",
            # "#valtargets" if worker.id == "validation" else
            "#traintargets",
        )
        worker.load_data([selected_data, selected_targets])

    grid: sy.PrivateGridNetwork = sy.PrivateGridNetwork(*workers.values())
    data = grid.search("pneumonia", "#traindata")
    target = grid.search("pneumonia", "#traintargets")
    train_loader = {}
    total_L = 0
    for worker in data.keys():
        dist_dataset = [  # TODO: in the future transform here would be nice but currently raise errors
            sy.BaseDataset(
                data[worker][0], target[worker][0],
            )  # transform=federated_tf
        ]
        fed_dataset = sy.FederatedDataset(dist_dataset)
        total_L += len(fed_dataset)
        tl = sy.FederatedDataLoader(
            fed_dataset, batch_size=args.batch_size, shuffle=True
        )
        train_loader[workers[worker]] = tl

    means = [m[0] for m in grid.search("#datamean").values()]
    stds = [s[0] for s in grid.search("#datastd").values()]
    if len(means) > 0 and len(stds) > 0 and len(means) == len(stds):
        mean = (
            means[0]
            .fix_precision()
            .share(*workers, crypto_provider=crypto_provider)
            .get()
        )
        std = (
            stds[0]
            .fix_precision()
            .share(*workers, crypto_provider=crypto_provider)
            .get()
        )
        for m, s in zip(means[1:], stds[1:]):
            mean += (
                m.fix_precision().share(*workers, crypto_provider=crypto_provider).get()
            )
            std += (
                s.fix_precision().share(*workers, crypto_provider=crypto_provider).get()
            )
        mean = mean.get().float_precision() / len(stds)
        std = std.get().float_precision() / len(stds)
    else:
        mean, std = 0.5, 0.2  # BAM nice hard-coding @a1302z
    val_mean_std = torch.stack([mean, std])  # pylint:disable=no-member
    val_tf = [
        transforms.Resize(args.train_resolution),
        transforms.CenterCrop(args.train_resolution),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    target_dict_pneumonia = {0: 1, 1: 0, 2: 2}
    valset = datasets.ImageFolder(
        path.join("data/server_simulation/", "validation"),
        transform=transforms.Compose(val_tf),
        target_transform=lambda x: target_dict_pneumonia[x],
    )
    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=args.test_batch_size, shuffle=False
    )
    assert len(train_loader.keys()) == (
        len(workers.keys())
    ), "data was not correctly loaded"

    print(
        "Found a total dataset with {:d} samples on remote workers".format(
            sum([len(dl.federated_dataset) for dl in train_loader.values()])
        )
    )
    print(
        "Found a total validation set with {:d} samples (locally)".format(
            len(val_loader.dataset)
        )
    )
    cw = None

    scheduler = LearningRateScheduler(
        args.epochs, np.log10(args.lr), np.log10(args.end_lr), restarts=0
    )

    ## visdom
    vis_params = None

    model_type = resnet18
    model_args = {
        "pretrained": True,
        "num_classes": num_classes,
        "in_channels": 3,
        "adptpool": False,
        "input_size": 224,
        "pooling": "max",
    }
    models = model_type(**model_args)
    models = {
        key: models.copy() for key in [w.id for w in workers.values()] + ["local_model"]
    }

    optimizer = {
        idt: optim.Adam(
            m.parameters(),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )
        for idt, m in models.items()
        if idt not in ["local_model", "crypto_provider"]
    }
    loss_args = {"weight": cw, "reduction": "mean"}
    loss_fns = Cross_entropy_one_hot(**loss_args).to(device)
    loss_fns = {w: loss_fns.copy() for w in [*workers, "local_model"]}

    start_at_epoch = 1
    for m in models.values():
        m.to(device)

    test(
        models["local_model"], val_loader, 0, num_classes, class_names=None,
    )
    roc_auc_scores = []
    model_paths = []
    test_params = {
        "device": device,
        "val_loader": val_loader,
        "loss_fn": loss_fns,
        "num_classes": num_classes,
        "class_names": class_names,
        "exp_name": "minimal_script",
        "optimizer": optimizer,
        "roc_auc_scores": roc_auc_scores,
        "model_paths": model_paths,
    }
    for epoch in range(start_at_epoch, args.epochs + 1):

        for w in worker_names:
            new_lr = scheduler.adjust_learning_rate(
                optimizer[w],  # if args.secure_aggregation else optimizer.get_optim(w),
                epoch - 1,
            )

        for worker in train_loader.keys():
            if models[worker.id].location:
                continue
            models[worker.id].send(worker)
            loss_fns[worker.id].send(worker)
        # 1. Send new version of the model
        models = send_new_models(models["local_model"], models)

        for worker in optimizer.keys():
            kwargs = {"lr": args.lr, "weight_decay": args.weight_decay}
            ## TODO implement for SGD (also in train_federated)
            kwargs["betas"] = (args.beta1, args.beta2)
            optimizer[worker] = torch.optim.Adam(models[worker].parameters(), **kwargs)

        avg_loss = []

        num_batches = {key.id: len(loader) for key, loader in train_loader.items()}
        dataloaders = {key: iter(loader) for key, loader in train_loader.items()}
        pbar = tqdm.tqdm(
            range(max(num_batches.values())),
            total=max(num_batches.values()),
            leave=False,
            desc="Training with secure aggregation",
        )
        for batch_idx in pbar:
            for worker, dataloader in tqdm.tqdm(
                dataloaders.items(),
                total=len(dataloaders),
                leave=False,
                desc="Train batch {:d}".format(batch_idx),
            ):
                if batch_idx > num_batches[worker.id]:
                    continue
                loss_fn = loss_fns[worker.id]
                optimizer[worker.id].zero_grad()
                data, target = next(dataloader)
                pred = models[worker.id](data)
                loss = loss_fn(pred, target)
                loss.backward()
                optimizer[worker.id].step()
                avg_loss.append(loss.detach().cpu().get().item())
            if batch_idx > 0 and batch_idx % args.sync_every_n_batch == 0:
                pbar.set_description_str("Synchronizing")
                models["local_model"] = secure_aggregation(
                    models["local_model"],
                    models,
                    train_loader.keys(),
                    crypto_provider,
                    args,
                    test_params,
                )
                models = send_new_models(models["local_model"], models)
                pbar.set_description_str("Training with secure aggregation")
                for worker in optimizer.keys():
                    kwargs = {"lr": args.lr, "weight_decay": args.weight_decay}
                    ## TODO implement for SGD (also in train_federated)
                    kwargs["betas"] = (args.beta1, args.beta2)
                    optimizer[worker] = torch.optim.Adam(
                        models[worker].parameters(), **kwargs
                    )

        models["local_model"] = secure_aggregation(
            models["local_model"],
            models,
            train_loader.keys(),
            crypto_provider,
            args,
            test_params,
        )
        avg_loss = np.mean(avg_loss)

        ##end secure_aggregation_epoch()
        print("Train Epoch: {} \tLoss: {:.6f}".format(epoch, avg_loss,))

        if (epoch % args.test_interval) == 0:
            _, roc_auc = test(
                models["local_model"], val_loader, epoch, num_classes, class_names=None,
            )

            roc_auc_scores.append(roc_auc)
    # reversal and formula because we want last occurance of highest value
    roc_auc_scores = np.array(roc_auc_scores)[::-1]
    best_auc_idx = np.argmax(roc_auc_scores)
    highest_acc = len(roc_auc_scores) - best_auc_idx - 1
    print(f"Highest ROC-AUC {roc_auc_scores[best_auc_idx]}")

# best_epoch = (
#     highest_acc + 1
# ) * args.test_interval  # actually -1 but we're switching to 1 indexed here
# # best_model_file = model_paths[highest_acc]
# print(
#     "Highest ROC AUC score was {:.1f}% in epoch {:d}".format(
#         roc_auc_scores[best_auc_idx],
#         best_epoch * (args.repetitions_dataset if args.train_federated else 1),
#     )
# )

