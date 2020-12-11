from tqdm import tqdm

import torch as th
import syft as sy
from torchvision import datasets, transforms
from opacus import PrivacyEngine, utils

from os.path import dirname, abspath
from sys import path as syspath
from inspect import getfile, currentframe
from torchvision.datasets.folder import default_loader

import albumentations as a
from PIL import Image
from numpy import array

currentdir = dirname(abspath(getfile(currentframe())))
parentdir = dirname(currentdir)
syspath.insert(0, parentdir)
from torchlib.models import (
    conv_at_resolution,  # pylint:disable=import-error
    resnet18,
    vgg16,
)

batch_size = 1


class AlbumentationsTorchTransform:
    def __init__(self, transform, **kwargs):
        # print("init albu transform wrapper")
        self.transform = transform
        self.kwargs = kwargs

    def __call__(self, img):
        # print("call albu transform wrapper")
        if Image.isImageType(img):
            img = array(img)
        elif th.is_tensor(img):
            img = img.cpu().numpy()
        img = self.transform(image=img, **self.kwargs)["image"]
        # if img.max() > 1:
        #     img = a.augmentations.functional.to_float(img, max_value=255)
        img = th.from_numpy(img)
        if img.shape[-1] < img.shape[0]:
            img = img.permute(2, 0, 1)
        return img


def make_model():
    # model_args = {
    #     "pretrained": True,
    #     "num_classes": 3,
    #     "in_channels": 3,
    #     "adptpool": False,
    #     "input_size": 224,
    #     "pooling": "max",
    # }
    # model = resnet18(**model_args)
    model_args = {
        "num_classes": 3,
        "in_channels": 3,
        "pooling": "max",
    }
    model = conv_at_resolution[224](**model_args)
    # model = utils.module_modification.convert_batchnorm_modules(model)
    bn_layers = []
    for layer_name, layer in model.named_modules():
        if isinstance(layer, th.nn.modules.batchnorm.BatchNorm2d):
            bn_layers.append(layer_name)
    for bn in bn_layers:
        layers = bn.split(".")
        bn = model
        semi_last_layer = model
        for i, l in enumerate(layers):
            bn = bn.__getattr__(l)
            if i == len(layers) - 2:
                semi_last_layer = bn
        semi_last_layer.__setattr__(
            layers[-1],
            th.nn.GroupNorm(
                num_channels=bn.num_features,
                num_groups=min(32, bn.num_features,),
                affine=True,
            ),
        )
    return model


hook = sy.TorchHook(th)
alice = sy.VirtualWorker(hook, id="alice")
bob = sy.VirtualWorker(hook, id="bob")
workers = [alice, bob]

sy.local_worker.is_client_worker = False

# TODO: add transforms
stats_tf = AlbumentationsTorchTransform(
    a.Compose([a.Resize(224, 224), a.RandomCrop(224, 224), a.ToFloat(max_value=255.0),])
)
dataset = datasets.ImageFolder("data/train", transform=stats_tf, loader=default_loader,)
train_datasets = dataset.federate(*workers)


# the local version that we will use to do the aggregation
local_model = make_model()

models, dataloaders, optimizers, privacy_engines = [], [], [], []
for worker in workers:
    model = make_model()
    optimizer = th.optim.SGD(model.parameters(), lr=0.1)
    model.send(worker)
    dataset = train_datasets[worker.id]
    dataloader = th.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    privacy_engine = PrivacyEngine(
        model,
        batch_size=batch_size,
        sample_size=len(dataset),
        alphas=range(2, 32),
        noise_multiplier=1.2,
        max_grad_norm=1.0,
    )
    privacy_engine.attach(optimizer)

    models.append(model)
    dataloaders.append(dataloader)
    optimizers.append(optimizer)
    privacy_engines.append(privacy_engine)


def send_new_models(local_model, models):
    with th.no_grad():
        for remote_model in models:
            for new_param, remote_param in zip(
                local_model.parameters(), remote_model.parameters()
            ):
                worker = remote_param.location
                remote_value = new_param.send(worker)
                remote_param.set_(remote_value)


def federated_aggregation(local_model, models):
    with th.no_grad():
        for local_param, *remote_params in zip(
            *([local_model.parameters()] + [model.parameters() for model in models])
        ):
            param_stack = th.zeros(*remote_params[0].shape)
            for remote_param in remote_params:
                param_stack += remote_param.copy().get()
            param_stack /= len(remote_params)
            local_param.set_(param_stack)


def train(epoch, delta):

    # 1. Send new version of the model
    send_new_models(local_model, models)

    # 2. Train remotely the models
    for i, worker in enumerate(workers):
        dataloader = dataloaders[i]
        model = models[i]
        optimizer = optimizers[i]

        model.train()
        criterion = th.nn.CrossEntropyLoss()
        losses = []
        for i, (data, target) in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.get().item())

        sy.local_worker.clear_objects()
        epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(delta)
        print(
            f"[{worker.id}]\t"
            f"Train Epoch: {epoch} \t"
            f"Loss: {sum(losses)/len(losses):.4f} "
            f"(ε = {epsilon:.2f}, δ = {delta}) for α = {best_alpha}"
        )

    # 3. Federated aggregation of the updated models
    federated_aggregation(local_model, models)


for epoch in range(5):
    train(epoch, delta=1e-5)
