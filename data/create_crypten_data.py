"""Convenience script to generate and save data for the CrypTen benchmark. Should not be modified. Reads data/test and saves the images and the corresponding labels to .pt files for usage with the crypten_inference.py script.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import sys, os.path

sys.path.insert(0, os.path.split(sys.path[0])[0])

from torchlib.dicomtools import CombinedLoader
from torchlib.dataloader import calc_mean_std

if __name__ == "__main__":

    dataset = ImageFolder(
        "data/test",
        transform=transforms.Compose(
            [transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),]
        ),
        loader=CombinedLoader(),
    )
    mean, std = calc_mean_std(dataset)
    dataset.transform.transforms.append(transforms.Normalize(mean, std))
    data, target = [], []
    for d, t in tqdm(dataset, total=len(dataset), leave=False, desc="Load data"):
        data.append(d)
        target.append(t)
    data = torch.stack(data)  # pylint:disable=no-member
    target = torch.tensor(target)  # pylint:disable=not-callable
    torch.save(data, "data/testdata.pt")
    torch.save(target, "data/testlabels.pt")
