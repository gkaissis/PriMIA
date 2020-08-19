import os
import random
import syft as sy
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch import (  # pylint:disable=no-name-in-module
    manual_seed,
    stack,
    cat,
    std_mean,
    save,
    is_tensor,
    from_numpy,
)
import albumentations as a
from torch.utils import data as torchdata
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.datasets.folder import default_loader


class AlbumentationsTorchTransform:
    def __init__(self, transform, **kwargs):
        # print("init albu transform wrapper")
        self.transform = transform
        self.kwargs = kwargs

    def __call__(self, img):
        # print("call albu transform wrapper")
        if Image.isImageType(img):
            img = np.array(img)
        elif is_tensor(img):
            img = img.cpu().numpy()
        img = self.transform(image=img, **self.kwargs)["image"]
        # if img.max() > 1:
        #     img = a.augmentations.functional.to_float(img, max_value=255)
        img = from_numpy(img)
        if img.shape[-1] < img.shape[0]:
            img = img.permute(2, 0, 1)
        return img


def calc_mean_std(dataset, save_folder=None):
    """
    Calculates the mean and standard deviation of `dataset` and
    saves them to `save_folder`.

    Needs a dataset where all images have the same size
    """
    accumulated_data = []
    for d in tqdm(
        dataset, total=len(dataset), leave=False, desc="accumulate data in dataset"
    ):
        if type(d) is tuple or type(d) is list:
            d = d[0]
        accumulated_data.append(d)
    if isinstance(dataset, torchdata.Dataset):
        accumulated_data = stack(accumulated_data)
    elif isinstance(dataset, torchdata.DataLoader):
        accumulated_data = cat(accumulated_data)
    else:
        raise NotImplementedError("don't know how to process this data input class")
    dims = (0, *range(2, len(accumulated_data.shape)))
    std, mean = std_mean(accumulated_data, dim=dims)
    if save_folder:
        save(stack([mean, std]), os.path.join(save_folder, "mean_std.pt"))
    return mean, std


def single_channel_loader(filename):
    """Converts `filename` to a grayscale PIL Image
    """
    with open(filename, "rb") as f:
        img = Image.open(f).convert("L")
        return img.copy()


class LabelMNIST(MNIST):
    def __init__(self, labels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        indices = np.isin(self.targets, labels).astype("bool")
        self.data = self.data[indices]
        self.targets = self.targets[indices]


class ImageFolderFromCSV(torchdata.Dataset):
    def __init__(
        self, csv_path, img_folder_path, transform=None, target_transform=None
    ):
        super().__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.img_folder_path = img_folder_path
        self.img_files = [
            i for i in os.listdir(img_folder_path) if not i.startswith(".")
        ]

        metastats = pd.read_csv(csv_path)

        metastats["class_label"] = metastats.apply(
            ImageFolderFromCSV.__meta_to_class__, axis=1
        )
        self.categorize_dict = dict(
            zip(metastats.X_ray_image_name, metastats.class_label)
        )
        for img in self.img_files:
            assert (
                img in self.categorize_dict.keys()
            ), "img label not known {:s}".format(str(img))
            if self.categorize_dict[img] == -1:
                self.img_files.remove(img)
                print("Ignore image {:s} because category is certain".format(img))

    @staticmethod
    def __meta_to_class__(row):
        if row["Label"] == "Normal":
            return 0
        if row["Label"] == "Pnemonia":  # i know this is a typo but was in original csv
            if row["Label_1_Virus_category"] == "bacteria":
                return 1
            if row["Label_1_Virus_category"] == "Virus":
                return 2
        return -1

    def __getitem__(self, i):
        img_path = self.img_files[i]
        label = self.categorize_dict[img_path]
        img = single_channel_loader(os.path.join(self.img_folder_path, img_path))
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        return img, label

    def __len__(self):
        return len(self.img_files)


class PPPP(torchdata.Dataset):
    def __init__(
        self, label_path="data/Labels.csv", train=False, transform=None, seed=1,
    ):
        super().__init__()
        random.seed(seed)
        manual_seed(seed)
        self.train = train
        self.labels = pd.read_csv(label_path)
        self.labels = self.labels[
            self.labels["Dataset_type"] == ("TRAIN" if train else "TEST")
        ]
        self.transform = transform
        """
        Split into train and validation set
        if self.train:
            indices = [
                i
                for i in range(len(self.labels))
                if ((i % self.val_split) != 0 and self.val)
                or (not self.val and (i % self.val_split) == 0)
            ]
            self.labels = self.labels.drop(index=indices)
        """

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        row = self.labels.iloc[index]
        label = row["Numeric_Label"]
        path = "train" if self.train else "test"
        path = os.path.join("data", path, row["X_ray_image_name"])
        img = single_channel_loader(path)
        if self.transform:
            img = self.transform(img)
        return img, label

    # def get_class_name(self, numeric_label):
    #    return self.class_names[numeric_label]

    """
    Works only if not torch.utils.torchdata.random_split is applied
    """

    def get_class_occurances(self):
        return dict(self.labels["Numeric_Label"].value_counts())

    def __compute_mean_std__(self):

        calc_mean_std(
            self, save_folder="data",
        )


if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    import sys
    from tqdm import tqdm
    import numpy as np

    sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
    )
    from torchlib.utils import AddGaussianNoise

    ds = PPPP(train=True, transform=transforms.ToTensor())
    print("Class distribution")
    print(ds.get_class_occurances())

    sizes = []

    for data, _ in tqdm(ds, total=len(ds), leave=False):
        sizes.append(data.size()[1:])
    sizes = np.array(sizes)
    print(
        "data resolution stats: \n\tmin: {:s}\n\tmax: {:s}\n\tmean: {:s}\n\tmedian: {:s}".format(
            str(np.min(sizes, axis=0)),
            str(np.max(sizes, axis=0)),
            str(np.mean(sizes, axis=0)),
            str(np.median(sizes, axis=0)),
        )
    )

    ds = PPPP(train=False)

    L = len(ds)
    print("length test set: {:d}".format(L))
    img, label = ds[1]
    img.show()

    tf = transforms.Compose(
        [transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),]
    )  # TODO: Add normalization
    ds = PPPP(train=True, transform=tf)

    ds.__compute_mean_std__()
    L = len(ds)
    print("length train set: {:d}".format(L))

    from matplotlib import pyplot as plt

    ds = PPPP()
    hist = ds.labels.hist(bins=3, column="Numeric_Label")
    plt.show()
