import os
import random
import syft as sy
import pandas as pd
import numpy as np
from PIL import Image
from torch import manual_seed
from torch.utils import data
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.datasets.folder import default_loader


def single_channel_loader(filename):
    with open(filename, "rb") as f:
        img = Image.open(f).convert("L")
        return img.copy()


class LabelMNIST(MNIST):
    def __init__(self, labels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        indices = np.isin(self.targets, labels).astype("bool")
        self.data = self.data[indices]
        self.targets = self.targets[indices]


class PPPP(data.Dataset):
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
    Works only if not torch.utils.data.random_split is applied
    """

    def get_class_occurances(self):
        return dict(self.labels["Numeric_Label"].value_counts())

    def __compute_mean_std__(self, crop_size=224, consider_black_pixels=False):
        import numpy as np
        from tqdm import tqdm
        from torch.utils.data import DataLoader

        batch_size = 90
        num_workers = 8

        loader = DataLoader(self, batch_size=batch_size, num_workers=num_workers)
        acc = np.zeros((3, crop_size, crop_size))
        sq_acc = np.zeros((3, crop_size, crop_size))
        n_black_pixels = 0
        for _, (imgs, _) in tqdm(
            enumerate(loader),
            total=len(loader),
            leave=False,
            desc="accumulating data",
        ):
            imgs = imgs.numpy()
            acc += np.sum(imgs, axis=0)
            sq_acc += np.sum(imgs ** 2, axis=0)
            n_black_pixels += np.where(imgs == 0)[0].size

            # if batch_idx % 50 == 0:
            #    print('Accumulated {:d} / {:d}'.format(
            #        batch_idx * batch_size, len(dset)))

        N = len(self) * acc.shape[1] * acc.shape[2]
        if not consider_black_pixels:
            print("{:d} pixels in dataset".format(N))
            N -= n_black_pixels
            print("{:d} black pixels in dataset".format(n_black_pixels))

        mean_p = np.asarray([np.sum(acc[c]) for c in range(3)])
        mean_p /= N
        print("Mean pixel = ", mean_p)

        # std = E[x^2] - E[x]^2
        var = np.asarray([np.sum(sq_acc[c]) for c in range(3)])
        var /= N
        var -= mean_p ** 2
        var = np.sqrt(var)
        print("Var. pixel = ", var)
        np.savetxt(
            os.path.join("data/mean_var.txt"), np.vstack((mean_p, var)), fmt="%8.7f"
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

    ds.__compute_mean_std__(consider_black_pixels=False)
    L = len(ds)
    print("length train set: {:d}".format(L))

    from matplotlib import pyplot as plt

    ds = PPPP()
    hist = ds.labels.hist(bins=3, column="Numeric_Label")
    plt.show()
