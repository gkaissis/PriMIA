import os
import pandas as pd
from PIL import Image
from torch.utils import data
from torchvision import transforms
from torchvision.datasets.folder import default_loader
import syft as sy


def single_channel_loader(filename):
    with open(filename, 'rb') as f:
        img = Image.open(f)
        return img.copy()


class PPPP(sy.BaseDataset):
    def __init__(self, label_path="Labels.csv", train=False, transform=None):
        self.train = train
        self.labels = pd.read_csv(label_path)
        self.labels = self.labels[
            self.labels["Dataset_type"] == ("TRAIN" if train else "TEST")
        ]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        row = self.labels.iloc[index]
        label = row["Numeric_Label"]
        path = "train" if self.train else "test"
        path = os.path.join(path, row["X_ray_image_name"])
        img = single_channel_loader(path)
        if self.transform:
            img = self.transform(img)
        return img, label

    


if __name__ == "__main__":
    ds = PPPP(transform=transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224)]))
    L = len(ds)
    img, label = ds[1]
    img.show()
    tf = transforms.Compose(
    [transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor()]
    )  # TODO: Add normalization
    ds = PPPP(train=True, transform=tf)
    img, label = ds[0]
    print(img.size())
    print(label)
