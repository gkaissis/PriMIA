import torch


from argparse import ArgumentParser
from os import listdir
import re
import csv
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from itertools import zip_longest
from sys import path as syspath
from os import path as ospath

syspath.append(ospath.abspath(ospath.join(ospath.dirname(__file__), ospath.pardir)))
from torchlib.models import vgg16, resnet18, conv_at_resolution
from torchlib.dataloader import single_channel_loader


class PathDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, loader=default_loader):
        super(PathDataset, self).__init__()
        self.root = root
        self.transform = transform

        self.imgs = [
            f
            for f in listdir(root)
            if re.search(r".*\.(jpg|jpeg|png|JPG|JPEG)$", f) and not f.startswith("._")
        ]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = default_loader(ospath.join(self.root, img_path))
        if self.transform:
            img = self.transform(img)
        return img, img_path


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="data to classify")
    parser.add_argument(
        "--output_file", default="results.csv", help="name of file where its stored"
    )
    parser.add_argument(
        "--model_weights",
        type=str,
        required=True,
        default=None,
        help="model weights to use",
    )
    args = parser.parse_args()

    device = torch.device(  # pylint:disable=no-member
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    state = torch.load(args.model_weights, map_location=device)

    model_args = state["args"]
    # if model_args.data_dir == "pneumonia":
    num_classes = 3
    class_names = ["normal", "bacterial", "viral"]
    # else:
    #     raise NotImplementedError("dataset not supported")
    if model_args.model == "vgg16":
        model = vgg16(
            pretrained=model_args.pretrained,
            num_classes=num_classes,
            in_channels=3 if args.pretrained else 1,
            adptpool=False,
            input_size=args.inference_resolution,
            pooling=model_args.pooling_type,
        )
    elif model_args.model == "simpleconv":
        model = conv_at_resolution[model_args.train_resolution](
            num_classes=num_classes,
            in_channels=3 if model_args.pretrained else 1,
            pooling=model_args.pooling_type,
        )
    elif model_args.model == "resnet-18":
        model = resnet18(
            pretrained=model_args.pretrained,
            num_classes=num_classes,
            in_channels=3 if model_args.pretrained else 1,
            adptpool=False,
            input_size=model_args.inference_resolution,
            pooling=model_args.pooling_type
            if hasattr(model_args, "pooling_type")
            else "avg",
        )
    else:
        raise NotImplementedError("model unknown")
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    mean_std = (
        state["val_mean_std"]
        if "val_mean_std" in state
        else torch.load("data/mean_std.pt")
    )
    tf = transforms.Compose(
        [
            transforms.Lambda(lambda x: adaptive_hist_equalization_on_PIL(x)),
            transforms.Resize(model_args.train_resolution),
            transforms.CenterCrop(model_args.train_resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean_std[0].cpu(), mean_std[1].cpu()),
        ]
    )
    dataset = PathDataset(
        args.data_dir,
        transform=tf,
        loader=default_loader if model_args.pretrained else single_channel_loader,
    )
    result_dict = {name: [] for name in class_names}

    for data, path in tqdm(dataset, total=len(dataset), leave=False, desc="Evaluating"):
        data = data.to(device)
        if len(data.shape) == 3:
            data = data.unsqueeze(0)
        label = model(data).argmax(dim=1)
        label = label.detach().cpu().item()
        result_dict[class_names[label]].append(path)
    print(
        "classified {:s}".format(
            "".join(
                [
                    "{:d} images as {:s} ".format(len(imgs), name)
                    for name, imgs in result_dict.items()
                ]
            )
        )
    )
    with open(args.output_file, "w") as f:
        w = csv.writer(f)
        w.writerow(result_dict.keys())
        w.writerows(zip_longest(*result_dict.values(), fillvalue=""))

