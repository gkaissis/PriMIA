import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import torch
import configparser
import argparse
import albumentations as a
from torchvision import datasets, transforms, models
from argparse import Namespace
from tqdm import tqdm
from sklearn import metrics as mt
from numpy import newaxis
from torchlib.utils import stats_table, Arguments  # pylint:disable=import-error
from torchlib.models import vgg16, resnet18, conv_at_resolution
from torchlib.dicomtools import CombinedLoader
from torchlib.dataloader import AlbumentationsTorchTransform


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        default="data/server_simulation/test",
        help='Select a data folder [if matches "mnist" mnist will be used].',
    )
    parser.add_argument(
        "--model_weights",
        type=str,
        required=True,
        default=None,
        help="model weights to use",
    )
    parser.add_argument("--cuda", action="store_true", help="Use CUDA acceleration.")
    cmd_args = parser.parse_args()

    use_cuda = cmd_args.cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")  # pylint: disable=no-member
    state = torch.load(cmd_args.model_weights, map_location=device)

    args = state["args"]
    if type(args) is Namespace:
        args = Arguments.from_namespace(args)
    args.from_previous_checkpoint(cmd_args)
    print(str(args))

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    class_names = None
    val_mean_std = (
        state["val_mean_std"]
        if "val_mean_std" in state.keys()
        else (
            torch.tensor([0.5]),  # pylint:disable=not-callable
            torch.tensor([0.2]),  # pylint:disable=not-callable
        )
        if args.pretrained
        else (
            torch.tensor([0.5, 0.5, 0.5]),  # pylint:disable=not-callable
            torch.tensor([0.2, 0.2, 0.2]),  # pylint:disable=not-callable
        )
    )
    mean, std = val_mean_std
    # mean = mean.to(device)
    # std = std.to(device)
    if args.data_dir == "mnist":
        num_classes = 10
        testset = datasets.MNIST(
            "../data",
            train=False,
            transform=transforms.Compose(
                [
                    transforms.Resize(args.inference_resolution),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            ),
        )
    else:
        num_classes = 3

        tf = [
            a.Resize(args.inference_resolution, args.inference_resolution),
            a.CenterCrop(args.inference_resolution, args.inference_resolution),
        ]
        if hasattr(args, "clahe") and args.clahe:
            tf.append(a.CLAHE(always_apply=True))
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
        # transforms.Lambda(lambda x: x.permute(2, 0, 1)),

        loader = CombinedLoader()
        if not args.pretrained:
            loader.change_channels(1)
        testset = datasets.ImageFolder(cmd_args.data_dir, transform=tf, loader=loader)
        assert (
            len(testset.classes) == 3
        ), "We can only handle data that has 3 classes: normal, bacterial and viral"
        class_names = {0: "normal", 1: "bacterial pneumonia", 2: "viral pneumonia"}

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=1, shuffle=True, **kwargs
    )
    if args.model == "vgg16":
        model = vgg16(
            pretrained=args.pretrained,
            num_classes=num_classes,
            in_channels=3 if args.pretrained else 1,
            adptpool=False,
            input_size=args.inference_resolution,
            pooling=args.pooling_type,
        )
    elif args.model == "simpleconv":
        if args.pretrained:
            raise RuntimeError("No pretrained version available")
        model = conv_at_resolution[args.train_resolution](
            num_classes=num_classes,
            in_channels=3 if args.pretrained else 1,
            pooling=args.pooling_type,
        )
    elif args.model == "resnet-18":
        model = resnet18(
            pretrained=args.pretrained,
            num_classes=num_classes,
            in_channels=3 if args.pretrained else 1,
            adptpool=False,
            input_size=args.inference_resolution,
            pooling=args.pooling_type if hasattr(args, "pooling_type") else "avg",
        )
    else:
        raise NotImplementedError("model unknown")
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    # test method
    model.eval()
    total_pred, total_target, total_scores = [], [], []
    with torch.no_grad():
        for data, target in tqdm(
            test_loader,
            total=len(test_loader),
            desc="performing inference",
            leave=False,
        ):
            if len(data.shape) > 4:
                data = data.squeeze(0)
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            total_pred.append(pred)
            total_scores.append(output)
            tgts = target.view_as(pred)
            total_target.append(tgts)
            equal = pred.eq(tgts)
    total_pred = torch.cat(total_pred).cpu().numpy()  # pylint: disable=no-member
    total_target = torch.cat(total_target).cpu().numpy()  # pylint: disable=no-member
    total_scores = torch.cat(total_scores).cpu().numpy()  # pylint: disable=no-member
    total_scores -= total_scores.min(axis=1)[:, newaxis]
    total_scores = total_scores / total_scores.sum(axis=1)[:, newaxis]

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
            epoch=0,
        )
    )

