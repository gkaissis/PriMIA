"""
This implementation is based on the pysyft tutorial:
https://github.com/OpenMined/PySyft/blob/master/examples/tutorials/Part%2011%20-%20Secure%20Deep%20Learning%20Classification.ipynb
"""


import torch
import configparser
import argparse
import syft as sy
import sys, os.path
from warnings import warn
from torchvision import datasets, transforms, models

sys.path.insert(0, os.path.split(sys.path[0])[0])  # TODO: make prettier
from utils import test, Arguments
from torchlib.dataloader import PPPP
from torchlib.models import vgg16, resnet18, conv_at_resolution


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    """parser.add_argument(
        "--config",
        type=str,
        required=True,
        default="configs/pneumonia.ini",
        help="Path to config",
    )"""
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        default="pneumonia",
        choices=["pneumonia", "mnist"],
        help="which dataset?",
    )
    parser.add_argument(
        "--model_weights",
        type=str,
        required=True,
        default=None,
        help="Start training from older model checkpoint",
    )
    parser.add_argument(
        "--encrypted_inference", action="store_true", help="Perform encrypted inference"
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="dont use a visdom server"
    )
    cmd_args = parser.parse_args()

    use_cuda = not cmd_args.no_cuda and torch.cuda.is_available()


    device = torch.device("cuda" if use_cuda else "cpu")  # pylint: disable=no-member
    state = torch.load(cmd_args.model_weights, map_location=device)

    args = state['args']
    args.from_previous_checkpoint(cmd_args)

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if cmd_args.encrypted_inference:
        hook = sy.TorchHook(torch)
        client = sy.VirtualWorker(hook, id="client")
        bob = sy.VirtualWorker(hook, id="bob")
        alice = sy.VirtualWorker(hook, id="alice")
        crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")


    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    if args.dataset == "mnist":
        num_classes = 10
        testset = datasets.MNIST(
            "../data",
            train=False,
            transform=transforms.Compose(
                [
                    transforms.Resize(56),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        )
    elif args.dataset == "pneumonia":
        num_classes = 3
        tf = [
            transforms.Resize(args.inference_resolution),
            transforms.CenterCrop(args.inference_resolution),
            transforms.ToTensor(),
            transforms.Normalize((0.57282609,), (0.17427578,)),
        ]

        if args.pretrained:
            repeat = transforms.Lambda(
                lambda x: torch.repeat_interleave(  # pylint: disable=no-member
                    x, 3, dim=0
                )
            )
            tf.append(repeat)
        testset = PPPP("data/Labels.csv", train=False, transform=transforms.Compose(tf))
    else:
        raise NotImplementedError("dataset not implemented")

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size, shuffle=True, **kwargs
    )
    if cmd_args.encrypted_inference:
        priv_test_loader = []
        for data, target in test_loader:
            data.to(device)
            target.to(device)
            priv_test_loader.append(
                (
                    data.fix_prec().share(alice, bob, crypto_provider=crypto_provider),
                    target.fix_prec().share(
                        alice, bob, crypto_provider=crypto_provider
                    ),
                )
            )
        test_loader = priv_test_loader
        del priv_test_loader
    if args.model == "vgg16":
        model = vgg16(
            pretrained=args.pretrained,
            num_classes=num_classes,
            in_channels=3 if args.pretrained else 1,
            adptpool=False,
            input_size=args.inference_resolution,
        )
    elif args.model == "simpleconv":
        if args.pretrained:
            warn("No pretrained version available")
        model = conv_at_resolution[args.train_resolution](
            num_classes=num_classes, in_channels=3 if args.pretrained else 1
        )
    elif args.model == "resnet-18":
        model = resnet18(
            pretrained=args.pretrained,
            num_classes=num_classes,
            in_channels=3 if args.pretrained else 1,
            adptpool=False,
            input_size=args.inference_resolution,
        )
    else:
        raise NotImplementedError("model unknown")
    model.load_state_dict(state["model_state_dict"])
    # model = models.vgg16(pretrained=False, num_classes=3)
    # model.classifier = vggclassifier()
    model.to(device)
    if args.encrypted_inference:
        model.fix_precision().share(alice, bob, crypto_provider=crypto_provider)
    loss_fn = lambda x, y: torch.Tensor([0])
    test(args, model, device, test_loader, 0, loss_fn, num_classes)
