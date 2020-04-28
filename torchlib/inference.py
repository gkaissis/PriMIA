"""
This implementation is based on the pysyft tutorial:
https://github.com/OpenMined/PySyft/blob/master/examples/tutorials/Part%2011%20-%20Secure%20Deep%20Learning%20Classification.ipynb
"""


import torch
import configparser
import argparse
import syft as sy
from torchvision import datasets, transforms, models
import sys, os.path

sys.path.insert(0, os.path.split(sys.path[0])[0])  # TODO: make prettier
from train_federated import test, Arguments
from torchlib.dataloader import PPPP
from torchlib.models import vgg16


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        default="configs/pneumonia.ini",
        help="Path to config",
    )
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

    config = configparser.ConfigParser()
    config.read(cmd_args.config)

    args = Arguments(cmd_args, config, mode="inference")

    if cmd_args.encrypted_inference:
        hook = sy.TorchHook(torch)
        client = sy.VirtualWorker(hook, id="client")
        bob = sy.VirtualWorker(hook, id="bob")
        alice = sy.VirtualWorker(hook, id="alice")
        crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")  # pylint: disable=no-member

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
        tf = transforms.Compose(
            [
                transforms.Resize(args.inference_resolution),
                transforms.CenterCrop(args.inference_resolution),
                transforms.ToTensor(),
            ]
        )
        testset = PPPP("data/Labels.csv", train=False, transform=tf)
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
    # model = Net().to(device)
    model = vgg16(pretrained=False, num_classes=num_classes, in_channels=1)
    state = torch.load(cmd_args.model_weights, map_location=device)
    if "optim_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    # model = models.vgg16(pretrained=False, num_classes=3)
    # model.classifier = vggclassifier()
    model.to(device)
    if args.encrypted_inference:
        model.fix_precision().share(alice, bob, crypto_provider=crypto_provider)
    loss_fn = lambda x, y: torch.Tensor([0])
    test(args, model, device, test_loader, 0, loss_fn, num_classes)
