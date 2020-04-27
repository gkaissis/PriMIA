import torch
import configparser
import argparse
from torchvision import datasets, transforms, models
from common.dataloader import PPPP
from common.models import vgg16
from train_federated import test, Arguments


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
    cmd_args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(cmd_args.config)



    args = Arguments(cmd_args, config)

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
        testset = PPPP("Labels.csv", train=False, transform=tf)
    else:
        raise NotImplementedError("dataset not implemented")

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size, shuffle=True, **kwargs
    )
    # model = Net().to(device)
    model = vgg16(pretrained=False, num_classes=num_classes, in_channels=1)
    state = torch.load(cmd_args.model_weights)
    if "optim_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    # model = models.vgg16(pretrained=False, num_classes=3)
    # model.classifier = vggclassifier()
    model.to(device)
    loss_fn = lambda x: 0
    test(args, model, device, test_loader, 0, loss_fn, num_classes)
