import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from common.dataloader import PPPP
import syft as sy

hook = sy.TorchHook(
    torch
)  # <-- NEW: hook PyTorch ie add extra functionalities to support Federated Learning
bob = sy.VirtualWorker(hook, id="bob")  # <-- NEW: define remote worker bob
alice = sy.VirtualWorker(hook, id="alice")  # <-- NEW: and alice


class Arguments:
    def __init__(self):
        self.batch_size = 64
        self.test_batch_size = 1000
        self.epochs = 10
        self.lr = 0.01
        self.momentum = 0.5
        self.no_cuda = False
        self.seed = 1
        self.log_interval = 10
        self.save_model = False


args = Arguments()

use_cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")  # pylint: disable=no-member

kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

tf = transforms.Compose(
    [transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor()]
)  # TODO: Add normalization

federated_train_loader = sy.FederatedDataLoader(
    PPPP("Labels.csv", train=False, transform=tf).federate((bob, alice)),
    # PPPP(label_path='Labels.csv', train=True, transform=transforms.ToTensor()).federate((bob, alice)),
    batch_size=args.batch_size,
    shuffle=True,
    **kwargs
)

test_loader = torch.utils.data.DataLoader(
    PPPP("Labels.csv", train=False, transform=tf),
    batch_size=args.test_batch_size,
    shuffle=True,
    **kwargs
)


"""class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        #print(x.size())
        #exit()
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
"""
class Net(nn.Module): # TODO: use something better
    def __init__(self, n_classes=3):
        super(Net, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128*5*5, 128),
            nn.LeakyReLU(),
            nn.Linear(128, n_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, 128*5*5)
        x = self.classifier(x)
        return x

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(
        federated_train_loader
    ):  # <-- now it is a distributed dataset
        model.send(data.location)  # <-- NEW: send the model to the right location
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        model.get()  # <-- NEW: get the model back
        if batch_idx % args.log_interval == 0:
            loss = loss.get()  # <-- NEW: get the loss back
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * args.batch_size,
                    len(train_loader)
                    * args.batch_size,  # batch_idx * len(data), len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


if __name__ == "__main__":
    model = Net().to(device)
    #model = models.vgg16().to(device)
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr
    )  # TODO momentum is not supported at the moment

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, federated_train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    if args.save_model:
        torch.save(model.state_dict(), "chestxray.pt")
