import sys
from torchvision import transforms
from common.dataloader import PPPP


def test_init():
    PPPP('Labels.csv', train=True)
    PPPP('Labels.csv', train=False)
    

def test_len():
    x = PPPP('Labels.csv', train=True)
    y = PPPP('Labels.csv', train=False)
    assert len(x) > len(y), 'Test > train?'


def test_load():
    tf = transforms.Compose(
    [transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor()]
    )  # TODO: Add normalization
    x = PPPP('Labels.csv', train=True, transform=tf)
    img, label = x[0]


