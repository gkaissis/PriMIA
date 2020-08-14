import torch
from torchvision import transforms

import sys, os.path

sys.path.insert(0, os.path.split(sys.path[0])[0])
from torchlib.dataloader import PPPP

if __name__ == "__main__":

    dataset = PPPP(
        train=False,
        transform=transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.57282609,), (0.17427578,)),
                transforms.Lambda(
                    lambda x: torch.repeat_interleave(  # pylint:disable=no-member
                        x, 3, dim=0
                    )
                ),
            ]
        ),
    )
    data, target = [], []
    for d, t in dataset:
        data.append(d)
        target.append(t)
    data = torch.stack(data)  # pylint:disable=no-member
    target = torch.tensor(target)  # pylint:disable=not-callable
    torch.save(data, "data/testdata.pt")
    torch.save(target, "data/testlabels.pt")
