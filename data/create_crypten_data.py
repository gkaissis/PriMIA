import torch
from torchvision import transforms
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
                transforms.Lambda(lambda x: torch.repeat_interleave(x, 3, dim=0)),
            ]
        ),
    )
    data, target = [], []
    for d, t in dataset:
        data.append(d)
        target.append(t)
    data = torch.stack(data)
    target = torch.tensor(target)
    torch.save(data, "data/testdata.pt")
    torch.save(target, "data/testlabels.pt")
