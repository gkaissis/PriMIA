import torch
from torch import nn
import torchsummary
from torchvision import models
import sys
import os.path

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
from torchlib.models import vgg16, resnet18


if __name__ == "__main__":
    # model = Net()
    model = vgg16(num_classes=3, avgpool=False, in_channels=1)
    #model = resnet18(num_classes=3, in_channels=1)
    # model.classifier = vggclassifier()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    torchsummary.summary(model, (1, 224, 224), batch_size=2)
