import torch
from torch import nn
import torchsummary
from torchvision import models
import sys
import os.path

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
from torchlib.models import Net, vgg16, resnet18




if __name__ == "__main__":
    #model = Net()
    #model = vgg16(num_classes=3)
    model = resnet18(num_classes=3)
    #model.classifier = vggclassifier()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    torchsummary.summary(model, (3, 224, 224), batch_size=64)
