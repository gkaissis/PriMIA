import torch
import torchsummary
from torchvision import models
import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from train_federated import Net

if __name__ == '__main__':
    net = Net()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net.to(device)
    torchsummary.summary(net, (1, 224, 224), batch_size=64)