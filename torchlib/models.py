import torch


from syft import Plan
from numpy import prod
from torch import nn
from torch.hub import load_state_dict_from_url
from numpy.random import seed as npseed
from random import seed as rseed
import os, pickle


model_urls = {
    "vgg11": "https://download.pytorch.org/models/vgg11-bbd30ac9.pth",
    "vgg13": "https://download.pytorch.org/models/vgg13-c768596a.pth",
    "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",
    "vgg19": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
    "vgg11_bn": "https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
    "vgg13_bn": "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
    "vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
    "vgg19_bn": "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}
cfgs = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class identity(nn.Module):
    def __init__(self):
        super(identity, self).__init__()

    def forward(self, x):
        return x


class VGG(nn.Module):
    def __init__(
        self,
        features,
        num_classes=1000,
        init_weights=True,
        adptpool=True,
        input_size=224,
    ):
        super(VGG, self).__init__()
        self.features = features
        if adptpool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.avgpool = nn.AvgPool2d(
                int(input_size / 32)
            )  # only works for input of size (224, 224)
            # print("No avg pooling")
        self.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
            # nn.LogSoftmax(dim=1),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # pylint: disable=no-member
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False, in_channels=3, pooling="avg"):
    layers = []
    for v in cfg:
        if v == "M":
            if pooling == "avg":
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            elif pooling == "max":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                raise NotImplementedError(
                    "pooling type unknown: {:s}".format(str(pooling))
                )
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def _vgg(
    arch,
    cfg,
    batch_norm,
    pretrained,
    progress,
    in_channels=3,
    pooling="avg",
    num_classes=1000,
    **kwargs
):
    if pretrained:
        kwargs["init_weights"] = False
    assert not (pretrained and in_channels != 3), "If pretrained you need 3 in channels"
    model = VGG(
        make_layers(
            cfgs[cfg], batch_norm=batch_norm, in_channels=in_channels, pooling=pooling
        ),
        **kwargs
    )
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    if num_classes != 1000:
        model.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
            # nn.LogSoftmax(dim=1),
        )
    return model


def vgg16(pretrained=False, progress=True, in_channels=3, pooling="avg", **kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg(
        "vgg16",
        "D",
        False,
        pretrained,
        progress,
        in_channels=in_channels,
        pooling=pooling,
        **kwargs
    )


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        in_channels=3,
        adptpool=True,
        input_size=224,
        pooling="avg",
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        if pooling == "max":
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        elif pooling == "avg":
            self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        else:
            raise NotImplementedError("pooling type unknown: {:s}".format(str(pooling)))
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = (
            nn.AdaptiveAvgPool2d((1, 1))
            if adptpool
            else nn.AvgPool2d(int(input_size / 32))
        )
        self.fc = nn.Linear(512 * block.expansion, 1000)
        # self.sm = nn.LogSoftmax(dim=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # pylint: disable=no-member
        x = self.fc(x)
        # x = self.sm(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(
    arch, block, layers, pretrained, progress, num_classes, pooling="avg", **kwargs
):
    model = ResNet(block, layers, pooling=pooling, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    model.fc = nn.Linear(512 * block.expansion, num_classes)
    return model


def resnet18(pretrained=False, progress=True, in_channels=3, pooling="avg", **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet18",
        BasicBlock,
        [2, 2, 2, 2],
        pretrained,
        progress,
        in_channels=in_channels,
        pooling=pooling,
        **kwargs
    )


def resnet34(pretrained=False, progress=True, in_channels=3, pooling="avg", **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet34",
        BasicBlock,
        [3, 4, 6, 3],
        pretrained,
        progress,
        in_channels=in_channels,
        pooling=pooling,
        **kwargs
    )


def _initialize_weights(model):
    torch.manual_seed(1)
    rseed(1)
    npseed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)


class ConvNet512(nn.Module):
    def __init__(self, num_classes=10, in_channels=1, pooling="avg"):
        super(ConvNet512, self).__init__()
        if pooling == "avg":
            pool_layer = nn.AvgPool2d
        elif pooling == "max":
            pool_layer = nn.MaxPool2d
        else:
            raise NotImplementedError("pooling type unknown: {:s}".format(str(pooling)))
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, 3),
            nn.ReLU(),
            pool_layer(2),
            pool_layer(2),
            nn.Conv2d(8, 32, 3),
            nn.ReLU(),
            pool_layer(2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            pool_layer(2),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            pool_layer(2),
            nn.Conv2d(128, 256, 3),
            nn.ReLU(),
            pool_layer(2),
            nn.Conv2d(256, 512, 3),
            nn.ReLU(),
            pool_layer(2),
            pool_layer(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )
        _initialize_weights(self)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 512)
        x = self.classifier(x)
        return x


class ConvNet224(nn.Module):
    def __init__(self, num_classes=10, in_channels=1, pooling="avg"):
        super(ConvNet224, self).__init__()
        if pooling == "avg":
            pool_layer = nn.AvgPool2d
        elif pooling == "max":
            pool_layer = nn.MaxPool2d
        else:
            raise NotImplementedError("pooling type unknown: {:s}".format(str(pooling)))
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, 3),
            nn.ReLU(),
            pool_layer(2),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 32, 3),
            nn.ReLU(),
            pool_layer(2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            pool_layer(2),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            pool_layer(2),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 3),
            nn.ReLU(),
            pool_layer(2),
            nn.Conv2d(256, 512, 3),
            nn.ReLU(),
            pool_layer(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )
        _initialize_weights(self)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 512)
        x = self.classifier(x)
        return x


class ConvNetMNIST(nn.Module):
    def __init__(self, num_classes=10, in_channels=1, pooling="avg"):
        super(ConvNetMNIST, self).__init__()
        if pooling == "avg":
            pool_layer = nn.AvgPool2d
        elif pooling == "max":
            pool_layer = nn.MaxPool2d
        else:
            raise NotImplementedError("pooling type unknown: {:s}".format(str(pooling)))
        # self.features = nn.Sequential(
        self.conv1 = nn.Conv2d(in_channels, 8, 3)
        self.relu1 = nn.ReLU()
        # self.bn1 = nn.BatchNorm2d(8),
        self.conv2 = nn.Conv2d(8, 32, 3)
        # self.relu2 = nn.ReLU(),
        self.conv3 = nn.Conv2d(32, 64, 3)
        # self.relu3 = nn.ReLU(),
        # self.bn3 = nn.BatchNorm2d(64),
        self.conv4 = nn.Conv2d(64, 128, 3)
        # self.relu4 = nn.ReLU(),
        self.pool4 = pool_layer(2)
        self.conv5 = nn.Conv2d(128, 256, 3)
        # self.relu5 = nn.ReLU(),
        self.pool5 = pool_layer(2)
        self.conv6 = nn.Conv2d(256, 512, 3)
        # self.relu6 = nn.ReLU(),
        self.pool6 = pool_layer(2)
        # )
        # self.classifier = nn.Sequential(
        self.linear1 = nn.Linear(512, 512)
        # nn.ReLU(),
        self.linear2 = nn.Linear(512, 512)
        # nn.ReLU(),
        self.linear3 = nn.Linear(512, num_classes)
        # )
        # _initialize_weights(self)

    def forward(self, x):  # pylint: disable=method-hidden
        # x = self.features(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu1(x)
        x = self.conv3(x)
        x = self.relu1(x)
        x = self.conv4(x)
        x = self.relu1(x)
        x = self.pool4(x)
        x = self.conv5(x)
        x = self.relu1(x)
        x = self.pool5(x)
        x = self.conv6(x)
        x = self.relu1(x)
        x = self.pool6(x)
        # print(x.size())

        x = x.reshape(-1, 512)
        # x = self.classifier(x)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu1(x)
        x = self.linear3(x)
        return x


conv_at_resolution = {28: ConvNetMNIST, 224: ConvNet224, 512: ConvNet512}

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

"""
    1.  Segmentation Net
"""

from collections import OrderedDict

class SimpleSegNet(nn.Module): 
    def __init__(self): 
        super(SimpleSegNet, self).__init__() 

        # Take pretrained feature extractor - pretrained vgg11 on ImageNet (small for prototyping)
        #self.features = _vgg(arch="vgg11", cfg="A", batch_norm=False, pretrained=True, progress=True, num_classes=23)

        # get VGG using existing functions 

        # problem?! corrupted file - invalid checksum 
        #arch = "vgg11"
        #cfg = "A"

        arch = "vgg16"
        cfg = "D"

        batch_norm = False
        in_channels = 3
        pooling = "avg"
        progress = True 
        #num_classes = 23
        kwargs = {}
        kwargs["init_weights"] = False

        feature_extractor = VGG(
            make_layers(
                cfgs[cfg], batch_norm=batch_norm, in_channels=in_channels, pooling=pooling
            ),
            **kwargs
        )

        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        feature_extractor.load_state_dict(state_dict)

        # create very simple segmentation net using nn.Upsample()
        
        # res of MSRC-v2 pics
        H, W = 240, 240 

        self.model = nn.Sequential(OrderedDict([
                                   ('decoder', nn.Sequential(feature_extractor.features)), 
                                   ('bridge_conv', nn.Conv2d(512, 512, kernel_size=(1, 1))), 
                                   ('bridge_relu', nn.ReLU()), 
                                   ('encoder_1', nn.Upsample(size=(int(H/4), int(W/2)))), 
                                   ('encoder_1_conv', nn.Conv2d(512, 256, kernel_size=(1, 1))), 
                                   ('encoder_1_relu', nn.ReLU()),
                                   ('encoder_2', nn.Upsample(size=(int(H/2), int(W/2)))), 
                                   ('encoder_2_conv', nn.Conv2d(256, 128, kernel_size=(1, 1))), 
                                   ('encoder_2_relu', nn.ReLU()),
                                   ('encoder_3', nn.Upsample(size=(H, W))), 
                                   ('encoder_3_conv', nn.Conv2d(128, 23, kernel_size=(1, 1))), 
                                   ('encoder_3_relu', nn.ReLU())
                                   ]))

    def forward(self, x): 
        out = self.model(x)

        return out 

#from models.MoNet import MoNet

"""
    MoNet - ported from TF 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F 

class ConvBnElu(nn.Module): 
    """
        Conv-Batchnorm-Elu block
    """
    def __init__(
        self,  
        old_filters,
        filters, 
        kernel_size=3, 
        strides=1, 
        dilation_rate=1
        ): 
        super(ConvBnElu, self).__init__()

        # Conv
        # 'SAME' padding => Output-Dim = Input-Dim/stride -> exact calculation: if uneven add more padding to the right
        # int() floors padding
        # TODO: how to add asymmetric padding? tuple option for padding only specifies the different dims 
        same_padding = int(dilation_rate*(kernel_size-1)*0.5)

        # TODO: kernel_initializer="he_uniform",

        self.conv = nn.Conv2d(
            in_channels=old_filters, 
            out_channels=filters, 
            kernel_size=kernel_size, 
            stride=strides, 
            padding=same_padding, 
            dilation=dilation_rate, 
            bias=False)

        # BatchNorm
        self.batch_norm = nn.BatchNorm2d(filters)

    def forward(self, x): 
        out = self.conv(x)
        out = self.batch_norm(out)
        out = F.elu(out)
        return out 

class deconv(nn.Module): 
    """
        Transposed Conv. with BatchNorm and ELU-activation
        Deconv upsampling of x. Doubles x and y dimension and maintains z.
    """
    def __init__(self, old_filters):
        super(deconv, self).__init__() 

        kernel_size = 4
        stride = 2
        dilation_rate = 1

        # TODO: how to add asymmetric padding? possibly use "output_padding here"
        same_padding = int(dilation_rate*(kernel_size-1)*0.5)

        # TODO: kernel_initializer="he_uniform",

        self.transp_conv = nn.ConvTranspose2d(
            in_channels=old_filters, 
            out_channels=old_filters, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=same_padding, 
            bias=False)

        self.batch_norm = nn.BatchNorm2d(old_filters)

    def forward(self, x):
        out = self.transp_conv(x)
        out = self.batch_norm(out)
        out = F.elu(out)
        return out 

class repeat_block(nn.Module): 
    """ 
        RDDC - Block
        Reccurent conv block with decreasing kernel size. 
        Makes use of atrous convolutions to make large kernel sizes computationally feasible

    """
    def __init__(
        self,  
        in_filters,
        out_filters, 
        dropout=0.2
        ): 
        super(repeat_block, self).__init__()

        # Skip connection 
        # TODO: Reformatting necessary?

        self.convBnElu1 = ConvBnElu(in_filters, out_filters, dilation_rate=4)
        self.dropout1 = nn.Dropout2d(dropout)
        self.convBnElu2 = ConvBnElu(out_filters, out_filters, dilation_rate=3)
        self.dropout2 = nn.Dropout2d(dropout)
        self.convBnElu3 = ConvBnElu(out_filters, out_filters, dilation_rate=2)
        self.dropout3 = nn.Dropout2d(dropout)
        self.convBnElu4 = ConvBnElu(out_filters, out_filters, dilation_rate=1)

    def forward(self, x): 
        skip1 = x
        out = self.convBnElu1(x)
        out = self.dropout1(out)
        out = self.convBnElu2(out + skip1)
        out = self.dropout2(out)
        skip2 = out
        out = self.convBnElu3(out)
        out = self.dropout3(out)
        out = self.convBnElu4(out + skip2)

        #TODO: In this implementation there was again a skip connection from first input, not shown in paper however? 
        out = skip1 + out
        return out


class MoNet(nn.Module): 
    def __init__(
        self, 
        input_shape=(1, 256, 256),
        output_classes=1,
        depth=2,
        n_filters_init=16,
        dropout_enc=0.2,
        dropout_dec=0.2,
        activation=None,
        ):
        super(MoNet, self).__init__()
        
        # store param in case they're needed later
        self.input_shape = input_shape
        self.output_classes = output_classes
        self.depth = depth
        self.features = n_filters_init
        self.dropout_enc = dropout_enc
        self.dropout_dec = dropout_dec
        
        # encoder
        encoder_list = []

        old_filters = 1
        features = n_filters_init
        for i in range(depth):
            encoder_list.append([
                f"Enc_ConvBnElu_Before_{i}", ConvBnElu(old_filters, features)
                ])
            old_filters = features
            encoder_list.append([
                f"Enc_RDDC_{i}", repeat_block(old_filters, features, dropout=dropout_enc)
                ])
            encoder_list.append([
                f"Enc_ConvBnElu_After_{i}", ConvBnElu(old_filters, features, kernel_size=4, strides=2)
                ])
            features *= 2

        # ModulList instead of Sequential because we don't want the layers to be connected yet 
        # we still need to add the skip connections. Dict to know when to add skip connection in forward
        self.encoder = nn.ModuleDict(encoder_list)

        # bottleneck
        bottleneck_list = []
        bottleneck_list.append(ConvBnElu(old_filters, features))
        old_filters = features
        bottleneck_list.append(repeat_block(old_filters, features))

        self.bottleneck = nn.Sequential(*bottleneck_list)

        # decoder
        decoder_list = []
        for i in reversed(range(depth)):
            features //= 2
            decoder_list.append([
                f"Dec_deconv_Before_{i}", deconv(old_filters)
                ])
            # deconv maintains number of channels 
            decoder_list.append([
                f"Dec_ConvBnElu_{i}", ConvBnElu(old_filters, features)
                ])
            old_filters = features
            decoder_list.append([
                f"Dec_RDDC_{i}", repeat_block(old_filters, features, dropout=dropout_dec)
                ])

        self.decoder = nn.ModuleDict(decoder_list)

        # head
        head_list = []
        # TODO: kernel_initializer="he_uniform",
        head_list.append(nn.Conv2d(
            in_channels=old_filters, 
            out_channels=output_classes, 
            kernel_size=1, 
            stride=1, 
            bias=False))

        head_list.append(nn.BatchNorm2d(output_classes))

        # TODO: Consider nn.logsoftmax --> works with NLLoss out of the box --> what we want to use. 
        #if output_classes > 1:
        #    activation = nn.Softmax(dim=1)
        #else:
        #    activation = nn.Sigmoid()
        #head_list.append(activation)
        # BCELoss doesn't include sigmoid layer (not as in CELoss)
        # BCELoss can't handle negative number so no log-space 
        #activation = nn.Sigmoid()
        #head_list.append(activation)
        # INSTEAD: Added BCEWithLogitsLoss which combines both in a numerically stable way sssss

        self.header = nn.Sequential(*head_list)
        if activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = None

    def forward(self, x): 
        skip = []

        # encoder 
        out = x
        for key in self.encoder: 
            out = self.encoder[key](out)
            if key == "RDDC": 
                skip.append(out)

        # bottleneck 
        out = self.bottleneck(out)

        # decoder
        for key in self.decoder: 
            out = self.decoder[key](out)
            if key == "deconv": 
                # Concatenate along channel-dim (last dim)
                # skip.pop() -> get last element and remove it 
                out = torch.cat((out, skip.pop()), dim=-1)

        # header
        out = self.header(out).squeeze()

        if self.activation:
             out = self.activation(out)

        return out

def getMoNet(pretrained=False, **kwargs): 
    # preprocessing step due to version problem (model was saved from torch 1.7.1)
    PRETRAINED_PATH = os.getcwd() + '/pretrained_models/monet_weights.pickle'
    with open(PRETRAINED_PATH, 'rb') as handle:
        state_dict = pickle.load(handle)
    # Init. MoNet
    model = MoNet(**kwargs)
    if pretrained: 
        # load weights from storage 
        #model.load_state_dict(torch.load(PRETRAINED_PATH))
        model.load_state_dict(state_dict)
    return model