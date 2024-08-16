from torchvision.models import resnet18, ResNet18_Weights
from torch import nn
import torch

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)

class Block(nn.Module):
    def __init__(self, inplanes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.layer1 = self._make_layer(64)
        self.layer2 = self._make_layer(128, stride=2)
        self.layer3 = self._make_layer(256, stride=2)
        self.layer4 = self._make_layer(512, stride=2)
        self.fc = nn.Linear(512, n_embd, bias=False)
        nn.init.xavier_normal_(self.fc.weight)

    def _make_layer(self, planes: int, stride: int = 1):
        layers = []
        layers.append(Block(self.inplanes, planes, stride))
        self.inplanes = planes
        layers.append(Block(self.inplanes, planes))
        return nn.Sequential(*layers)
    
    def init_from_pretrain(self):
        rn = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        state_dict = rn.state_dict()
        state_dict.pop('fc.weight')
        state_dict.pop('fc.bias')
        self.load_state_dict(state_dict, strict=False)
        del rn
        del state_dict
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # (B, C, H, W) -> (B, H*W, C)
        x = x.permute(0, 2, 3, 1)
        x = x.view(x.shape[0], -1, x.shape[3])
        x = self.fc(x)
        return x


if __name__ == '__main__':
    model = ResNet18(n_embd=384)
    resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    state_dict = resnet.state_dict()
    state_dict.pop('fc.weight')
    state_dict.pop('fc.bias')
    breakpoint()

