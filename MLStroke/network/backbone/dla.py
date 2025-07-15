import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class DummyDLANet(nn.Module):
    def __init__(self, heads, head_conv=64):
        super(DummyDLANet, self).__init__()
        self.base = nn.Sequential(
            BasicBlock(3, 32),
            BasicBlock(32, 64),
            BasicBlock(64, 128),
        )
        self.heads = heads
        self.output_heads = nn.ModuleDict()
        for head, num_classes in heads.items():
            self.output_heads[head] = nn.Sequential(
                nn.Conv2d(128, head_conv, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, num_classes, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, x):
        features = self.base(x)
        out = {}
        for head in self.heads:
            out[head] = self.output_heads[head](features)
        return out, features

def DLASeg(name, heads, pretrained, down_ratio, final_kernel, last_level, head_conv, use_dcn):
    # 简化版接口，保持一致性
    return DummyDLANet(heads, head_conv=head_conv)
