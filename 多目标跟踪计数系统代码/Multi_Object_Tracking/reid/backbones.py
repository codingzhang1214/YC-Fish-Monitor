import torch
import torch.nn as nn


class DepthwiseConv(nn.Module):
    def __init__(self, c, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(c, c, k, s, p, groups=c, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class PointwiseConv(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class RIVU(nn.Module):
    def __init__(self, c, expand=1.0):
        super().__init__()
        ce = max(1, int(c * expand))
        self.use_res = (c == ce)
        self.dw = DepthwiseConv(ce if ce != c else c)
        self.pw1 = PointwiseConv(c, ce) if ce != c else nn.Identity()
        self.pw2 = PointwiseConv(ce, c)

    def forward(self, x):
        out = self.pw1(x) if not isinstance(self.pw1, nn.Identity) else x
        out = self.dw(out)
        out = self.pw2(out)
        return x + out


class RIPL(nn.Module):
    def __init__(self, c, rep=True):
        super().__init__()
        self.rep = rep
        self.conv3 = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(c, c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.rep:
            y = self.conv3(x) + self.conv1(x)
        else:
            y = self.conv3(x)
        y = self.act(y)
        return x + y
