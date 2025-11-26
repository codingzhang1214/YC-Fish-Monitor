import torch
import torch.nn as nn


class ConvBNAct(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SE(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(c, c // r, 1, 1, 0)
        self.fc2 = nn.Conv2d(c // r, c, 1, 1, 0)
        self.act = nn.SiLU(inplace=True)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        w = self.pool(x)
        w = self.act(self.fc1(w))
        w = self.gate(self.fc2(w))
        return x * w


class Ghost(nn.Module):
    def __init__(self, c, ratio=2):
        super().__init__()
        self.primary = ConvBNAct(c, c)
        self.cheap = nn.Sequential(
            nn.Conv2d(c, c * (ratio - 1), 1, 1, 0, bias=False),
            nn.BatchNorm2d(c * (ratio - 1)),
            nn.SiLU(inplace=True),
        )
        self.out = ConvBNAct(c * ratio, c, 1, 1, 0)

    def forward(self, x):
        y = self.primary(x)
        z = self.cheap(y)
        y = torch.cat([y, z], dim=1)
        return self.out(y)


class RIVU(nn.Module):
    def __init__(self, c, csp_ratio=0.5, ghost_ratio=2, se_r=16):
        super().__init__()
        c1 = int(c * csp_ratio)
        c2 = c - c1
        self.branch_id = nn.Identity()
        self.branch_ghost = Ghost(c2, ratio=ghost_ratio)
        self.se = SE(c, r=se_r)
        self.gamma = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        c = x.shape[1]
        c1 = c // 2
        x1, x2 = x[:, :c1], x[:, c1:]
        y1 = self.branch_id(x1)
        y2 = self.branch_ghost(x2)
        y = torch.cat([y1, y2], dim=1)
        y = self.se(y)
        return y * self.gamma + self.beta


class RIPL(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv3 = nn.Sequential(nn.Conv2d(c, c, 3, 1, 1, bias=False), nn.BatchNorm2d(c))
        self.conv1 = nn.Sequential(nn.Conv2d(c, c, 1, 1, 0, bias=False), nn.BatchNorm2d(c))
        self.id = nn.BatchNorm2d(c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        y = self.conv3(x) + self.conv1(x) + self.id(x)
        return self.act(y)

