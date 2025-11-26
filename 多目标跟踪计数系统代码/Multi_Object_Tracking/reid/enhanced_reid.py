import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import RIVU, RIPL


class EnhancedReID(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, feat_dim=256, use_rivu=True, use_ripl=True, dropout=0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, 2, 1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )
        c = base_ch
        blocks = []
        if use_rivu:
            blocks.append(RIVU(c, expand=1.0))
        if use_ripl:
            blocks.append(RIPL(c, rep=True))
        blocks += [nn.Conv2d(c, c, 3, 2, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
        if use_rivu:
            blocks.append(RIVU(c, expand=1.0))
        if use_ripl:
            blocks.append(RIPL(c, rep=True))
        blocks += [nn.Conv2d(c, c, 3, 2, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
        if use_rivu:
            blocks.append(RIVU(c, expand=1.0))
        if use_ripl:
            blocks.append(RIPL(c, rep=True))
        self.body = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.bnneck = nn.BatchNorm1d(c)
        self.dropout = nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity()
        self.fc = nn.Linear(c, feat_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.body(x)
        x = self.pool(x).flatten(1)
        x = self.bnneck(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x

    @torch.no_grad()
    def extract(self, crops):
        if isinstance(crops, torch.Tensor):
            x = crops
        else:
            raise TypeError("extract expects a 4D Tensor [N,3,H,W]")
        feats = self.forward(x)
        return feats
