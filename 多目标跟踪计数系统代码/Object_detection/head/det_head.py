import torch
import torch.nn as nn


class DecoupledHead(nn.Module):
    def __init__(self, c=256, num_classes=7):
        super().__init__()
        self.cls3 = nn.Sequential(nn.Conv2d(c, c, 3, 1, 1), nn.SiLU(), nn.Conv2d(c, num_classes, 1))
        self.box3 = nn.Sequential(nn.Conv2d(c, c, 3, 1, 1), nn.SiLU(), nn.Conv2d(c, 4, 1))
        self.cls4 = nn.Sequential(nn.Conv2d(c, c, 3, 1, 1), nn.SiLU(), nn.Conv2d(c, num_classes, 1))
        self.box4 = nn.Sequential(nn.Conv2d(c, c, 3, 1, 1), nn.SiLU(), nn.Conv2d(c, 4, 1))
        self.cls5 = nn.Sequential(nn.Conv2d(c, c, 3, 1, 1), nn.SiLU(), nn.Conv2d(c, num_classes, 1))
        self.box5 = nn.Sequential(nn.Conv2d(c, c, 3, 1, 1), nn.SiLU(), nn.Conv2d(c, 4, 1))

    def forward(self, F3, F4, F5):
        out = []
        for F, cls_head, box_head in [(F3, self.cls3, self.box3), (F4, self.cls4, self.box4), (F5, self.cls5, self.box5)]:
            cls = cls_head(F)
            box = box_head(F)
            out.append((cls, box))
        return out

