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


class AIFI(nn.Module):
    def __init__(self, d_model=256, nheads=8):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, nheads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.SiLU(), nn.Linear(d_model * 4, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        b, c, h, w = x.shape
        t = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        attn, _ = self.mha(t, t, t)
        t = self.norm1(t + attn)
        f = self.ffn(t)
        t = self.norm2(t + f)
        return t.reshape(b, h, w, c).permute(0, 3, 1, 2)


class SCALiF(nn.Module):
    def __init__(self, c=256):
        super().__init__()
        self.lateral3 = ConvBNAct(c, c, 1, 1, 0)
        self.lateral4 = ConvBNAct(c, c, 1, 1, 0)
        self.lateral5 = ConvBNAct(c, c, 1, 1, 0)
        self.fp3 = ConvBNAct(c, c)
        self.fp4 = ConvBNAct(c, c)
        self.pan4 = ConvBNAct(c, c, 3, 2, 1)
        self.pan5 = ConvBNAct(c, c, 3, 2, 1)
        self.out4 = ConvBNAct(c * 2, c)
        self.out5 = ConvBNAct(c * 2, c)
        self.aifi = AIFI(c, 8)
        self.mlp = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c * 3, c), nn.SiLU(), nn.Linear(c, 3))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, C3, C4, C5):
        P5 = self.lateral5(C5)
        P4 = self.lateral4(C4) + nn.functional.interpolate(P5, scale_factor=2, mode='nearest')
        P3 = self.lateral3(C3) + nn.functional.interpolate(P4, scale_factor=2, mode='nearest')
        P3 = self.fp3(P3)
        P4 = self.fp4(P4)
        D4 = torch.cat([self.pan4(P3), P4], dim=1)
        D5 = torch.cat([self.pan5(P4), P5], dim=1)
        F4 = self.out4(D4)
        F5 = self.out5(D5)
        A = self.aifi(P5)
        g = torch.cat([nn.functional.adaptive_avg_pool2d(P3, 1), nn.functional.adaptive_avg_pool2d(F4, 1), nn.functional.adaptive_avg_pool2d(F5, 1)], dim=1)
        w = self.softmax(self.mlp(g))
        w1, w2, w3 = w[:, 0].view(-1, 1, 1, 1), w[:, 1].view(-1, 1, 1, 1), w[:, 2].view(-1, 1, 1, 1)
        F3 = w1 * P3
        F4 = w2 * F4
        F5 = w3 * F5
        return F3, F4, F5

