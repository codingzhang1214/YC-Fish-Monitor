import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbones.rivu import ConvBNAct, RIVU, RIPL
from ..neck.scalif import SCALiF
from ..head.det_head import DecoupledHead
from ..utils.ops import sigmoid, xywh_to_xyxy, nms


class FLiPTrackNetDetector(nn.Module):
    def __init__(self, num_classes=7, base_ch=64):
        super().__init__()
        c = base_ch
        self.stem = ConvBNAct(3, c, 3, 2, 1)
        self.layer2 = nn.Sequential(ConvBNAct(c, c * 2, 3, 2, 1), RIVU(c * 2), RIPL(c * 2))
        self.layer3 = nn.Sequential(ConvBNAct(c * 2, c * 4, 3, 2, 1), RIVU(c * 4), RIPL(c * 4))
        self.layer4 = nn.Sequential(ConvBNAct(c * 4, c * 6, 3, 2, 1), RIVU(c * 6), RIPL(c * 6))
        self.neck = SCALiF(c=256)
        self.align3 = ConvBNAct(c * 4, 256, 1, 1, 0)
        self.align4 = ConvBNAct(c * 6, 256, 1, 1, 0)
        self.align5 = ConvBNAct(c * 6, 256, 1, 1, 0)
        self.head = DecoupledHead(c=256, num_classes=num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        b, _, h, w = x.shape
        s2 = self.stem(x)
        c3 = self.layer2(s2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        C3 = self.align3(c4)
        C4 = self.align4(c5)
        C5 = self.align5(c5)
        F3, F4, F5 = self.neck(C3, C4, C5)
        outs = self.head(F3, F4, F5)
        return outs

    @torch.no_grad()
    def predict(self, img, conf_thresh=0.25, iou_thresh=0.5):
        self.eval()
        if isinstance(img, torch.Tensor):
            x = img
        else:
            import cv2
            x = torch.from_numpy(img[:, :, ::-1]).float().permute(2, 0, 1) / 255.0
            x = x.unsqueeze(0)
        outs = self.forward(x)
        all_boxes = []
        all_scores = []
        all_cls = []
        H, W = x.shape[2], x.shape[3]
        for (cls_logits, box_reg) in outs:
            B, C, Hf, Wf = cls_logits.shape
            cls_scores = sigmoid(cls_logits).permute(0, 2, 3, 1).reshape(B, -1, self.num_classes)
            box_pred = box_reg.permute(0, 2, 3, 1).reshape(B, -1, 4)
            grid_y, grid_x = torch.meshgrid(torch.arange(Hf), torch.arange(Wf), indexing='ij')
            grid = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2).to(x.device).float()
            scale = torch.tensor([Wf, Hf], device=x.device).float()
            cxcy = (grid + sigmoid(box_pred[..., :2])) / scale
            wh = F.relu(box_pred[..., 2:]).clamp(max=1.0)
            xywh = torch.cat([cxcy, wh], dim=-1)
            boxes = xywh_to_xyxy(xywh)
            boxes[..., 0::2] *= W
            boxes[..., 1::2] *= H
            score, cls_idx = cls_scores.max(dim=-1)
            mask = score[0] >= conf_thresh
            all_boxes.append(boxes[0][mask])
            all_scores.append(score[0][mask])
            all_cls.append(cls_idx[0][mask])
        if len(all_boxes) == 0:
            return torch.zeros((0, 6))
        boxes = torch.cat(all_boxes, dim=0)
        scores = torch.cat(all_scores, dim=0)
        clss = torch.cat(all_cls, dim=0)
        keep = nms(boxes, scores, iou_threshold=iou_thresh)
        boxes = boxes[keep]
        scores = scores[keep]
        clss = clss[keep]
        return torch.cat([boxes, scores[:, None], clss.float()[:, None]], dim=1)
