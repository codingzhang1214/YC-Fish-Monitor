import argparse
import csv
import json
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import yaml
except ImportError:
    yaml = None

def iou_matrix(a, b):
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=float)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iou = np.zeros((a.shape[0], b.shape[0]), dtype=float)
    for i in range(a.shape[0]):
        xx1 = np.maximum(a[i, 0], b[:, 0])
        yy1 = np.maximum(a[i, 1], b[:, 1])
        xx2 = np.minimum(a[i, 2], b[:, 2])
        yy2 = np.minimum(a[i, 3], b[:, 3])
        w = np.clip(xx2 - xx1, a_min=0, a_max=None)
        h = np.clip(yy2 - yy1, a_min=0, a_max=None)
        inter = w * h
        union = area_a[i] + area_b - inter
        iou[i, :] = np.where(union > 0, inter / union, 0.0)
    return iou


def crop_bboxes_safely(im, boxes):
    h, w = im.shape[:2]
    crops = []
    for x1, y1, x2, y2 in boxes.astype(int):
        x1 = int(max(0, min(w - 1, x1)))
        y1 = int(max(0, min(h - 1, y1)))
        x2 = int(max(0, min(w, x2)))
        y2 = int(max(0, min(h, y2)))
        if x2 > x1 and y2 > y1:
            crops.append(im[y1:y2, x1:x2].copy())
        else:
            crops.append(np.zeros((16, 16, 3), dtype=im.dtype))
    return crops

class TrackState(Enum):
    Tracked = 0
    Lost = 1
    Removed = 2


class BaseTrack:
    _count = 0

    @classmethod
    def next_id(cls):
        cls._count += 1
        return cls._count

class FeatureBank:
    def __init__(self, dim: int, capacity: int = 200, momentum: float = 0.9):
        self.dim = dim
        self.capacity = max(1, int(capacity))
        self.momentum = float(np.clip(momentum, 0.0, 1.0))
        self.queue = []
        self.avg = None

    def add(self, feat: np.ndarray):
        if feat is None or feat.size == 0:
            return
        f = feat.astype(np.float32)
        f = f / (np.linalg.norm(f) + 1e-12)
        self.queue.append(f)
        if len(self.queue) > self.capacity:
            self.queue.pop(0)
        if self.avg is None:
            self.avg = f.copy()
        else:
            self.avg = self.momentum * self.avg + (1 - self.momentum) * f
            self.avg = self.avg / (np.linalg.norm(self.avg) + 1e-12)

    def representation(self, k_last: int = 5) -> np.ndarray:
        if self.queue:
            k = min(k_last, len(self.queue))
            rep = np.mean(self.queue[-k:], axis=0)
            rep = rep / (np.linalg.norm(rep) + 1e-12)
            return rep
        return self.avg if self.avg is not None else np.zeros(self.dim, dtype=np.float32)

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

@dataclass
class DeepSortArgs:
    max_cosine_distance: float = 0.25
    max_iou_distance: float = 0.6
    track_high_thresh: float = 0.30
    track_low_thresh: float = 0.05
    new_track_thresh: float = 0.30
    max_age: int = 40
    n_init: int = 1
    track_buffer: int = 50
    nn_budget: int = 200
    feature_dim: int = 256
    feature_momentum: float = 0.9
    quality_threshold: float = 0.8
    use_rivu: bool = True
    use_ripl: bool = True
    match_iou_weight: float = 0.6
    match_feat_weight: float = 0.4
    adaptive: bool = True
    device: str = 'cpu'


class Tracklet(BaseTrack):
    def __init__(self, tlbr: np.ndarray, score: float, cls_id: int, feat: np.ndarray, feat_dim: int, nn_budget: int, momentum: float = 0.9):
        self._tlbr = tlbr.astype(float)
        self.score = float(score)
        self.cls = int(cls_id)
        self.bank = FeatureBank(dim=feat_dim, capacity=nn_budget, momentum=momentum)
        if feat is not None and feat.size > 0:
            self.bank.add(feat)
        self.state = TrackState.Tracked
        self.is_activated = True
        self.track_id = self.next_id()
        self.end_frame = 0
        self.age = 0
        self.time_since_update = 0

    @property
    def tlbr(self):
        return self._tlbr

    def update(self, tlbr: np.ndarray, score: float, cls_id: int, feat: np.ndarray):
        self._tlbr = tlbr.astype(float)
        self.score = float(score)
        self.cls = int(cls_id)
        if feat is not None and feat.size > 0:
            self.bank.add(feat)
        self.state = TrackState.Tracked
        self.is_activated = True
        self.age = 0
        self.time_since_update = 0

    def mark_lost(self):
        self.state = TrackState.Lost
        self.is_activated = False

    def mark_removed(self):
        self.state = TrackState.Removed
        self.is_activated = False


class DeepSortMOD:
    def __init__(self, args: DeepSortArgs):
        self.args = args
        self.device = torch.device(args.device)
        self.reid = EnhancedReID(feat_dim=args.feature_dim, use_rivu=args.use_rivu, use_ripl=args.use_ripl).to(self.device).eval()
        self.tracked: List[Tracklet] = []
        self.lost: List[Tracklet] = []
        self.removed: List[Tracklet] = []
        self.frame_id = 0

    @torch.no_grad()
    def _extract_features(self, im0: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
        if bboxes.size == 0:
            return np.zeros((0, self.args.feature_dim), dtype=np.float32)
        crops = crop_bboxes_safely(im0, bboxes)
        if cv2 is None:
            raise ImportError("OpenCV is required for feature extraction.")
        xs = []
        for c in crops:
            c = cv2.cvtColor(c, cv2.COLOR_BGR2RGB) if c.ndim == 3 else c
            r = cv2.resize(c, (64, 128))
            r = torch.from_numpy(r).float().permute(2, 0, 1) / 255.0
            xs.append(r)
        x = torch.stack(xs, dim=0).to(self.device)
        feats = self.reid.extract(x).cpu().numpy()
        return feats

    def _cosine_distance(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        if A.size == 0 or B.size == 0:
            return np.ones((A.shape[0], B.shape[0]), dtype=float)
        A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
        B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
        sim = A @ B.T
        return np.clip(1.0 - sim, 0.0, 2.0)

    def _match(self, dets: List[Tracklet], tracks: List[Tracklet]) -> Tuple[List[Tuple[int,int]], List[int], List[int]]:
        if not dets or not tracks:
            return [], list(range(len(dets))), list(range(len(tracks)))
        det_boxes = np.stack([d.tlbr for d in dets], axis=0)
        trk_boxes = np.stack([t.tlbr for t in tracks], axis=0)
        iou = iou_matrix(det_boxes, trk_boxes)
        det_feats = np.stack([d.bank.representation(1) for d in dets], axis=0)
        trk_feats = []
        for t in tracks:
            trk_feats.append(t.bank.representation(5))
        trk_feats = np.stack(trk_feats, axis=0)
        cos = self._cosine_distance(det_feats, trk_feats)
        w_iou = float(self.args.match_iou_weight)
        w_feat = float(self.args.match_feat_weight)
        if self.args.adaptive:
            crowd = max(len(dets), len(tracks))
            alpha = np.clip((crowd - 5) / 10.0, 0.0, 0.5)
            w_iou = np.clip(w_iou + alpha, 0.0, 1.0)
            w_feat = 1.0 - w_iou
        dist = w_iou * (1 - iou) + w_feat * cos
        iou_dist = 1 - iou
        dist[iou_dist > self.args.max_iou_distance] = 1e9
        matches = []
        det_idx = set(range(len(dets)))
        trk_idx = set(range(len(tracks)))
        while det_idx and trk_idx:
            i, j = divmod(dist.argmin(), dist.shape[1])
            if i not in det_idx or j not in trk_idx:
                dist[i, j] = 1e9
                continue
            if dist[i, j] > self.args.max_cosine_distance:
                break
            matches.append((i, j))
            det_idx.remove(i)
            trk_idx.remove(j)
            dist[i, :] = 1e9
            dist[:, j] = 1e9
        u_dets = list(det_idx)
        u_trks = list(trk_idx)
        return matches, u_dets, u_trks

    def update(self, det_results: np.ndarray, im0: np.ndarray = None) -> np.ndarray:
        self.frame_id += 1
        detections = []
        if det_results is None or len(det_results) == 0:
            det_results = np.zeros((0, 6), dtype=float)
        det_results = np.array(det_results, dtype=float)
        if det_results.shape[1] == 4:
            boxes = det_results
            confs = np.ones((len(boxes),), dtype=float) * 0.7
            clss = np.zeros((len(boxes),), dtype=int)
        else:
            boxes = det_results[:, :4]
            confs = det_results[:, 4]
            clss = det_results[:, 5].astype(int)
        feats = self._extract_features(im0, boxes) if im0 is not None and len(boxes) else np.zeros((len(boxes), self.args.feature_dim), dtype=float)
        for i in range(len(boxes)):
            detections.append(Tracklet(boxes[i], confs[i], int(clss[i]), feats[i], self.args.feature_dim, self.args.nn_budget, self.args.feature_momentum))
        high_det_idx = [i for i, d in enumerate(detections) if d.score >= self.args.track_high_thresh]
        high_dets = [detections[i] for i in high_det_idx]
        matches, u_dets, u_trks = self._match(high_dets, self.tracked)
        for (di, tj) in matches:
            det = high_dets[di]
            tr = self.tracked[tj]
            tr.update(det.tlbr, det.score, det.cls, det.bank.representation(1))
        for di in u_dets:
            det = high_dets[di]
            if det.score >= self.args.new_track_thresh:
                self.tracked.append(det)
        for tj in u_trks:
            tr = self.tracked[tj]
            tr.age += 1
            tr.mark_lost()
            self.lost.append(tr)
        self.tracked = [t for t in self.tracked if t.state == TrackState.Tracked]
        low_det_idx = [i for i in range(len(detections)) if i not in high_det_idx and detections[i].score >= self.args.track_low_thresh]
        low_dets = [detections[i] for i in low_det_idx]
        if low_dets and self.lost:
            matches2, u_dets2, u_trks2 = self._match(low_dets, self.lost)
            refind = []
            for (di, lj) in matches2:
                det = low_dets[di]
                tr = self.lost[lj]
                tr.update(det.tlbr, det.score, det.cls, det.bank.representation(1))
                tr.state = TrackState.Tracked
                tr.is_activated = True
                tr.age = 0
                refind.append(tr)
            self.tracked += refind
            keep_lost = []
            for idx, tr in enumerate(self.lost):
                if tr.state == TrackState.Lost:
                    tr.age += 1
                    if tr.age <= self.args.max_age:
                        keep_lost.append(tr)
                    else:
                        tr.mark_removed()
                        self.removed.append(tr)
            self.lost = keep_lost
        outputs = []
        for t in self.tracked:
            x1, y1, x2, y2 = t.tlbr
            outputs.append([x1, y1, x2, y2, t.score, t.cls, t.track_id])
        return np.array(outputs, dtype=float)

def load_deepsort_args(path: str) -> Optional[DeepSortArgs]:
    p = Path(path)
    if not p.exists():
        return None
    data = None
    try:
        if p.suffix.lower() in {'.yml', '.yaml'} and yaml is not None:
            with p.open('r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        elif p.suffix.lower() == '.json':
            with p.open('r', encoding='utf-8') as f:
                data = json.load(f)
    except Exception:
        data = None
    if not isinstance(data, dict):
        return None
    base = DeepSortArgs()
    base_dict = asdict(base)
    for k, v in data.items():
        if k in base_dict:
            setattr(base, k, v)
    return base

def synthetic_detections(frame_idx: int, w: int, h: int):
    boxes = []
    x1 = 50 + frame_idx * 3
    y1 = 100
    x2 = x1 + 40
    y2 = y1 + 60
    boxes.append([x1, y1, x2, y2, 0.9, 0])
    x1b = w - 90 - frame_idx * 2
    y1b = 200
    x2b = x1b + 50
    y2b = y1b + 70
    boxes.append([x1b, y1b, x2b, y2b, 0.85, 1])
    return np.array(boxes, dtype=float)


def main_demo():
    parser = argparse.ArgumentParser(description='Single-file tracker demo')
    parser.add_argument('--frames', type=int, default=120, help='Number of frames to simulate.')
    parser.add_argument('--save', type=str, default='', help='Path to save output video.')
    parser.add_argument('--show', action='store_true', help='Display tracking visualization.')
    parser.add_argument('--config', type=str, default='', help='Path to a YAML/JSON config file for DeepSortArgs.')
    args = parser.parse_args()

    if (args.save or args.show) and cv2 is None:
        print("OpenCV is required for visualization and saving. Please install it (`pip install opencv-python`).")
        return

    ds_args = DeepSortArgs(device='cpu')
    if args.config:
        loaded_args = load_deepsort_args(args.config)
        if loaded_args:
            ds_args = loaded_args
            print(f"Loaded config from: {args.config}")
        else:
            print(f"Warning: Could not load config from {args.config}. Using default arguments.")
    
    H, W = 360, 640
    tracker = DeepSortMOD(ds_args)

    vw = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vw = cv2.VideoWriter(args.save, fourcc, 30, (W, H))
        print(f"Saving output video to: {args.save}")

    for i in range(args.frames):
        im = np.zeros((H, W, 3), dtype=np.uint8)
        dets = synthetic_detections(i, W, H)
        
        start_time = time.time()
        tracks = tracker.update(dets, im)
        end_time = time.time()

        print(f"Frame {i+1}/{args.frames} | Processed in {end_time - start_time:.3f}s | Found {len(tracks)} tracks")

        if (args.save or args.show):
            for x1, y1, x2, y2, conf, cls, tid in tracks:
                color = (0, 255, 0) if int(cls) == 0 else (0, 200, 255)
                cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f'ID:{int(tid)} C:{int(cls)}'
                cv2.putText(im, label, (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            
            if args.show:
                cv2.imshow('Tracking Demo', im)
                if cv2.waitKey(1) & 0xFF == 27: # ESC key to exit
                    break
            if vw is not None:
                vw.write(im)
        
        time.sleep(0.01)

    if vw is not None:
        vw.release()
    if args.show and cv2 is not None:
        cv2.destroyAllWindows()
    print("Demo finished.")


if __name__ == '__main__':
    main_demo()