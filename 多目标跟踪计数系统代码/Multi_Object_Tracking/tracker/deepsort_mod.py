from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
from ..utils.geometry import iou_matrix, crop_bboxes_safely
from .basetrack import BaseTrack, TrackState
from ..reid import EnhancedReID
from .feature_bank import FeatureBank


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
        import cv2
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
