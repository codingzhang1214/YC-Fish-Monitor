import numpy as np


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
