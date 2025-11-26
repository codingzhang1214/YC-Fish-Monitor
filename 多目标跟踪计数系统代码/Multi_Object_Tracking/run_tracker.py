#!/usr/bin/env python3
import argparse
import csv
import os
from collections import defaultdict
import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

from .tracker.deepsort_mod import DeepSortMOD, DeepSortArgs


def load_detections_csv(path: str):
    per_frame = defaultdict(list)
    with open(path, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            frame = int(row[0])
            vals = list(map(float, row[1:6])) + [int(row[6])]
            per_frame[frame].append(vals)
    return per_frame


def main():
    parser = argparse.ArgumentParser(description='runner')
    parser.add_argument('--video', type=str, default='')
    parser.add_argument('--dets', type=str, default='')
    parser.add_argument('--save', type=str, default='')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    H, W = 480, 640
    detections = defaultdict(list)
    if args.dets and os.path.isfile(args.dets):
        detections = load_detections_csv(args.dets)
    tracker = DeepSortMOD(DeepSortArgs(device='cpu'))
    cap = None
    if args.video and cv2 is not None and os.path.isfile(args.video):
        cap = cv2.VideoCapture(args.video)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = None
    if args.save and cv2 is not None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vw = cv2.VideoWriter(args.save, fourcc, 30, (W, H))
    frame_idx = 0
    while True:
        if cap is not None:
            ok, im = cap.read()
            if not ok:
                break
        else:
            if frame_idx > 300:
                break
            im = np.zeros((H, W, 3), dtype=np.uint8)
        if detections:
            det_list = np.array(detections.get(frame_idx, []), dtype=float)
        else:
            x = 40 + 3 * frame_idx
            det_list = np.array([[x, 100, x + 40, 170, 0.9, 0]], dtype=float)
        tracks = tracker.update(det_list, im)
        if cv2 is not None:
            for x1, y1, x2, y2, conf, cls, tid in tracks:
                color = (0, 255, 0)
                cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(im, f'ID:{int(tid)} C:{int(cls)} {conf:.2f}', (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            if args.show:
                cv2.imshow('deepsort_mod', im)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            if vw is not None:
                vw.write(im)
        else:
            print(frame_idx, tracks)
        frame_idx += 1
    if cap is not None:
        cap.release()
    if vw is not None:
        vw.release()
    if args.show and cv2 is not None:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
