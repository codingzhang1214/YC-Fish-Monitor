import argparse
import time
import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

from .tracker.deepsort_mod import DeepSortMOD, DeepSortArgs


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


def main():
    parser = argparse.ArgumentParser(description='demo')
    parser.add_argument('--frames', type=int, default=120)
    parser.add_argument('--save', type=str, default='')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    H, W = 360, 640
    tracker = DeepSortMOD(DeepSortArgs(device='cpu'))

    vw = None
    if args.save and cv2 is not None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vw = cv2.VideoWriter(args.save, fourcc, 30, (W, H))

    for i in range(args.frames):
        im = np.zeros((H, W, 3), dtype=np.uint8)
        dets = synthetic_detections(i, W, H)
        tracks = tracker.update(dets, im)
        if cv2 is not None:
            for x1, y1, x2, y2, conf, cls, tid in tracks:
                color = (0, 255, 0) if int(cls) == 0 else (0, 200, 255)
                cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(im, f'ID:{int(tid)} C:{int(cls)}', (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            if args.show:
                cv2.imshow('tracking', im)
                cv2.waitKey(1)
            if vw is not None:
                vw.write(im)
        else:
            print(f'Frame {i}:', tracks)
        time.sleep(0.01)

    if vw is not None:
        vw.release()
    if args.show and cv2 is not None:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
