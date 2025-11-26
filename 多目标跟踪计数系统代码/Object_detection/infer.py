import argparse
import numpy as np
import torch

try:
    import cv2
except Exception:
    cv2 = None

from .model.detector import FLiPTrackNetDetector


class DetectorRunner:
    def __init__(self, num_classes=7, device='cpu'):
        self.model = FLiPTrackNetDetector(num_classes=num_classes).to(device).eval()
        self.device = device

    @torch.no_grad()
    def run_image(self, img_bgr, conf=0.25, iou=0.5):
        if not isinstance(img_bgr, np.ndarray):
            raise TypeError('img_bgr must be numpy array BGR')
        img = torch.from_numpy(img_bgr[:, :, ::-1]).float().permute(2, 0, 1) / 255.0
        img = img.unsqueeze(0).to(self.device)
        dets = self.model.predict(img, conf_thresh=conf, iou_thresh=iou)
        return dets.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description='FLiP-TrackNet Detector Inference')
    parser.add_argument('--image', type=str, default='')
    parser.add_argument('--video', type=str, default='')
    parser.add_argument('--save', type=str, default='')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--iou', type=float, default=0.5)
    args = parser.parse_args()

    runner = DetectorRunner(num_classes=7, device='cpu')

    if args.image and cv2 is not None:
        im = cv2.imread(args.image)
        dets = runner.run_image(im, args.conf, args.iou)
        for x1, y1, x2, y2, s, c in dets:
            cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(im, f'{int(c)} {s:.2f}', (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        if args.save:
            cv2.imwrite(args.save, im)
        else:
            cv2.imshow('det', im)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    if args.video and cv2 is not None:
        cap = cv2.VideoCapture(args.video)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vw = None
        if args.save:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            vw = cv2.VideoWriter(args.save, fourcc, 30, (W, H))
        while True:
            ok, im = cap.read()
            if not ok:
                break
            dets = runner.run_image(im, args.conf, args.iou)
            for x1, y1, x2, y2, s, c in dets:
                cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(im, f'{int(c)} {s:.2f}', (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            if args.save:
                vw.write(im)
            else:
                cv2.imshow('det', im)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        cap.release()
        if vw is not None:
            vw.release()
        if not args.save and cv2 is not None:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

