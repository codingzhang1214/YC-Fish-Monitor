# Object Detection Library

## Overview

- A lightweight object detection library tailored for vertical slot fishway videos, implementing a single-channel data flow: "Backbone → Multi-scale Fusion → Detection Head → Post-processing". It is optimized for both small objects and inference efficiency.
- **Backbone:** Utilizes RIVU (CSP + Ghost + SE) units.
- **Training-Inference Decoupling:** Achieved through RIPL (3x3/1x1/Identity parallel branches during training, merged into a single 3x3 for inference).
- **Multi-scale Fusion:** Employs SCALiF (FPN + PAN + AIFI + Adaptive Weighting) to output three feature maps {F3, F4, F5}.
- **Detection Head:** A decoupled one-stage head predicts classes and box parameters in parallel across the three scales.
- **Post-processing:** Includes confidence thresholding and Non-Maximum Suppression (NMS).

## Modules

- `backbones/rivu.py`: Implementation of RIVU and RIPL (CSP splitting, Ghost redundant features, SE channel calibration, and mergeable three-branch convolution for inference).
- `neck/scalif.py`: Implementation of SCALiF (FPN/PAN paths, AIFI self-attention, GAP+MLP for generating adaptive fusion weights).
- `head/det_head.py`: Decoupled detection head (classification/regression branches).
- `model/detector.py`: Integrates the backbone, neck, and detection head, providing a `predict` interface (with thresholding and NMS).
- `utils/ops.py`: Basic operations like NMS, coordinate transformations, etc.
- `infer.py`: Inference entry point, supporting single images or videos.

## Quick Start

### Image:
```bash
python -m Object_detection.infer --image your.jpg --save result.jpg --conf 0.25 --iou 0.5
```

### Video:
```bash
python -m Object_detection.infer --video your.mp4 --save result.mp4 --conf 0.25 --iou 0.5
```

### Code Integration:
```python
import numpy as np
from Object_detection import DetectorRunner

runner = DetectorRunner(num_classes=7, device='cpu')
img = np.zeros((480, 640, 3), dtype=np.uint8)
dets = runner.run_image(img, conf=0.25, iou=0.5)
# dets: [x1, y1, x2, y2, score, cls]
```

## Key Designs

### 1. Candidate and Cropping (Low-level Texture Retention)
- The backbone is designed to preserve shallow-level edge details, which is crucial for small-sized, low-contrast targets. This is particularly useful for the ReID branch in the tracking stage.

### 2. Backbone: RIVU
- Follows the principles described in the corresponding chapter. It combines two CSP branches (identity/lightweight transformation) with Ghost-generated proxy channels. The result is calibrated with SE and then linearly adjusted for the output.

### 3. Training-Inference: RIPL
- **Training:** 3x3, 1x1, and identity branches are used in parallel.
- **Inference:** Batch-Norm absorption and kernel merging are applied to fuse them into a single 3x3 convolution.

### 4. Multi-scale Fusion: SCALiF
- FPN provides a top-down path, PAN provides a bottom-up path, and AIFI introduces global context. A GAP+MLP module generates weights for adaptive feature fusion.

### 5. Detection Head and Post-processing
- Classification and regression are performed in parallel on three scales. Post-processing filters results using confidence and IoU thresholds, followed by NMS.

## Parameters and Interface

- **Default channel alignment:** 256
- **AIFI:** `d_model=256`, `nheads=8`
- **FPN:** Upsampling factor of 2
- **PAN:** Downsampling with a strided convolution (stride=2)
- **`predict(img, conf_thresh, iou_thresh)`:** Returns `[x1, y1, x2, y2, score, cls]` in pixel coordinates.