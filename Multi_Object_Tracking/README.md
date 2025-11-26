# Multi-Object Tracking Library

## Overview

- This library implements a real-time multi-object tracking pipeline for vertical slot fishways, aiming to produce stable trajectories and minimize identity switches. It follows the "detection → appearance encoding → motion prediction → cost matrix construction & association → track maintenance → output" paradigm.
- **Appearance Branch:** A lightweight ReID network (using RIVU/RIPL architecture) outputs L2-normalized embeddings.
- **Geometry Branch:** A constant velocity model (Kalman filter) is used with Mahalanobis distance gating.
- **Association:** The association cost is a combined metric of appearance and geometry. After gating, the Hungarian algorithm is used for optimal one-time matching, supplemented by cascaded matching and an IoU fallback.
- **Lifecycle Management:** Tracks are managed through confirmed, lost, and terminated states.

## Core Modules

### 1. Candidate Generation and Cropping

- **Input:** A set of detections D_t = {cx, cy, w, h, score, cls}.
- Each candidate bounding box is proportionally expanded (with α ∈ [0.05, 0.20]), cropped to the ReID input resolution, and then normalized (channel-wise and scale).

### 2. Appearance Re-identification (ReID) Network

- **Backbone:** RIVU units are stacked in Stage2/3/4/5, with channels and strides aligned with the embedding head to emphasize representation quality and gradient flow.
- **RIPL (Reparameterization-Inference-for-Plain-Layers):** RIPL multi-branch blocks (3x3, 1x1, and identity, all with BN) are introduced at the 3x3 convolution sites in Stage4/5. During training, all branches are active. For inference, BN fusion and kernel merging are performed to yield a single, efficient 3x3 convolutional layer.
- **Embedding Head:** A BN-Neck is followed by a fully-connected layer and L2 normalization to produce a unit-length embedding `e`. The online distance metric is d(e1, e2) = 1 - cos(e1, e2).
- **Training:** Batch-hard triplet loss is used (P×K sampling). For each anchor, the "hardest" positive and negative samples within the batch are selected to form a triplet. The loss minimizes d(ap) - d(an) + margin. A minimum time interval is maintained between samples from adjacent frames.

### 3. Motion Model and Geometric Gating

- **State:** The track state is defined as x = [cx, cy, a, h, cx_dot, cy_dot, a_dot, h_dot]. A Kalman filter predicts the next state `ẑ` and its covariance `S`.
- **Gating:** Mahalanobis distance, d_mah = (z - ẑ)^T S^(-1)(z - ẑ), is used to gate and prune unlikely pairings.

### 4. Appearance Gating and Combined Cost

- **Gating:** A cosine distance threshold is used for appearance gating.
- **Combined Cost:** `cost = λ * (1 - cos(e_track, e_det)) + (1 - λ) * min(d_mah, γ)`.
- The parameter `λ` ∈ [0, 1] balances the appearance/geometry trade-off, and `γ` truncates the Mahalanobis distance. The geometry weight is adaptively increased in crowded scenes.

### 5. Data Association and Lifecycle Management

- The Hungarian algorithm performs optimal matching on the gated cost matrix. Unmatched tracks enter cascaded matching and an IoU fallback to handle short-term occlusions and drift.
- New pairings are initially tentative and must be matched for ≥ `n_init` consecutive frames to become stable. Tracks that are unmatched for more than `max_age` frames are terminated.

### 6. Appearance Template and Memory

- Each track maintains an appearance queue `E_k` of length `N`. The appearance template `e_track` is updated using a sliding weighted average or the mean of the last K frames. Cosine distance is used consistently for both online and offline phases.

### 7. Output and Interface

- **Output per frame:** `[x1, y1, x2, y2, conf, cls, track_id]`.
- The output can be extended to include species, confidence, and appearance summaries.

## Installation and Dependencies

- Python 3.8+
- NumPy
- PyTorch (CPU/CUDA)
- `opencv-python` (optional, for visualization and saving)

## Quick Start

### Synthetic Demo

```bash
python -m Multi_Object_Tracking.demo --frames 120 --show --save track_demo.mp4
```

### From Video and Detection CSV

```bash
python -m Multi_Object_Tracking.run_tracker --video your.mp4 --dets dets.csv --save result.mp4 --show
```
- **CSV Format:** `frame,x1,y1,x2,y2,conf,cls`

### Code Integration

```python
import numpy as np
from .tracker.deepsort_mod import DeepSortMOD, DeepSortArgs

# Configure arguments
args = DeepSortArgs()
tracker = DeepSortMOD(args)

# Create a dummy image and detections
img = np.zeros((480, 640, 3), dtype=np.uint8)
dets = np.array([[100, 120, 180, 200, 0.85, 0]], dtype=float)

# Update tracker
tracks = tracker.update(dets, img)
```

## Parameters

### Association & Gating
- `max_cosine_distance`: 0.25
- `max_iou_distance`: 0.6
- `track_high_thresh`: 0.30
- `track_low_thresh`: 0.05
- `new_track_thresh`: 0.30

### Lifecycle
- `n_init`: 1
- `max_age`: 40
- `nn_budget`: 200

### Appearance Queue
- `feature_dim`: 256
- `feature_momentum`: 0.9

### Cost Weights
- `match_iou_weight`: 0.6
- `match_feat_weight`: 0.4
- `adaptive`: True

### Cropping
- `alpha` ∈ [0.05, 0.20]

## External Configuration

- Parameters can be overridden via an environment variable pointing to a YAML or JSON file:
    `export DEEPSORT_ARGS_PATH=/path/to/deepsort_args.yaml`
- Field names in the file must match the attributes of `DeepSortArgs`. Fields not specified in the file will retain their default values.