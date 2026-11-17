# DriveGuard Research: MobileNetV3-Small Eye State Classifier & Benchmarks

This module contains the research artifacts, dataset ingest, training pipeline, and benchmarking utilities for Stage 2 of DriveGuard's **Confidence-Gated Eye Cascade**.

---

## 1. MRL Eye Dataset Requirements & Specifications

The Stage 2 secondary eye-state verifier is trained on the **MRL Eye Dataset** (Large Scale Dataset for Eye Blink Detection, 2018):
- **Total Images:** 84,898 grayscale/IR eye-crop images.
- **Participants:** 37 distinct human subjects (`s0001` through `s0037`).
- **Classes:** Binary (OPEN = 0, CLOSED = 1).
- **Attribute Metadata:** 8-field standardized naming convention:
  `sXXXX_YYYYY_gender_glasses_eyeState_reflections_lighting_sensor.png`
  - `subject_id` (`s0001` - `s0037`)
  - `image_id`
  - `gender` (0 = male, 1 = female)
  - `glasses` (0 = no, 1 = yes)
  - `eye_state` (0 = closed, 1 = open) — mapped in DriveGuard to `0 = OPEN`, `1 = CLOSED`
  - `reflections` (0 = none, 1 = small, 2 = high)
  - `lighting` (0 = bad, 1 = good)
  - `sensor_id` (01 = infrared / camera sensor)

---

## 2. Directory Structure

Place the unpacked dataset into a directory accessible locally:

```text
mrlEyes_2018_01/
├── s0001/
│   ├── s0001_00001_0_0_0_0_0_01.png
│   └── ...
├── s0002/
│   └── ...
...
└── s0037/
    └── ...
```

*Alternative (Kaggle class-folder layout):*
```text
mrl_kaggle/
├── Open_Eyes/
└── Closed_Eyes/
```

---

## 3. Methodology & Integrity Guarantees

### Strict Subject-Disjoint Partitioning
To prevent data leakage and inflated accuracy claims:
1. **Zero Subject Overlap:** No participant (`sXXXX`) appears in more than one partition (Train, Validation, or Test).
2. **Deterministic Partitioning:** Random state is strictly seeded (`seed=42`) ensuring exact reproducibility.
3. **Partition Verification:** Programmatic assertions verify empty intersections:
   $$\text{Subjects}_{\text{train}} \cap \text{Subjects}_{\text{val}} = \emptyset, \quad \text{Subjects}_{\text{train}} \cap \text{Subjects}_{\text{test}} = \emptyset, \quad \text{Subjects}_{\text{val}} \cap \text{Subjects}_{\text{test}} = \emptyset$$

### Honest ROC-AUC Metric
If an evaluation partition contains only a single class (or zero samples):
- ROC-AUC is mathematically undefined.
- `compute_roc_auc` returns `None` (logged as `N/A`), **never** a fabricated score of `1.0`.

### Pretrained Backbone Source
- **Model:** MobileNetV3-Small (`IMAGENET1K_V1` weights from `torchvision.models.mobilenet_v3_small`).
- **Input Dimensions:** $3 \times 64 \times 64$, normalized with standard ImageNet mean and variance.
- **Output:** 2-class logits (Open probability, Closed probability).

---

## 4. Commands for Reproduction

### A. Quick Smoke-Test & ONNX Export (< 30 seconds)
Runs on 240 synthetic samples with known ground truth to verify the complete training loop, validation, metrics calculation, and ONNX export without downloading the full 85k-image dataset:

```bash
python -m backend.research.train_mobilenetv3_mrl --smoke-test
```

### B. Full Model Training on MRL Eye Dataset
```bash
python -m backend.research.train_mobilenetv3_mrl \
  --data-dir "C:/path/to/mrlEyes_2018_01" \
  --epochs-warmup 3 \
  --epochs-finetune 7 \
  --batch-size 64 \
  --seed 42
```

Outputs:
- Exported ONNX model: `backend/weights/mobilenetv3_eye.onnx`
- PyTorch state dict: `backend/weights/mobilenetv3_eye.pth`
- High-resolution training curves & confusion matrix: `backend/research/outputs/figures/`
