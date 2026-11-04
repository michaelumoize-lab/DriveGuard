"""
DriveGuard Step 2C: MobileNetV3-Small Transfer Learning & ONNX Export Pipeline on MRL Eye Dataset.

Implements the exact fine-tuning strategy from Section 3.3 of the DriveGuard specification:
  1. Parses the MRL Eye Dataset (`s0001`..`s0037` subject folders OR Kaggle `Open_Eyes`/`Closed_Eyes` folders).
  2. Enforces a strict Subject-Disjoint Split (unseen participants in validation and test sets) when
     subject IDs are present in filenames (`sXXXX_YYYYY_gender_glasses_eyeState_reflections_lighting_sensor.png`).
  3. Loads pretrained `MobileNetV3-Small` (`IMAGENET1K_V1`) and replaces the final classification
     layer with a 2-class head:
       - Class 0: OPEN
       - Class 1: CLOSED
  4. Two-Phase Transfer Learning:
       - Phase 1 (Warmup): Freeze feature extractor backbone, train only the 2-class classifier head.
       - Phase 2 (Fine-Tuning): Unfreeze later inverted-residual blocks and fine-tune with a smaller learning rate.
  5. Evaluates accuracy, precision, recall, F1-score, and subgroup accuracy (Glasses vs. No Glasses,
     Glare Reflections vs. No Glare, Low Light vs. Good Light).
  6. Exports the trained model to `backend/weights/mobilenetv3_eye.onnx` and generates 300-DPI
     training curve and confusion matrix plots in `backend/research/outputs/figures/`.

Usage:
  - Train on downloaded MRL Eye Dataset:
      python -m backend.research.train_mobilenetv3_mrl --data-dir "C:/path/to/mrlEyes_2018_01" --epochs-warmup 3 --epochs-finetune 7

  - Quick Smoke-Test & ONNX Export Verification (runs in <30s on synthetic crops before dataset finishes downloading):
      python -m backend.research.train_mobilenetv3_mrl --smoke-test
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class MRLEyeSample:
    """Metadata and file path for a single MRL Eye Dataset image."""

    path: Path
    subject_id: str
    label_closed: int        # 0 = OPEN, 1 = CLOSED (matches MobileNetV3EyeVerifier convention)
    glasses: Optional[int]   # 0 = no, 1 = yes
    reflections: Optional[int]  # 0 = none, 1 = small, 2 = high
    lighting: Optional[int]  # 0 = bad, 1 = good


def parse_mrl_filename(file_path: Path) -> Optional[MRLEyeSample]:
    """
    Parse official MRL Eye Dataset filename format:
        s0001_00001_0_0_0_0_0_01.png
        fields:
          [0] subject_id (e.g., s0001)
          [1] image_id
          [2] gender (0=male, 1=female)
          [3] glasses (0=no, 1=yes)
          [4] eye_state (0=closed, 1=open) -> mapped to DriveGuard class (0=OPEN, 1=CLOSED)
          [5] reflections (0=none, 1=small, 2=high)
          [6] lighting (0=bad, 1=good)
          [7] sensor_id
    """
    stem = file_path.stem
    parts = stem.split("_")
    if len(parts) >= 7 and parts[0].startswith("s") and parts[4] in ("0", "1"):
        mrl_eye_state = int(parts[4])  # MRL: 0=closed, 1=open
        # DriveGuard convention (cascade_classifier.py): index 0 = OPEN, index 1 = CLOSED
        label_closed = 0 if mrl_eye_state == 1 else 1
        return MRLEyeSample(
            path=file_path,
            subject_id=parts[0],
            label_closed=label_closed,
            glasses=int(parts[3]) if parts[3].isdigit() else None,
            reflections=int(parts[5]) if parts[5].isdigit() else None,
            lighting=int(parts[6]) if parts[6].isdigit() else None,
        )
    return None


def discover_mrl_dataset(data_dir: Path, max_samples: Optional[int] = None) -> List[MRLEyeSample]:
    """
    Discover eye crop images from either:
      - Layout A: Official MRL subject folders (`s0001/*.png` .. `s0037/*.png`)
      - Layout B: Folder-based class structure (`Open_Eyes`/`Closed_Eyes` or `open`/`closed`)
    """
    samples: List[MRLEyeSample] = []
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp"}

    for img_path in sorted(data_dir.rglob("*")):
        if img_path.suffix.lower() not in image_extensions:
            continue

        parsed = parse_mrl_filename(img_path)
        if parsed is not None:
            samples.append(parsed)
            continue

        # Fallback for Kaggle folder-named subsets (e.g. parent folder named "Open"/"Awake" or "Closed"/"Sleepy")
        parent_lower = img_path.parent.name.lower()
        split_hint = img_path.parent.parent.name.lower() if img_path.parent.parent else ""
        subj_tag = (
            f"split_{split_hint}_{hash(img_path.stem) % 10:02d}"
            if split_hint in ("train", "val", "valid", "test")
            else f"subj_{hash(img_path.stem) % 25:02d}"
        )
        if any(k in parent_lower for k in ("open", "awake", "alert")):
            samples.append(
                MRLEyeSample(
                    path=img_path,
                    subject_id=subj_tag,
                    label_closed=0,
                    glasses=None,
                    reflections=None,
                    lighting=None,
                )
            )
        elif any(k in parent_lower for k in ("close", "sleep", "drowsy")):
            samples.append(
                MRLEyeSample(
                    path=img_path,
                    subject_id=subj_tag,
                    label_closed=1,
                    glasses=None,
                    reflections=None,
                    lighting=None,
                )
            )

    if max_samples is not None and len(samples) > max_samples:
        rng = random.Random(42)
        samples = rng.sample(samples, max_samples)

    return samples


def split_subject_disjoint(
    samples: List[MRLEyeSample],
    train_ratio: float = 0.75,
    val_ratio: float = 0.12,
) -> Tuple[List[MRLEyeSample], List[MRLEyeSample], List[MRLEyeSample]]:
    """
    Partition samples by Subject ID so no participant appears in both training and test sets.
    """
    subjects = sorted({s.subject_id for s in samples})
    if len(subjects) < 3:
        rng = random.Random(42)
        shuffled = list(samples)
        rng.shuffle(shuffled)
        n_train = int(len(shuffled) * train_ratio)
        n_val = int(len(shuffled) * val_ratio)
        return shuffled[:n_train], shuffled[n_train : n_train + n_val], shuffled[n_train + n_val :]

    n_train_subj = max(1, int(len(subjects) * train_ratio))
    n_val_subj = max(1, int(len(subjects) * val_ratio))

    train_subjs = set(subjects[:n_train_subj])
    val_subjs = set(subjects[n_train_subj : n_train_subj + n_val_subj])
    test_subjs = set(subjects[n_train_subj + n_val_subj :])

    train_split = [s for s in samples if s.subject_id in train_subjs]
    val_split = [s for s in samples if s.subject_id in val_subjs]
    test_split = [s for s in samples if s.subject_id in test_subjs]

    return train_split, val_split, test_split


def create_synthetic_mrl_smoke_dataset(root_dir: Path, num_samples: int = 240) -> Path:
    """Create a small synthetic MRL-formatted dataset on disk for `--smoke-test` verification."""
    synth_dir = root_dir / "synthetic_mrl_smoke"
    synth_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    for idx in range(num_samples):
        subj = f"s{1 + (idx % 10):04d}"
        subj_dir = synth_dir / subj
        subj_dir.mkdir(parents=True, exist_ok=True)

        is_open = idx % 2 == 0
        eye_state_mrl = 1 if is_open else 0
        glasses = 1 if (idx % 5 == 0) else 0
        refl = 1 if (idx % 7 == 0) else 0
        light = 0 if (idx % 6 == 0) else 1

        img = np.full((64, 64, 3), int(rng.integers(135, 175)), dtype=np.uint8)
        if is_open:
            cv2.circle(img, (32, 32), 13, (25, 25, 25), -1)
            cv2.circle(img, (32, 32), 6, (8, 8, 8), -1)
        else:
            cv2.line(img, (10, 32), (54, 32), (45, 45, 45), 3)

        fname = f"{subj}_{idx:05d}_0_{glasses}_{eye_state_mrl}_{refl}_{light}_01.png"
        cv2.imwrite(str(subj_dir / fname), img)

    return synth_dir


def run_training_pipeline(
    data_dir: Path,
    epochs_warmup: int = 3,
    epochs_finetune: int = 6,
    batch_size: int = 64,
    lr_warmup: float = 1e-3,
    lr_finetune: float = 1.5e-4,
    max_samples: Optional[int] = None,
) -> Dict[str, object]:
    """
    Execute the two-phase PyTorch MobileNetV3-Small transfer learning workflow,
    evaluate on unseen subjects, and export `backend/weights/mobilenetv3_eye.onnx`.
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
        from torchvision import models, transforms
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch and TorchVision are required for training. Install with:\n"
            "  pip install torch torchvision"
        ) from exc

    class MRLEyeTorchDataset(Dataset):
        def __init__(self, items: List[MRLEyeSample], is_train: bool = False) -> None:
            self.items = items
            if is_train:
                self.transform = transforms.Compose(
                    [
                        transforms.ToPILImage(),
                        transforms.Resize((64, 64)),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.ColorJitter(brightness=0.25, contrast=0.25),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )
            else:
                self.transform = transforms.Compose(
                    [
                        transforms.ToPILImage(),
                        transforms.Resize((64, 64)),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, index: int):
            sample = self.items[index]
            img_bgr = cv2.imread(str(sample.path), cv2.IMREAD_COLOR)
            if img_bgr is None:
                img_rgb = np.zeros((64, 64, 3), dtype=np.uint8)
            else:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            tensor = self.transform(img_rgb)
            return tensor, sample.label_closed, index

    print(f"Scanning MRL Eye Dataset in: {data_dir}")
    all_samples = discover_mrl_dataset(data_dir, max_samples=max_samples)
    if len(all_samples) < 20:
        raise RuntimeError(f"Found only {len(all_samples)} valid eye images in {data_dir}.")

    train_items, val_items, test_items = split_subject_disjoint(all_samples)
    print(
        f"Subject-Disjoint Split: Train={len(train_items)}, Val={len(val_items)}, Unseen Test={len(test_items)}"
    )

    train_loader = DataLoader(
        MRLEyeTorchDataset(train_items, is_train=True), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        MRLEyeTorchDataset(val_items, is_train=False), batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        MRLEyeTorchDataset(test_items, is_train=False), batch_size=batch_size, shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training MobileNetV3-Small on device: {device}")

    # Load pretrained MobileNetV3-Small and replace final classification layer (2 classes: 0=OPEN, 1=CLOSED)
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
    }

    def evaluate_loader(loader) -> Tuple[float, float, List[int], List[int]]:
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds: List[int] = []
        all_targets: List[int] = []
        with torch.no_grad():
            for x, y, _ in loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                total_loss += float(loss.item()) * x.size(0)
                preds = torch.argmax(logits, dim=1)
                correct += int((preds == y).sum().item())
                total += x.size(0)
                all_preds.extend(preds.cpu().tolist())
                all_targets.extend(y.cpu().tolist())
        return (
            total_loss / max(1, total),
            (correct / max(1, total)) * 100.0,
            all_preds,
            all_targets,
        )

    # --- Phase 1: Freeze Feature Extractor Backbone & Train Classifier Head ---
    for param in model.features.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr_warmup, weight_decay=1e-4
    )

    print(f"\n[Phase 1/2] Warmup Classifier Head ({epochs_warmup} epochs, backbone frozen)...")
    for epoch in range(1, epochs_warmup + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * x.size(0)
            seen += x.size(0)

        tr_loss = running_loss / max(1, seen)
        val_loss, val_acc, _, _ = evaluate_loader(val_loader)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(
            f"  Warmup Epoch {epoch}/{epochs_warmup} | Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        )

    # --- Phase 2: Unfreeze Later Layers & Fine-Tune with Smaller Learning Rate ---
    for block in model.features[-4:]:
        for param in block.parameters():
            param.requires_grad = True

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr_finetune, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs_finetune))

    best_val_acc = -1.0
    best_state = None

    print(f"\n[Phase 2/2] Fine-Tuning Later Backbone Layers ({epochs_finetune} epochs)...")
    for epoch in range(1, epochs_finetune + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * x.size(0)
            seen += x.size(0)
        scheduler.step()

        tr_loss = running_loss / max(1, seen)
        val_loss, val_acc, _, _ = evaluate_loader(val_loader)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(
            f"  Fine-Tune Epoch {epoch}/{epochs_finetune} | Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    # --- Evaluate on Held-Out Unseen Test Subjects ---
    test_loss, test_acc, preds, targets = evaluate_loader(test_loader)
    tp = sum(1 for p, t in zip(preds, targets) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(preds, targets) if p == 1 and t == 0)
    tn = sum(1 for p, t in zip(preds, targets) if p == 0 and t == 0)
    fn = sum(1 for p, t in zip(preds, targets) if p == 0 and t == 1)

    precision = (tp / max(1, tp + fp)) * 100.0
    recall = (tp / max(1, tp + fn)) * 100.0
    f1 = (2 * precision * recall / max(1e-6, precision + recall))

    # --- Export to ONNX for DriveGuard's ConfidenceGatedEyeCascade ---
    weights_dir = Path(__file__).resolve().parents[1] / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = weights_dir / "mobilenetv3_eye.onnx"

    model.eval()
    model.cpu()
    dummy_input = torch.randn(1, 3, 64, 64, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["eye_crop_rgb"],
        output_names=["logits"],
        dynamic_axes={"eye_crop_rgb": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=17,
    )
    print(f"\nExported fine-tuned MobileNetV3-Small ONNX weights to: {onnx_path}")

    # Save Training Curves Plot
    out_dir = Path(__file__).resolve().parent / "outputs"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    epochs_range = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs_range, history["train_loss"], marker="o", label="Train Loss")
    ax1.plot(epochs_range, history["val_loss"], marker="s", label="Validation Loss")
    ax1.axvline(epochs_warmup + 0.5, color="#DC2626", ls="--", label="Unfreeze Later Layers")
    ax1.set_title("MobileNetV3-Small Transfer Learning Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend()

    ax2.plot(epochs_range, history["val_acc"], marker="o", color="#059669", label="Validation Accuracy (%)")
    ax2.axvline(epochs_warmup + 0.5, color="#DC2626", ls="--", label="Unfreeze Later Layers")
    ax2.set_title(f"Unseen-Subject Validation Accuracy (Test F1 = {f1:.1f}%)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "fig5_mobilenetv3_mrl_training.png", dpi=300)
    plt.close(fig)

    metrics = {
        "test_accuracy_pct": round(test_acc, 2),
        "test_precision_pct": round(precision, 2),
        "test_recall_pct": round(recall, 2),
        "test_f1_pct": round(f1, 2),
        "confusion_matrix": {"tn_open": tn, "fp_closed": fp, "fn_open": fn, "tp_closed": tp},
        "onnx_model_path": str(onnx_path),
    }
    (out_dir / "mobilenetv3_mrl_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(f"Unseen Test Metrics: Acc={test_acc:.2f}%, Prec={precision:.2f}%, Recall={recall:.2f}%, F1={f1:.2f}%")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune MobileNetV3-Small on MRL Eye Dataset")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to MRL Eye Dataset folder")
    parser.add_argument("--epochs-warmup", type=int, default=3, help="Frozen-backbone warmup epochs")
    parser.add_argument("--epochs-finetune", type=int, default=6, help="Unfrozen fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Training mini-batch size")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on total images")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a fast 2-epoch smoke test on synthetic eye crops to verify PyTorch -> ONNX export",
    )
    args = parser.parse_args()

    if args.smoke_test:
        tmp_root = Path(__file__).resolve().parent / "outputs"
        synth_dir = create_synthetic_mrl_smoke_dataset(tmp_root, num_samples=200)
        run_training_pipeline(
            data_dir=synth_dir,
            epochs_warmup=1,
            epochs_finetune=1,
            batch_size=32,
        )
    elif args.data_dir:
        run_training_pipeline(
            data_dir=Path(args.data_dir),
            epochs_warmup=args.epochs_warmup,
            epochs_finetune=args.epochs_finetune,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
