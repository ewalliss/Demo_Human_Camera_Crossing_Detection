# Pedestrian Re-ID Training Pipeline — User Guide

## Overview

This pipeline trains a **CLIP-based pedestrian Re-Identification** model using the **ViT-B/16 + SIE + OLP** variant. It follows a two-stage training strategy:

| Stage | What trains | Loss |
|-------|-------------|------|
| Stage 1 | `PromptLearner` only (text prompts) | SupCon (i2t + t2i) |
| Stage 2 | Image encoder + SIE + OLP + BN + Classifier | 0.25·ID + 1.0·Triplet + 1.0·I2T |
| LoRA | LoRA adapters on `q_proj`/`v_proj` | Same as Stage 2 |

---

## Prerequisites

### Python & Environment

```bash
# Create virtual environment (Python 3.13 recommended)
python3.13 -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate            # Windows

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate peft tqdm pytest
```

### Dataset

Market-1501 must be located at:
```
/Users/dangnguyen/Downloads/pedestrian/Market-1501-v15.09.15/
├── bounding_box_train/      ← training images
├── query/                   ← query images
└── bounding_box_test/       ← gallery images
```

---

## Quick Start — Mini Pipeline (1000 images)

Use this first to verify the full pipeline works end-to-end before committing to full training.

### Step 1: Stage 1 — Train PromptLearner

```bash
python3.13 src/train/train_stage1.py \
  --mini \
  --mini-num-ids 100 \
  --output-dir output/mini
```

Expected output:
```
[Stage 1] device=cuda  fp16=True
[Stage 1] train_pids=100  train_cams=6  batches=25
[Stage 1] trainable=8,192  total=151,277,572
  Epoch   1/3  batch    0/25  loss=4.1234
...
[Stage 1] epoch=3  avg_loss=1.2345  lr=3.50e-04
[Stage 1] saved → output/mini/stage1_epoch003.pth
```

### Step 2: Stage 2 — Fine-tune Image Encoder

```bash
python3.13 src/train/train_stage2.py \
  --mini \
  --stage1-checkpoint output/mini/stage1_epoch003.pth \
  --output-dir output/mini
```

Expected output:
```
[Stage 2] loaded Stage 1 from output/mini/stage1_epoch003.pth
[Stage 2] trainable=85,917,184
[Stage 2] precomputed text_feats: torch.Size([100, 512])
[Stage 2] epoch=1  loss=2.3456  lr=3.50e-05
[Stage 2] epoch=1  mAP=12.34%  R1=21.00%
[Stage 2] ✓ new best mAP=12.34% → output/mini/stage2_best.pth

── Model Health Check ──────────────────────────────
  [PASS] prompt_template contains 'person'
  [PASS] prompt_template has no 'vehicle'
  [PASS] No NaN/Inf in fused_feat
  [PASS] Feature norms in [0.1, 2.0]
  [PASS] SIE cam_embedding weights non-zero
  [PASS] OLP selected > 0 patches  (k=16)
  [PASS] cls_score std > 1e-4 (not collapsed)
────────────────────────────────────────────────────
  Overall: ALL PASS ✓
```

### Step 3: Smoke Test (automated)

```bash
python3.13 tests/integration/test_mini_pipeline.py
```

---

## Full Training

### Stage 1 (60 epochs)

```bash
python3.13 src/train/train_stage1.py \
  --market1501-root /Users/dangnguyen/Downloads/pedestrian/Market-1501-v15.09.15 \
  --output-dir output/full
```

### Stage 2 (120 epochs)

```bash
python3.13 src/train/train_stage2.py \
  --stage1-checkpoint output/full/stage1_epoch060.pth \
  --market1501-root /Users/dangnguyen/Downloads/pedestrian/Market-1501-v15.09.15 \
  --output-dir output/full
```

Expected results on Market-1501:
| Metric | Target |
|--------|--------|
| mAP | ~85% |
| Rank-1 | ~93% |
| Rank-5 | ~97% |

### LoRA Fine-Tuning (optional, 30 epochs)

After Stage 2, apply LoRA for additional gains with minimal parameter overhead:

```bash
python3.13 src/finetune/lora_finetune.py \
  --stage2-checkpoint output/full/stage2_best.pth \
  --output-dir output/lora
```

LoRA config: `r=16, alpha=16, dropout=0.1, target=[q_proj, v_proj]`

---

## Evaluation

### Evaluate a Checkpoint

```bash
python3.13 src/eval/model_health_check.py \
  --checkpoint output/full/stage2_best.pth \
  --num-pids 751 \
  --num-cams 6
```

### Run Full Evaluation with Re-ranking

```python
from src.config.defaults import PedestrianReIDConfig
from src.datasets.pedestrian_dataset import build_pedestrian_loaders
from src.eval.evaluate import evaluate
import torch

cfg = PedestrianReIDConfig()
device = torch.device("cuda")

_, query_loader, gallery_loader, _, _ = build_pedestrian_loaders(
    market1501_root=cfg.market1501_root,
    batch_size=256, num_instances=4, num_workers=2,
)

# load model ...
metrics = evaluate(model, query_loader, gallery_loader, device, use_rerank=True)
print(f"mAP={metrics['mAP']:.2f}%  R1={metrics['rank1']:.2f}%")
```

---

## Key CLI Flags

### train_stage1.py

| Flag | Default | Description |
|------|---------|-------------|
| `--market1501-root` | config default | Path to Market-1501 dataset |
| `--output-dir` | `output/pedestrian` | Checkpoint save directory |
| `--mini` | False | Use 1000-image mini set |
| `--mini-num-ids` | 100 | Identities in mini mode |
| `--num-workers` | 2 | DataLoader workers (keep ≤2 on Windows) |
| `--no-fp16` | False | Disable FP16 (auto-disabled on CPU) |

### train_stage2.py

| Flag | Required | Description |
|------|----------|-------------|
| `--stage1-checkpoint` | **yes** | Path to Stage 1 `.pth` checkpoint |
| `--market1501-root` | no | Dataset path override |
| `--output-dir` | no | Save directory override |
| `--mini` | no | Mini mode |
| `--no-fp16` | no | Disable FP16 |

### lora_finetune.py

| Flag | Required | Description |
|------|----------|-------------|
| `--stage2-checkpoint` | **yes** | Path to Stage 2 best checkpoint |
| `--market1501-root` | no | Dataset path override |
| `--output-dir` | no | Save directory override |
| `--mini` | no | Mini mode |

---

## Tests

```bash
# Unit tests (fast, no GPU needed)
python3.13 -m pytest tests/unit/ -v

# Integration smoke test (needs Market-1501)
python3.13 -m pytest tests/integration/ -v

# All tests
python3.13 -m pytest tests/ -v
```

---

## File Structure

```
src/
├── config/
│   └── defaults.py          ← PedestrianReIDConfig (all hyperparameters)
├── datasets/
│   ├── market1501.py        ← Market-1501 parser with mini mode
│   ├── pedestrian_dataset.py← PedestrianDataset + build_pedestrian_loaders()
│   ├── samplers.py          ← RandomIdentitySampler (P×K batches)
│   └── transforms.py        ← 224×224 augmentations + RandomErasing
├── models/
│   ├── clip_reid_pedestrian.py ← CLIPReIDPedestrianModel (main model)
│   ├── prompt_learner.py    ← Learnable text context tokens
│   ├── sie_layer.py         ← Camera + view ID embeddings
│   └── olp_head.py          ← Top-k patch pooling + fusion
├── losses/
│   ├── id_loss.py           ← CrossEntropy + label smoothing
│   ├── sup_con_loss.py      ← Supervised contrastive loss
│   └── triplet_loss.py      ← Batch-hard triplet loss
├── train/
│   ├── train_stage1.py      ← Stage 1: PromptLearner only
│   └── train_stage2.py      ← Stage 2: Image encoder + SIE + OLP
├── finetune/
│   └── lora_finetune.py     ← PEFT LoRA fine-tuning
└── eval/
    ├── evaluate.py          ← mAP + CMC@1/5/10
    ├── reranking.py         ← k-reciprocal re-ranking
    └── model_health_check.py← 8 PASS/FAIL checks

configs/person/
├── vit_clipreid_mini.yml    ← Mini config (100 IDs, 3+2 epochs)
└── vit_clipreid_pedestrian.yml ← Full config (751 IDs, 60+120 epochs)

tests/
├── unit/
│   ├── test_sie_layer.py
│   ├── test_olp_head.py
│   └── test_reranking.py
└── integration/
    └── test_mini_pipeline.py← Full end-to-end smoke test
```

---

## Architecture Notes

**PromptLearner (Stage 1)**
- Template: `"a photo of a X X X X person"` (must contain "person", must NOT contain "vehicle")
- 4 shared learnable tokens + per-identity class tokens
- Frozen CLIP text encoder produces identity-specific text features

**SIELayer**
- Additive camera-ID embedding `nn.Embedding(num_cams, 512)`
- Additive view-ID embedding `nn.Embedding(num_views, 512)`
- Injected into image features before OLP

**OLPHead**
- Selects top-k ViT patch tokens by L2 norm (occlusion-robust)
- Max-pools selected patches → local feature
- Fuses with global [CLS] via `Linear(1024→512)` + L2 norm

**Re-ranking (inference)**
- k-reciprocal encoding: k1=20, k2=6, λ=0.3
- FP16 cosine distance matrix, FP32 Jaccard distance
- Adds ~2-3% mAP over cosine baseline

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `RuntimeError: CUDA out of memory` | Reduce `batch_size` to 32 or use `--no-fp16` |
| `num_workers` crash on Windows | Keep `--num-workers 2` or use `0` |
| `KeyError: 'fused_feat'` | Ensure model is in `eval()` mode for health checks |
| `AssertionError: prompt contains vehicle` | Check `prompt_template` in config |
| Stage 2 mAP stays at 0% | Verify Stage 1 checkpoint path is correct |
| `peft` not found for LoRA | `pip install peft` |
