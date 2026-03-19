#!/usr/bin/env bash
# Full pipeline: preprocess → train → inference → export viewer
set -e

echo "=============================="
echo " CBCT Tooth Segmentation Pipeline"
echo "=============================="

# 1. setup dirs
echo "[1/4] Setting up directories..."
mkdir -p data/processed/{images,labels,splits} \
         results/{graphs,models,videos,predictions} \
         viewer/exports

# 2. preprocess (run as script from preprocessing notebook logic)
echo "[2/4] Preprocessing..."
uv run python - <<'EOF'
import sys, json
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import yaml
from sklearn.model_selection import train_test_split

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

IMAGES_DIR = Path(cfg["paths"]["images_raw"])
LABELS_DIR = Path(cfg["paths"]["labels_raw"])
PROC_IMGS  = Path(cfg["paths"]["processed_imgs"])
PROC_LBLS  = Path(cfg["paths"]["processed_lbl"])
SPLITS_DIR = Path(cfg["paths"]["splits"])

for d in [PROC_IMGS, PROC_LBLS, SPLITS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

img_files = sorted(IMAGES_DIR.glob("*_0000.mha"))
lbl_files = sorted(LABELS_DIR.glob("*.mha"))
img_map = {p.name.replace("_0000.mha",""): p for p in img_files}
lbl_map = {p.name.replace(".mha",""):      p for p in lbl_files}
cases   = sorted(set(img_map) & set(lbl_map))
print(f"Found {len(cases)} cases")

CLIP_LO, CLIP_HI = cfg["data"]["intensity_clip"]
NORM_LO, NORM_HI = cfg["data"]["intensity_norm"]

for cid in cases:
    io = PROC_IMGS / f"{cid}.mha"
    lo = PROC_LBLS / f"{cid}.mha"
    if io.exists() and lo.exists():
        continue
    img = sitk.ReadImage(str(img_map[cid]))
    lbl = sitk.ReadImage(str(lbl_map[cid]))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    arr = np.clip(arr, CLIP_LO, CLIP_HI)
    arr = (arr - CLIP_LO) / (CLIP_HI - CLIP_LO) * (NORM_HI - NORM_LO) + NORM_LO
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(img)
    sitk.WriteImage(sitk.Cast(out, sitk.sitkFloat32), str(io), True)
    sitk.WriteImage(lbl, str(lo), True)

np.random.seed(42)
train_val, test = train_test_split(cases, test_size=0.15, random_state=42)
train, val      = train_test_split(train_val, test_size=0.15/0.85, random_state=42)
for split, ids in [("train",train),("val",val),("test",test)]:
    entries = [{"image": str(PROC_IMGS/f"{c}.mha"), "label": str(PROC_LBLS/f"{c}.mha"), "case_id":c} for c in ids]
    with open(SPLITS_DIR/f"{split}.json","w") as f:
        json.dump(entries, f, indent=2)
print("Preprocessing done.")
EOF

# 3. train
echo "[3/4] Training..."
PYTORCH_ALLOC_CONF=expandable_segments:True uv run python -m src.train

# 4. inference on first test case
echo "[4/4] Running inference demo..."
uv run python - <<'EOF'
import json, sys
from pathlib import Path
import numpy as np
import torch
import SimpleITK as sitk
import yaml
from monai.inferers import sliding_window_inference

sys.path.insert(0, ".")
from src.model import build_model
from src.utils import load_config, load_label_mapping

cfg = load_config("config.yaml")
splits_dir = Path(cfg["paths"]["splits"])
models_dir = Path(cfg["paths"]["models"])
pred_dir   = Path(cfg["paths"]["predictions"])
viewer_dir = Path(cfg["paths"]["viewer_exports"])
pred_dir.mkdir(parents=True, exist_ok=True)
viewer_dir.mkdir(parents=True, exist_ok=True)

_, idx_to_label, num_classes = load_label_mapping(splits_dir)
patch_size = cfg["training"]["patch_size"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = build_model(cfg["training"]["model"], num_classes, patch_size)
ckpt  = torch.load(models_dir/"best.pth", map_location=device)
model.load_state_dict(ckpt["model"])
model = model.to(device).eval()

with open(splits_dir/"test.json") as f:
    test_cases = json.load(f)

case    = test_cases[0]
img_sitk = sitk.ReadImage(case["image"])
arr      = sitk.GetArrayFromImage(img_sitk).astype(np.float32)
tensor   = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)

with torch.no_grad():
    logits = sliding_window_inference(tensor, tuple(patch_size), 1, model,
                                      overlap=0.5, sw_device=device, device="cpu")

pred = logits.squeeze(0).argmax(0).numpy().astype(np.uint8)
pred_fdi = np.zeros_like(pred)
for idx, raw in idx_to_label.items():
    pred_fdi[pred==idx] = raw

cid = case["case_id"]
out = sitk.GetImageFromArray(pred_fdi)
out.CopyInformation(img_sitk)
sitk.WriteImage(sitk.Cast(out, sitk.sitkUInt8), str(pred_dir/f"{cid}_pred.mha"), True)
sitk.WriteImage(img_sitk, str(viewer_dir/"scan.nii.gz"), True)
sitk.WriteImage(out,      str(viewer_dir/"mask.nii.gz"), True)
print(f"Inference done. Viewer exports → {viewer_dir}")
EOF

echo ""
echo "=============================="
echo " Pipeline complete!"
echo " Viewer: cd viewer && python -m http.server 8888"
echo "=============================="