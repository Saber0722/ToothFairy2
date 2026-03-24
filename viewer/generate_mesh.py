"""
Converts mask.nii.gz → mesh_data.json for the 3D HTML viewer.
Uses marching cubes to extract per-class surface meshes.
Run: uv run python scripts/generate_mesh.py
"""

import json
import gzip
import struct
import numpy as np
from pathlib import Path
from skimage.measure import marching_cubes
import SimpleITK as sitk
import yaml

ROOT = Path(__file__).parent.parent
with open(ROOT / "config.yaml") as f:
    cfg = yaml.safe_load(f)

VIEWER_DIR = ROOT / cfg["paths"]["viewer_exports"]
OUT_FILE   = VIEWER_DIR / "mesh_data.json"

FDI_NAMES = {
    0:"BG",1:"Upper jaw",2:"Lower jaw",3:"UR8",4:"UR7",5:"UR6",6:"UR5",7:"UR4",
    8:"UR3",9:"UR2",10:"UR1",11:"UL1",12:"UL2",13:"UL3",14:"UL4",15:"UL5",
    16:"UL6",17:"UL7",18:"UL8",21:"LL8",22:"LL7",23:"LL6",24:"LL5",25:"LL4",
    26:"LL3",27:"LL2",28:"LL1",31:"LR1",32:"LR2",33:"LR3",34:"LR4",35:"LR5",
    36:"LR6",37:"LR7",38:"LR8",41:"LL8b",42:"LL7b",43:"LL6b",44:"LL5b",
    45:"LL4b",46:"LL3b",47:"LL2b",48:"LL1b",
}

COLORS = {
    0:[0,0,0],1:[138,43,226],2:[30,144,255],3:[255,215,0],4:[255,165,0],
    5:[255,99,71],6:[255,69,0],7:[220,20,60],8:[255,20,147],9:[148,0,211],
    10:[75,0,130],11:[0,191,255],12:[0,255,127],13:[50,205,50],14:[173,255,47],
    15:[255,255,0],16:[255,215,0],17:[255,140,0],18:[255,69,0],21:[0,250,154],
    22:[0,255,255],23:[0,206,209],24:[95,158,160],25:[70,130,180],
    26:[100,149,237],27:[0,0,205],28:[0,0,139],31:[255,182,193],
    32:[255,105,180],33:[255,20,147],34:[199,21,133],35:[219,112,147],
    36:[255,160,122],37:[250,128,114],38:[233,150,122],41:[144,238,144],
    42:[0,255,0],43:[34,139,34],44:[0,128,0],45:[85,107,47],
    46:[107,142,35],47:[154,205,50],48:[124,252,0],
}

def smooth_mask(mask, iterations=1):
    """Simple binary erosion+dilation to reduce mesh noise."""
    from scipy.ndimage import binary_erosion, binary_dilation
    smoothed = binary_erosion(mask, iterations=iterations)
    smoothed = binary_dilation(smoothed, iterations=iterations)
    return smoothed

def main():
    print("Loading mask...")
    mask_sitk = sitk.ReadImage(str(VIEWER_DIR / "mask.nii.gz"))
    mask_arr  = sitk.GetArrayFromImage(mask_sitk)  # (Z, Y, X)
    spacing   = mask_sitk.GetSpacing()              # (sx, sy, sz)
    sp = [spacing[2], spacing[1], spacing[0]]       # → (sz, sy, sx) for Z,Y,X

    Z, Y, X = mask_arr.shape
    present  = [int(v) for v in np.unique(mask_arr) if v > 0]
    print(f"Mask shape: {mask_arr.shape} | Classes present: {present}")

    meshes = []
    for cls in present:
        print(f"  Meshing class {cls} ({FDI_NAMES.get(cls, cls)})...")
        binary = (mask_arr == cls).astype(np.uint8)

        # skip very small regions
        if binary.sum() < 50:
            print(f"    Skipping (too small: {binary.sum()} voxels)")
            continue

        # smooth to reduce jagged mesh
        binary_f = smooth_mask(binary, iterations=1).astype(np.float32)

        try:
            verts, faces, normals, _ = marching_cubes(
                binary_f,
                level=0.5,
                spacing=tuple(sp),  # apply voxel spacing
                allow_degenerate=False,
            )
        except Exception as e:
            print(f"    Marching cubes failed: {e}")
            continue

        # center the mesh around origin
        center = np.array([Z * sp[0] / 2, Y * sp[1] / 2, X * sp[2] / 2])
        verts  = verts - center

        # decimate — keep every Nth vertex reference to reduce JSON size
        # for jaw bones (large), downsample more aggressively
        max_faces = 8000 if cls <= 2 else 3000
        if len(faces) > max_faces:
            step = len(faces) // max_faces
            faces = faces[::step]

        c = COLORS.get(cls, [200, 200, 200])

        meshes.append({
            "cls":     cls,
            "name":    FDI_NAMES.get(cls, f"cls{cls}"),
            "color":   [c[0]/255, c[1]/255, c[2]/255],
            "verts":   verts.flatten().tolist(),   # flat [x,y,z, x,y,z, ...]
            "faces":   faces.flatten().tolist(),   # flat [a,b,c, a,b,c, ...]
            "n_verts": len(verts),
            "n_faces": len(faces),
        })
        print(f"    → {len(verts)} verts, {len(faces)} faces")

    out = {
        "dims":    list(mask_arr.shape),
        "spacing": sp,
        "classes": meshes,
    }

    with open(OUT_FILE, "w") as f:
        json.dump(out, f, separators=(',', ':'))  # compact JSON

    size_mb = OUT_FILE.stat().st_size / 1e6
    print(f"\nSaved {len(meshes)} meshes → {OUT_FILE} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    main()