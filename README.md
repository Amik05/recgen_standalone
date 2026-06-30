# RecGen: Blackwell Edition (RTX 4500 PRO / sm_120)

This is a customized fork of the [TRI-ML RecGen](https://github.com/TRI-ML/recgen) 3D reconstruction pipeline.

The original codebase was hard-coded to PyTorch 2.4 / CUDA 12.1 and relied heavily on FP32 `xformers` attention fallbacks. That specific configuration instantly crashes on NVIDIA Blackwell architecture GPUs (compute capability `sm_120`) with a "no kernel image available" error.

This fork modernizes the dependency stack to PyTorch 2.6 Nightly and patches the custom 3D sparse transformer modules to natively route attention operations through PyTorch's built-in Scaled Dot Product Attention (SDPA), completely bypassing the `xformers` hardware bottleneck.

## Key Modifications

### 1. Modernized Dependency Stack (`pixi.toml`)
* **Upgraded to PyTorch Nightly:** Shifted the deep learning backbone to pull `torch` and `torchvision` directly from PyPI's `cu128` nightly index to guarantee Blackwell (`sm_120`) tensor compatibility.
* **Compiler Synchronization:** Downgraded the internal Pixi `nvcc` toolkit from 13.x to exactly `12.8.*` to perfectly match the PyTorch Nightly wheel. This prevents PyTorch's C++ extension builder from aborting during the compilation of `nvdiffrast` and `diff_gaussian_rasterization`.
* **SciPy Pinning:** Pinned `scipy>=1.11,<1.15` to prevent the `uv` solver from pulling Python 3.10 incompatible versions from the PyTorch registries.
* **Evicted xformers:** Removed `xformers` from the environment completely to force the pipeline to fall back to native math engines.

### 2. Native SDPA Attention Patches
The TRI-ML developers hard-coded their custom 3D sparse transformer modules to strictly accept `"xformers"` or `"flash_attn"`. Since `xformers` Cutlass FP32 fallbacks do not support Hopper/Blackwell, we injected native PyTorch SDPA handlers.

* **`serialized_attn.py`:** Added an import bypass for `ATTN == 'sdpa'` to prevent `ValueError` crashes, allowing downstream native tensor routing.
* **`windowed_attn.py`:** Added the `'sdpa'` import bypass, imported `math`, and explicitly wrote the memory layout transpositions for both fixed-batched and variable-length sequence `torch.nn.functional.scaled_dot_product_attention()` calculations.

---

## Installation & Setup

You must use the [pixi](https://pixi.sh) package manager. Because we are pulling pre-release PyTorch wheels, set these environment variables before installing:

```bash
# Allow UV to pull PyTorch Nightly pre-releases
export UV_PRERELEASE=allow
export UV_LOCK_TIMEOUT=600

# Force the solver to check PyPI for SciPy instead of giving up at the PyTorch index
export UV_INDEX_STRATEGY=unsafe-best-match

# Build the lockfile and download the CUDA 12.8 / PyTorch 2.6 stack
pixi install
```

### Optional CUDA extensions

These are not required for basic mesh inference, but unlock extra outputs:

| Task | Command | Enables |
|------|---------|---------|
| Flash Attention (faster) | `pixi run post-install` | Faster attention; falls back to SDPA if install fails |
| nvdiffrast | `pixi run build-nvdiffrast` | `textured_mesh.glb`, turntable rendering |
| Gaussian rasterizer | `pixi run build-gaussian-rasterizer` | `textured_mesh.glb`, `turntable.mp4` |

Alternatively, use the shell helper directly:

```bash
pixi shell
bash scripts/setup_cuda.sh              # spconv + flash-attn (default)
bash scripts/setup_cuda.sh --nvdiffrast # build nvdiffrast from source
bash scripts/setup_cuda.sh --all        # everything
```

Model weights are downloaded automatically from HuggingFace (`TRI-ML/RecGen`) on first run.

### SAM 2 + Grounding DINO (masks for your own captures)

RecGen does not segment objects — you need a mask PNG.

**Semantic mask (recommended — text prompt, no clicks):**

```bash
pixi run python scripts/mask_from_text.py \
    --rgb custom_examples/mug70_rgb.png \
    --prompt "white mug" \
    --output custom_examples/mug70_mask.png \
    --preview custom_examples/mug70_mask_preview.png
```

Or one command for mask + reconstruct:

```bash
./scripts/run_capture.sh mug70 --prompt "white mug" --all
```

**Manual mask (fallback)** — SAM 2 with browser clicks (`make_mask_sam.py --interactive-web`).

`run_capture.sh` passes `--refine-depth` automatically when `<name>_depth.png` exists.

Dependencies are installed via `pixi install` in the **default** environment (`sam2`, `transformers`, `timm`). The CUDA 11.8 variant (`pixi install -e cu118`) does not include SAM 2 — use the default env for text masking.

#### Prompt tips (text masking)

- Use **specific, visual** phrases: `"white ceramic mug"` rather than `"mug"`.
- If nothing is detected, lower `--box-threshold` (e.g. `0.15`) or try a shorter noun phrase.
- When several objects match, list candidates with `--list-detections`, then pick one with `--pick N` (`0` = highest score).
- Check `<name>_mask_bbox.png` to verify Grounding DINO found the right object before running RecGen.
- If text masking fails, fall back to `./scripts/run_capture.sh <name> --mask` (browser clicks).

**Python API:**

```python
from recgen_inference import mask_from_prompt
import cv2

rgb = cv2.cvtColor(cv2.imread("custom_examples/mug70_rgb.png"), cv2.COLOR_BGR2RGB)
mask, meta = mask_from_prompt(rgb, "white mug")
# meta: box, score, label, fg_pct, prompt
```

---

## Pipeline Overview

```
RGB + Depth + Intrinsics
        │
        ▼
  mask_from_text.py    ← text prompt → Grounding DINO → SAM 2  (autonomous)
  make_mask_sam.py     ← click/box prompts                     (manual fallback)
        │
        ▼
     mask.png
        │
        ▼
  run_inference.py     ← RecGen (mesh + pose + overlay)
        │
        ▼
  mesh.obj, overlay.png, metadata.json, …
```

| Step | Tool | When |
|------|------|------|
| Mask (semantic) | `scripts/mask_from_text.py` or `--prompt` on `run_capture.sh` | Your captures — text prompt |
| Mask (manual) | `scripts/make_mask_sam.py` | Fallback when text fails |
| Reconstruct | `scripts/run_inference.py` / `run_capture.sh` | Always |
| Intrinsics | `scripts/dump_orbbec_intrinsics.py` | Optional — Orbbec Femto Bolt |

Shipped examples under `examples/` already include masks — skip the mask step for those.

### Short commands (recommended)

Put captures in `custom_examples/` as `<name>_rgb.png`, `<name>_depth.png`, and (after masking) `<name>_mask.png`. Then:

```bash
cd recgen

# 1) New capture — semantic mask from text
./scripts/run_capture.sh mug70 --prompt "white mug" --all

# Or manual mask in browser (SSH: ssh -L 8765:localhost:8765 … first)
./scripts/run_capture.sh mug70 --mask

# 2) Reconstruct (mask must already exist)
./scripts/run_capture.sh mug70

# Or both in one go after you have RGB+depth saved:
./scripts/run_capture.sh mug70 --all

# Shipped example
./scripts/run_capture.sh ex0 --example
```

You only re-run **step 1** when you need a new or better mask. Step 2 is the one you repeat if you tweak intrinsics or want another export.

---

## Quick Start: Shipped Examples

The repo includes six ready-to-run examples under `examples/` (`ex0`–`ex5`). Each has a matching RGB image, depth map, and object mask. All examples share `examples/intrinsics.yaml`.

From the repository root (after `pixi install`):

```bash
# Run example 0 (default output: outputs/inference_outputs/ex0/)
pixi run python scripts/run_inference.py \
    --rgb examples/ex0_rgb.png \
    --depth examples/ex0_depth.png \
    --mask examples/ex0_mask.png \
    --intrinsics examples/intrinsics.yaml \
    --name ex0
```

Check `outputs/inference_outputs/ex0/overlay.png` to verify the reconstruction lines up with the input image.

**Run any other shipped example** — change the file prefix and `--name`:

```bash
# ex1 … ex5
pixi run python scripts/run_inference.py \
    --rgb examples/ex1_rgb.png \
    --depth examples/ex1_depth.png \
    --mask examples/ex1_mask.png \
    --intrinsics examples/intrinsics.yaml \
    --name ex1
```

**With Gaussian splat export:**

```bash
pixi run python scripts/run_inference.py \
    --rgb examples/ex0_rgb.png \
    --depth examples/ex0_depth.png \
    --mask examples/ex0_mask.png \
    --intrinsics examples/intrinsics.yaml \
    --name ex0 \
    --save-splat
```

**Write to a custom output folder:**

```bash
pixi run python scripts/run_inference.py \
    --rgb examples/ex0_rgb.png \
    --depth examples/ex0_depth.png \
    --mask examples/ex0_mask.png \
    --intrinsics examples/intrinsics.yaml \
    --out ./my_output
```

| Example | RGB | Depth | Mask |
|---------|-----|-------|------|
| ex0 | `examples/ex0_rgb.png` | `examples/ex0_depth.png` | `examples/ex0_mask.png` |
| ex1 | `examples/ex1_rgb.png` | `examples/ex1_depth.png` | `examples/ex1_mask.png` |
| ex2 | `examples/ex2_rgb.png` | `examples/ex2_depth.png` | `examples/ex2_mask.png` |
| ex3 | `examples/ex3_rgb.png` | `examples/ex3_depth.png` | `examples/ex3_mask.png` |
| ex4 | `examples/ex4_rgb.png` | `examples/ex4_depth.png` | `examples/ex4_mask.png` |
| ex5 | `examples/ex5_rgb.png` | `examples/ex5_depth.png` | `examples/ex5_mask.png` |

Preview thumbnails for each example are in `examples/thumbnails/`.

---

## Your Own Captures (`custom_examples/`)

Sample Orbbec captures live in `custom_examples/` (`color2`, `color3`, `frame_0145`, etc.). Unlike shipped examples, **you must create a mask** before running inference.

### 1. Create a mask with SAM

**Remote SSH (no display)** — use the browser UI with port forwarding:

```bash
# On your laptop (separate terminal):
ssh -L 8765:localhost:8765 user@remote-host

# On the remote machine (repo root):
pixi run python scripts/make_mask_sam.py \
    --rgb custom_examples/color3.png \
    --interactive-web \
    --port 8765 \
    --output custom_examples/mask3.png \
    --preview custom_examples/mask3_preview.png
```

Open **http://localhost:8765** on your laptop. **Left-click** = object, **right-click** = background, then **Generate mask**.

**Local machine with a display:**

```bash
pixi run python scripts/make_mask_sam.py \
    --rgb custom_examples/color3.png \
    --interactive \
    --output custom_examples/mask3.png \
    --preview custom_examples/mask3_preview.png
```

**CLI prompts (no GUI)** — box and/or pixel coordinates:

```bash
pixi run python scripts/make_mask_sam.py \
    --rgb custom_examples/color3.png \
    --box 400 200 900 600 \
    --point 640 420 \
    --output custom_examples/mask3.png
```

Avoid `--refine-depth` unless depth on the object is reliable (it can destroy masks when depth is sparse).

### 2. Run RecGen

```bash
pixi run python scripts/run_inference.py \
    --rgb custom_examples/color3.png \
    --depth custom_examples/depth3.png \
    --mask custom_examples/mask3.png \
    --intrinsics custom_examples/frame_intrinsics_approx.yaml \
    --out ./out_color3 \
    --save-splat
```

### 3. Orbbec camera intrinsics

For accurate pose and overlay alignment, dump real intrinsics from a connected Femto Bolt:

```bash
pixi run python scripts/dump_orbbec_intrinsics.py \
    --width 1280 --height 720 \
    --output custom_examples/frame_intrinsics.yaml
```

Requires `pyorbbecsdk`. `custom_examples/frame_intrinsics_approx.yaml` is a rough 1280×720 guess — fine for mesh shape, less accurate for pose.

---

## Running Inference

### CLI: `scripts/run_inference.py`

The main entry point for single-view reconstruction on your own captures:

```bash
pixi run python scripts/run_inference.py \
    --rgb   path/to/rgb.png \
    --depth path/to/depth.png \
    --mask  path/to/mask.png \
    --intrinsics path/to/intrinsics.yaml \
    --name my_run
```

**All flags:**

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--rgb` | yes | — | RGB image (PNG/JPG), same resolution as depth/mask |
| `--depth` | yes | — | Depth map (see [Input formats](#input-formats) below) |
| `--mask` | yes | — | Object mask; non-zero pixels = object |
| `--intrinsics` | yes | — | YAML file with camera intrinsics (see below) |
| `--out` | no | `outputs/inference_outputs/<name>/` | Output directory |
| `--name` | no | `run` | Subfolder name when `--out` is not set |
| `--checkpoint` | no | `recgen_base.multiview_stereo` | Model checkpoint name |
| `--seed` | no | `42` | Random seed |
| `--save-splat` | no | off | Also write Gaussian splat `.ply` files |
| `--save-glb` | no | off | Also write textured `textured_mesh.glb` (needs nvdiffrast) |

**Example with custom output directory and all optional exports:**

```bash
pixi run python scripts/run_inference.py \
    --rgb examples/ex0_rgb.png \
    --depth examples/ex0_depth.png \
    --mask examples/ex0_mask.png \
    --intrinsics examples/intrinsics.yaml \
    --out ./my_output \
    --save-splat \
    --save-glb
```

### Python API

```python
import cv2
import numpy as np
import yaml
from recgen_inference import build_recgen, generate

# Load inputs
rgb   = cv2.cvtColor(cv2.imread("rgb.png"), cv2.COLOR_BGR2RGB)
depth = cv2.imread("depth.png", cv2.IMREAD_UNCHANGED)
mask  = cv2.imread("mask.png", cv2.IMREAD_UNCHANGED)
if mask.ndim == 3:
    mask = mask[:, :, 0]

with open("intrinsics.yaml") as f:
    d = yaml.safe_load(f)
K = np.array([[d["fu"], 0, d["pu"]],
              [0, d["fv"], d["pv"]],
              [0, 0, 1]], dtype=np.float64)

# Run
pipeline = build_recgen.build("recgen_base.multiview_stereo")
result = generate(pipeline, image=rgb, depth=depth, mask=mask, intrinsics=K, seed=42)
result.save("./out", save_splat=True, save_glb=False)
```

---

## Input Formats

Place your input files anywhere on disk and pass their paths to `--rgb`, `--depth`, `--mask`, and `--intrinsics`. All four images must share the same width and height.

### RGB (`--rgb`)

* **Format:** PNG or JPG
* **Channels:** 3-channel color (loaded as RGB)
* **Content:** A single camera view of the object you want to reconstruct

### Depth (`--depth`)

* **Format:** PNG (recommended) or any format OpenCV can read
* **Units (auto-detected):**
  * `uint16` — treated as **millimetres** (typical Kinect / RealSense export)
  * `float32` with max ≤ 30 — treated as **metres**
  * `float32` with max > 30 — treated as **millimetres**, divided by 1000
* **Content:** Per-pixel depth; pixels outside the object mask are zeroed before processing

### Mask (`--mask`)

* **Format:** PNG (grayscale or single channel)
* **Values:** Any non-zero pixel marks the object; zero = background
* **How to create:** Use `scripts/mask_from_text.py` (text prompt, recommended) or `scripts/make_mask_sam.py` (SAM 2 clicks). Shipped `examples/ex*_mask.png` files are pre-made.
* **Tip:** The pipeline erodes the mask by default (5×5 kernel, 1 iteration) to trim noisy depth edges

#### `mask_from_text.py` flags

| Flag | Description |
|------|-------------|
| `--rgb` | Input RGB image (required) |
| `--prompt` | Object description, e.g. `"white mug"` (required) |
| `--output` | Output mask PNG |
| `--preview` | Green overlay preview PNG |
| `--bbox-preview` | DINO detection box drawn on RGB (default: `<mask>_bbox.png`) |
| `--depth` | Depth map path (used with `--refine-depth`) |
| `--refine-depth` | Zero mask pixels where depth == 0 |
| `--box-threshold` | Grounding DINO box threshold (default: `0.25`) |
| `--text-threshold` | Grounding DINO text threshold (default: `0.25`) |
| `--pick` | Which detection when several match (`0` = highest score) |
| `--list-detections` | Print all DINO hits and exit (no mask written) |
| `--skip-validation` | Skip foreground / bbox quality checks |
| `--dino-model` | Hugging Face Grounding DINO model id |
| `--sam-model` | Hugging Face SAM 2.1 model id |

#### `make_mask_sam.py` flags

| Flag | Description |
|------|-------------|
| `--rgb` | Input RGB image (required) |
| `--interactive` | Click prompts in a matplotlib window (needs local display) |
| `--interactive-web` | Click prompts in a browser (SSH-friendly; use with `ssh -L`) |
| `--port` | Port for `--interactive-web` (default: `8765`) |
| `--point X Y` | Foreground pixel prompt (repeatable) |
| `--background-point X Y` | Background pixel prompt (repeatable) |
| `--box X1 Y1 X2 Y2` | Bounding box around the object |
| `--refine-depth` | Zero mask pixels where depth == 0 (use with caution) |
| `--output` | Output mask PNG |
| `--preview` | Optional green overlay preview PNG |

### Intrinsics (`--intrinsics`)

A YAML file with four scalar fields that define the pinhole camera matrix:

```yaml
fu: 1062.203      # focal length x (pixels)
fv: 1060.9691     # focal length y (pixels)
pu: 971.3832      # principal point x (pixels)
pv: 540.0661      # principal point y (pixels)
```

These map to the 3×3 intrinsic matrix:

```
K = [[fu,  0, pu],
     [ 0, fv, pv],
     [ 0,  0,  1]]
```

See `examples/intrinsics.yaml` for a reference file. Shipped sample images (`ex0_*.png`–`ex5_*.png`) are included under `examples/` — see [Quick Start: Shipped Examples](#quick-start-shipped-examples).

---

## Output Files

By default, results are written to:

```
outputs/inference_outputs/<name>/
```

or to the path given by `--out`. A typical run produces:

| File | Always? | Description |
|------|---------|-------------|
| `mesh.obj` | yes | Reconstructed mesh in the model's internal object-centric frame (vertex colors) |
| `posed_mesh.obj` | yes | Same mesh transformed into the **input camera frame** |
| `overlay.png` | yes | Input RGB with the posed mesh projected on top |
| `metadata.json` | yes | Pose matrix, quaternion, intrinsics, and normalisation transform |
| `gaussian.ply` | with `--save-splat` | Gaussian splat in object-centric frame (aligned with `mesh.obj`) |
| `posed_gaussian.ply` | with `--save-splat` | Gaussian splat in camera frame (aligned with `posed_mesh.obj`) |
| `turntable.mp4` | if CUDA renderers installed | Side-by-side turntable: gaussian color \| mesh normals |
| `textured_mesh.glb` | with `--save-glb` | Textured GLB export (requires nvdiffrast + rasterizer) |

### Coordinate frames

* **`mesh.obj` / `gaussian.ply`** — model-internal Z-up object frame, before camera alignment.
* **`posed_mesh.obj` / `posed_gaussian.ply`** — aligned to your input camera using the estimated pose. Use these when compositing back into the original RGB image.
* **`metadata.json`** — contains `pose_matrix`, `pose_quat`, `cam2ncam` (the unit-cube normalisation applied before inference), and the input `intrinsics`.

---

## Project Layout

```
recgen/
├── scripts/
│   ├── run_inference.py          # RecGen CLI entry point
│   ├── run_capture.sh            # ./run_capture.sh mug70 --prompt "mug" --all
│   ├── mask_from_text.py         # Grounding DINO + SAM (text prompt)
│   ├── make_mask_sam.py          # SAM 2 mask (click / web / box)
│   ├── dump_orbbec_intrinsics.py # Orbbec Femto Bolt intrinsics → YAML
│   └── setup_cuda.sh             # Optional CUDA extension installer
├── examples/
│   ├── ex0_rgb.png … ex5_*.png   # Shipped RGB-D + mask examples
│   ├── intrinsics.yaml           # Camera intrinsics for shipped examples
│   └── thumbnails/
├── custom_examples/              # Team RGB-D captures (masks not included)
├── outputs/
│   └── inference_outputs/        # Default CLI output root
├── recgen_inference/             # Python package (importable API)
└── pixi.toml                     # Environment definition
```

---

## Requirements

* Linux x86_64
* NVIDIA GPU with CUDA support (tested on Blackwell / `sm_120`; CUDA 11.8 via `pixi install -e cu118` for RecGen inference only — text masking needs the default env)
* ~8 GB+ VRAM recommended for inference
* Internet access on first run (HuggingFace model download)
