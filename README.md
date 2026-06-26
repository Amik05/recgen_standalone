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

---

## Running Inference

### CLI: `scripts/run_inference.py`

The main entry point for single-view reconstruction:

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
    --out ./out_color2 \
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
* **Tip:** The pipeline erodes the mask by default (5×5 kernel, 1 iteration) to trim noisy depth edges

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

See `examples/intrinsics.yaml` for a reference file. The example RGB/depth/mask images (`ex0_*.png`) referenced in the CLI are expected under `examples/` but may need to be supplied separately.

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
│   ├── run_inference.py      # CLI entry point
│   └── setup_cuda.sh         # Optional CUDA extension installer
├── examples/
│   └── intrinsics.yaml       # Sample camera intrinsics
├── outputs/
│   └── inference_outputs/    # Default CLI output root
├── recgen_inference/         # Python package (importable API)
└── pixi.toml                 # Environment definition
```

---

## Requirements

* Linux x86_64
* NVIDIA GPU with CUDA support (tested on Blackwell / `sm_120`; also supports CUDA 11.8 via `pixi install -e cu118`)
* ~8 GB+ VRAM recommended for inference
* Internet access on first run (HuggingFace model download)
