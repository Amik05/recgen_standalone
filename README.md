 # Standalone RecGen module

Quick, focused instructions for the current capture → mask → reconstruct workflow.

Prerequisites
- Install dependencies with Pixi (recommended):

```bash
# from repo root
pixi install
```

Capture and mask
- Place your RGB and depth captures in `custom_examples/` as `<name>_rgb.png` and `<name>_depth.png`.
- Create a mask automatically from a text prompt or manually via SAM:

Automatic (text prompt):
```bash
./scripts/run_capture.sh <name> --prompt "white mug" --all
```

Manual (SAM interactive):
```bash
./scripts/run_capture.sh <name> --mask
```

Reconstruction (inference)
- Run single-view reconstruction using the CLI:

```bash
pixi run python scripts/run_inference.py \
  --rgb custom_examples/<name>_rgb.png \
  --depth custom_examples/<name>_depth.png \
  --mask custom_examples/<name>_mask.png \
  --intrinsics custom_examples/frame_intrinsics_approx.yaml \
  --name <name>
```

Outputs
- Default outputs are written to `outputs/inference_outputs/<name>/` (or the `--out` folder you set).
- Typical files: `overlay.png`, `mesh.obj`, `metadata.json`, optional Gaussian splats or `textured_mesh.glb` when CUDA extensions are available.

Examples
- Ready-to-run examples live in `examples/`. Run an example like this:

```bash
pixi run python scripts/run_inference.py \
  --rgb examples/ex0_rgb.png \
  --depth examples/ex0_depth.png \
  --mask examples/ex0_mask.png \
  --intrinsics examples/intrinsics.yaml \
  --name ex0
```

Notes
- `run_capture.sh` is a convenience wrapper that handles masking and inference flags (`--prompt`, `--mask`, `--all`, `--example`).
- Use `pixi shell` and `scripts/setup_cuda.sh` if you want optional CUDA extensions (`nvdiffrast`, flash-attn) for extra exports.

Help / Contributing
- For details about flags and advanced options, see `scripts/run_inference.py` and the scripts in `scripts/`.
- Open an issue or PR if something is missing or unclear.


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
