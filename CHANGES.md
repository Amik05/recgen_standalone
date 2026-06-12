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

You must use the `pixi` package manager. Because we are pulling pre-release PyTorch wheels, you must set specific environment variables before installing.

### 1. Install the Environment
```bash
# Allow UV to pull PyTorch Nightly pre-releases
export UV_PRERELEASE=allow
export UV_LOCK_TIMEOUT=600

# Force the solver to check PyPI for SciPy instead of giving up at the PyTorch index
export UV_INDEX_STRATEGY=unsafe-best-match

# Build the lockfile and download the CUDA 12.8 / PyTorch 2.6 stack
pixi install