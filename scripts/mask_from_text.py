#!/usr/bin/env python3
"""Create a RecGen mask from a text prompt using Grounding DINO + SAM 2.

RecGen expects a single-channel mask PNG: 0 = background, 255 = object.

Examples:
    python scripts/mask_from_text.py \\
        --rgb custom_examples/mug70_rgb.png \\
        --prompt "white mug" \\
        --output custom_examples/mug70_mask.png \\
        --preview custom_examples/mug70_mask_preview.png

    python scripts/mask_from_text.py \\
        --rgb custom_examples/color3_rgb.png \\
        --prompt "cardboard box" \\
        --box-threshold 0.3 \\
        --list-detections
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

from recgen_inference.masking import (
    MaskValidationError,
    detect_all,
    draw_bbox_overlay,
    draw_mask_preview,
    load_depth,
    load_rgb,
    mask_from_prompt,
)


def _default_output(rgb_path: Path) -> Path:
    stem = rgb_path.stem
    if stem.endswith("_rgb"):
        stem = stem[: -len("_rgb")]
    return rgb_path.with_name(f"{stem}_mask.png")


def _default_bbox_preview(output: Path) -> Path:
    return output.with_name(output.stem + "_bbox.png")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Text-prompted mask via Grounding DINO + SAM 2")
    p.add_argument("--rgb", required=True, type=Path, help="RGB image path")
    p.add_argument("--prompt", required=True, help='Object description, e.g. "white mug"')
    p.add_argument("--output", type=Path, help="Output mask PNG")
    p.add_argument("--preview", type=Path, help="Optional green overlay preview PNG")
    p.add_argument("--bbox-preview", type=Path, help="DINO bbox overlay PNG (default: <mask>_bbox.png)")
    p.add_argument("--depth", type=Path, help="Optional depth map (for --refine-depth)")
    p.add_argument("--refine-depth", action="store_true", help="Zero mask pixels where depth == 0")
    p.add_argument("--box-threshold", type=float, default=0.25, help="Grounding DINO box threshold")
    p.add_argument("--text-threshold", type=float, default=0.25, help="Grounding DINO text threshold")
    p.add_argument(
        "--pick",
        type=int,
        default=0,
        help="Which detection to use if several match (0 = highest score)",
    )
    p.add_argument(
        "--list-detections",
        action="store_true",
        help="Print all DINO detections and exit (no mask written)",
    )
    p.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip foreground/bbox quality checks",
    )
    p.add_argument(
        "--dino-model",
        default="IDEA-Research/grounding-dino-tiny",
        help="Hugging Face Grounding DINO model id",
    )
    p.add_argument(
        "--sam-model",
        default="facebook/sam2.1-hiera-small",
        help="Hugging Face SAM 2.1 model id",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rgb = load_rgb(args.rgb)

    if args.list_detections:
        detections = detect_all(
            rgb,
            args.prompt,
            model_id=args.dino_model,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
        )
        if not detections:
            print(
                f"No detections for prompt {args.prompt!r}. "
                f"Try a more specific phrase or lower --box-threshold (now {args.box_threshold})."
            )
            sys.exit(1)
        print(f"[mask_from_text] {len(detections)} detection(s) for {args.prompt!r}:")
        for i, det in enumerate(detections):
            print(f"  [{i}] {det.label!r} score={det.score:.3f} box={det.box}")
        return

    depth = load_depth(args.depth) if args.depth is not None else None
    if args.refine_depth and depth is None:
        raise SystemExit("--refine-depth requires --depth")

    try:
        mask, meta = mask_from_prompt(
            rgb,
            args.prompt,
            depth=depth,
            refine_depth=args.refine_depth,
            pick=args.pick,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            dino_model=args.dino_model,
            sam_model=args.sam_model,
            validate=not args.skip_validation,
        )
    except MaskValidationError as exc:
        raise SystemExit(str(exc)) from exc
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    print(
        f"[mask_from_text] DINO: {meta['label']!r} score={meta['score']:.3f} box={meta['box']}"
    )

    out = args.output or _default_output(args.rgb)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(out)
    print(f"Saved mask: {out}  ({mask.shape[1]}x{mask.shape[0]}, {meta['fg_pct']:.2f}% foreground)")

    bbox_preview = draw_bbox_overlay(
        rgb, meta["box"], label=meta["label"], score=meta["score"]
    )
    bbox_path = args.bbox_preview or _default_bbox_preview(out)
    bbox_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(bbox_preview).save(bbox_path)
    print(f"Saved bbox preview: {bbox_path}")

    if args.preview:
        Image.fromarray(draw_mask_preview(rgb, mask)).save(args.preview)
        print(f"Saved preview: {args.preview}")


if __name__ == "__main__":
    main()
