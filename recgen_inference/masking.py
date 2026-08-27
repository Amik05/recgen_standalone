"""Text-prompted object masking via Grounding DINO + SAM 2."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

DEFAULT_DINO_MODEL = "IDEA-Research/grounding-dino-tiny"
DEFAULT_SAM_MODEL = "facebook/sam2.1-hiera-small"

FG_PCT_MIN = 0.5
FG_PCT_MAX = 70.0
BBOX_AREA_MIN = 0.01
BBOX_AREA_MAX = 0.90


class MaskValidationError(ValueError):
    """Raised when a generated mask fails quality checks."""


@dataclass(frozen=True)
class Detection:
    box: tuple[int, int, int, int]
    score: float
    label: str


def load_rgb(path: Path | str) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_depth(path: Path | str) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(path)
    return depth


def refine_mask_with_depth(mask: np.ndarray, depth: np.ndarray) -> np.ndarray:
    if depth.shape != mask.shape:
        raise ValueError(f"depth {depth.shape} and mask {mask.shape} must match")
    valid = depth > 0
    return np.where(valid, mask, 0).astype(np.uint8)


def segment_sam2(
    rgb: np.ndarray,
    *,
    points: list[tuple[int, int]] | None = None,
    labels: list[int] | None = None,
    box: tuple[int, int, int, int] | None = None,
    model_id: str = DEFAULT_SAM_MODEL,
) -> np.ndarray:
    import torch
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    points = points or []
    labels = labels or []

    predictor = SAM2ImagePredictor.from_pretrained(
        model_id, device="cuda" if torch.cuda.is_available() else "cpu"
    )
    predictor.set_image(rgb)

    point_coords = np.array(points, dtype=np.float32) if points else None
    point_labels = np.array(labels, dtype=np.int32) if points else None
    box_arr = np.array(box, dtype=np.float32)[None, :] if box is not None else None

    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=box_arr,
        multimask_output=True,
    )
    best = int(np.argmax(scores))
    return (masks[best] > 0).astype(np.uint8) * 255


def normalize_prompt(prompt: str) -> str:
    prompt = prompt.strip().lower()
    if not prompt.endswith("."):
        prompt += "."
    return prompt


@lru_cache(maxsize=1)
def _load_grounding_dino(model_id: str):
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.eval()
    return processor, model, device


def _clip_box(
    box: np.ndarray | tuple[float, ...],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return x1, y1, x2, y2


def detect_all(
    rgb: np.ndarray,
    prompt: str,
    *,
    model_id: str = DEFAULT_DINO_MODEL,
    box_threshold: float = 0.25,
    text_threshold: float = 0.25,
) -> list[Detection]:
    """Return all Grounding DINO detections for a prompt, sorted by score descending."""
    import torch

    processor, model, device = _load_grounding_dino(model_id)
    image = Image.fromarray(rgb)
    text = normalize_prompt(prompt)

    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    target_size = [(image.height, image.width)]
    if hasattr(processor, "post_process_grounded_object_detection"):
        import inspect

        post_kwargs: dict[str, object] = {
            "text_threshold": text_threshold,
            "target_sizes": target_size,
        }
        sig = inspect.signature(processor.post_process_grounded_object_detection)
        if "box_threshold" in sig.parameters:
            post_kwargs["box_threshold"] = box_threshold
        else:
            post_kwargs["threshold"] = box_threshold
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            **post_kwargs,
        )[0]
    else:
        results = processor.post_process_object_detection(
            outputs,
            threshold=box_threshold,
            target_sizes=target_size,
        )[0]

    boxes = results.get("boxes")
    scores = results.get("scores")
    labels = results.get("labels") or results.get("text_labels") or []

    if boxes is None or len(boxes) == 0:
        return []

    order = torch.argsort(scores, descending=True).tolist()
    detections: list[Detection] = []
    w, h = image.size
    for idx in order:
        box = _clip_box(boxes[idx].detach().cpu().numpy(), w, h)
        score = float(scores[idx].detach().cpu().item())
        label = str(labels[idx]) if idx < len(labels) else prompt
        detections.append(Detection(box=box, score=score, label=label))
    return detections


def detect_box(
    rgb: np.ndarray,
    prompt: str,
    *,
    model_id: str = DEFAULT_DINO_MODEL,
    box_threshold: float = 0.25,
    text_threshold: float = 0.25,
    pick: int = 0,
) -> Detection:
    """Return the selected Grounding DINO detection for a prompt."""
    detections = detect_all(
        rgb,
        prompt,
        model_id=model_id,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    if not detections:
        raise ValueError(
            f"No object found for prompt {prompt!r}. "
            f"Try a more specific phrase or lower box_threshold (now {box_threshold})."
        )
    return detections[min(pick, len(detections) - 1)]


def bbox_area_fraction(box: tuple[int, int, int, int], width: int, height: int) -> float:
    x1, y1, x2, y2 = box
    return ((x2 - x1) * (y2 - y1)) / max(width * height, 1)


def foreground_fraction(mask: np.ndarray) -> float:
    return 100.0 * float((mask > 0).mean())


def validate_mask(
    mask: np.ndarray,
    box: tuple[int, int, int, int],
    *,
    fg_pct_min: float = FG_PCT_MIN,
    fg_pct_max: float = FG_PCT_MAX,
    bbox_area_min: float = BBOX_AREA_MIN,
    bbox_area_max: float = BBOX_AREA_MAX,
) -> float:
    """Validate mask quality; return foreground percent or raise MaskValidationError."""
    h, w = mask.shape[:2]
    fg_pct = foreground_fraction(mask)
    bbox_frac = bbox_area_fraction(box, w, h)

    hints = (
        "Try a more specific prompt, lower --box-threshold (e.g. 0.15), "
        "use --pick N for another detection, or fall back to make_mask_sam.py --interactive-web."
    )

    if fg_pct < fg_pct_min:
        raise MaskValidationError(
            f"Mask foreground too small ({fg_pct:.2f}% < {fg_pct_min}%). "
            f"DINO box may have missed the object. {hints}"
        )
    if fg_pct > fg_pct_max:
        raise MaskValidationError(
            f"Mask foreground too large ({fg_pct:.2f}% > {fg_pct_max}%). "
            f"Detection box may be wrong or too broad. {hints}"
        )
    if bbox_frac > bbox_area_max:
        raise MaskValidationError(
            f"Detection box covers {bbox_frac * 100:.1f}% of the image (max {bbox_area_max * 100:.0f}%). "
            f"Likely a false positive. {hints}"
        )
    if bbox_frac < bbox_area_min:
        raise MaskValidationError(
            f"Detection box covers only {bbox_frac * 100:.2f}% of the image (min {bbox_area_min * 100:.0f}%). "
            f"Object may be too small or misdetected. {hints}"
        )
    return fg_pct


def draw_bbox_overlay(
    rgb: np.ndarray,
    box: tuple[int, int, int, int],
    *,
    label: str = "",
    score: float | None = None,
) -> np.ndarray:
    overlay = rgb.copy()
    x1, y1, x2, y2 = box
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 64, 0), 2)
    caption = label
    if score is not None:
        caption = f"{label} {score:.2f}" if label else f"{score:.2f}"
    if caption:
        cv2.putText(
            overlay,
            caption,
            (x1, max(y1 - 8, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 64, 0),
            1,
            cv2.LINE_AA,
        )
    return overlay


def draw_mask_preview(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = rgb.copy()
    overlay[mask > 0] = (overlay[mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
    return overlay


def mask_from_prompt(
    rgb: np.ndarray,
    prompt: str,
    *,  # tbh I had to google this, it just forces the rest to be keyword arguments
    depth: np.ndarray | None = None,
    refine_depth: bool = False,
    pick: int = 0,  # usually 0 for the best match, change if it grabs the wrong thing
    box_threshold: float = 0.25, # confidence threshold
    text_threshold: float = 0.25,
    dino_model: str = DEFAULT_DINO_MODEL,
    sam_model: str = DEFAULT_SAM_MODEL,
    validate: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Detect object by text prompt, segment with SAM 2, optionally refine with depth."""
    
    # gotta make sure we actually have a depth array if they want to refine it
    # otherwise it crashes down below
    if refine_depth and depth is None:
        raise ValueError("refine_depth=True requires a depth array")

    # Step 1: Find the object with Grounding DINO
    # this returns a detection object with the box coordinates
    detection = detect_box(
        rgb,
        prompt,
        model_id=dino_model,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        pick=pick,
    )

    # Step 2: Pass the box to SAM 2 to get the actual pixel mask
    # SAM takes the box and figures out the exact edges. It's kinda magic.
    mask = segment_sam2(rgb, box=detection.box, model_id=sam_model)

    # Step 3: Optional depth cleanup
    # if we have depth data, use it to cut out background noise
    if refine_depth and depth is not None:
        mask = refine_mask_with_depth(mask, depth)

    # Step 4: Check if the mask is actually good
    # calculate what percentage of the image is the object
    fg_pct = foreground_fraction(mask)
    
    if validate:
        # this might throw an error if the mask is weirdly big or tiny
        fg_pct = validate_mask(mask, detection.box)

    # pack up all the info we might need later into a dictionary
    metadata: dict[str, Any] = {
        "box": detection.box,
        "score": detection.score,  # how confident DINO was
        "label": detection.label,  
        "fg_pct": fg_pct,          
        "prompt": prompt,          # saving the original prompt just in case
    }
    
    # return the final image mask and the info dict together
    return mask, metadata
