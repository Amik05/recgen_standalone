#!/usr/bin/env bash
# Run the custom_examples capture pipeline without typing long commands.
#
# Naming convention (files in custom_examples/):
#   <name>_rgb.png, <name>_depth.png, <name>_mask.png
#
# Usage (from repo root):
#   ./scripts/run_capture.sh mug70 --prompt "white mug" --all   # text mask + infer
#   ./scripts/run_capture.sh mug70 --mask                      # web UI mask only
#   ./scripts/run_capture.sh mug70                               # infer only
#   ./scripts/run_capture.sh ex0 --example                     # shipped example
#
# Over SSH (web mask): ssh -L 8765:localhost:8765 user@host  then open http://localhost:8765

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

NAME="${1:?Usage: $0 <name> [--prompt TEXT | --mask | --all] [--example]}"
shift || true

MODE="infer"
USE_EXAMPLE=false
PORT=8765
SPLAT="--save-splat"
PROMPT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mask) MODE="mask" ;;
    --all) MODE="all" ;;
    --example) USE_EXAMPLE=true ;;
    --prompt) PROMPT="$2"; shift ;;
    --port) PORT="$2"; shift ;;
    --no-splat) SPLAT="" ;;
    -h|--help)
      sed -n '2,14p' "$0"
      echo "  --prompt TEXT   Semantic mask via Grounding DINO + SAM (use with --mask or --all)"
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
  shift
done

if [[ -n "$PROMPT" && "$MODE" == "infer" ]]; then
  MODE="all"
fi

run() { pixi run python "$@"; }

if $USE_EXAMPLE; then
  RGB="examples/${NAME}_rgb.png"
  DEPTH="examples/${NAME}_depth.png"
  MASK="examples/${NAME}_mask.png"
  INTR="examples/intrinsics.yaml"
  OUT="outputs/inference_outputs/${NAME}"
  OUT_FLAG=(--name "$NAME")
else
  RGB="custom_examples/${NAME}_rgb.png"
  DEPTH="custom_examples/${NAME}_depth.png"
  MASK="custom_examples/${NAME}_mask.png"
  INTR="custom_examples/frame_intrinsics_approx.yaml"
  OUT="out_${NAME}"
  OUT_FLAG=(--out "$OUT")
fi

do_mask_web() {
  echo "==> SAM mask (web UI): $RGB"
  echo "    Open http://localhost:${PORT} (port-forward if on SSH)"
  run scripts/make_mask_sam.py \
    --rgb "$RGB" \
    --interactive-web --port "$PORT" \
    --output "$MASK" \
    --preview "${MASK%.png}_preview.png"
}

do_mask_text() {
  echo "==> Semantic mask: $RGB  prompt=\"$PROMPT\""
  MASK_ARGS=(
    scripts/mask_from_text.py
    --rgb "$RGB"
    --prompt "$PROMPT"
    --output "$MASK"
    --preview "${MASK%.png}_preview.png"
  )
  if [[ -f "$DEPTH" ]]; then
    MASK_ARGS+=(--depth "$DEPTH" --refine-depth)
  else
    echo "    (no depth file at $DEPTH — skipping --refine-depth)"
  fi
  run "${MASK_ARGS[@]}"
}

do_mask() {
  if [[ -n "$PROMPT" ]]; then
    do_mask_text
  else
    do_mask_web
  fi
}

do_infer() {
  [[ -f "$MASK" ]] || { echo "Missing mask: $MASK (run with --mask, --all, or --prompt)" >&2; exit 1; }
  echo "==> RecGen: $NAME -> $OUT"
  run scripts/run_inference.py \
    --rgb "$RGB" --depth "$DEPTH" --mask "$MASK" \
    --intrinsics "$INTR" \
    "${OUT_FLAG[@]}" $SPLAT
  echo "==> Done. Check ${OUT}/overlay.png"
}

case "$MODE" in
  mask) do_mask ;;
  infer) do_infer ;;
  all) do_mask; do_infer ;;
esac
