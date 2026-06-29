#!/usr/bin/env bash
# Run the custom_examples capture pipeline without typing long commands.
#
# Naming convention (files in custom_examples/):
#   <name>_rgb.png, <name>_depth.png, <name>_mask.png
#
# Usage (from repo root):
#   ./scripts/run_capture.sh mug70 --mask     # web UI mask only
#   ./scripts/run_capture.sh mug70            # infer only (mask must exist)
#   ./scripts/run_capture.sh mug70 --all      # mask UI, then infer
#   ./scripts/run_capture.sh ex0 --example    # shipped example (no custom_examples prefix)
#
# Over SSH: ssh -L 8765:localhost:8765 user@host  then open http://localhost:8765

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

NAME="${1:?Usage: $0 <name> [--mask | --all | --example]}"
shift || true

MODE="infer"
USE_EXAMPLE=false
PORT=8765
SPLAT="--save-splat"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mask) MODE="mask" ;;
    --all) MODE="all" ;;
    --example) USE_EXAMPLE=true ;;
    --port) PORT="$2"; shift ;;
    --no-splat) SPLAT="" ;;
    -h|--help)
      sed -n '2,14p' "$0"
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
  shift
done

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

do_mask() {
  echo "==> SAM mask: $RGB"
  echo "    Open http://localhost:${PORT} (port-forward if on SSH)"
  run scripts/make_mask_sam.py \
    --rgb "$RGB" \
    --interactive-web --port "$PORT" \
    --output "$MASK" \
    --preview "${MASK%.png}_preview.png"
}

do_infer() {
  [[ -f "$MASK" ]] || { echo "Missing mask: $MASK (run with --mask or --all first)" >&2; exit 1; }
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
