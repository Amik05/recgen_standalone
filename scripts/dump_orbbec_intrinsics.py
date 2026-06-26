#!/usr/bin/env python3
"""Read Orbbec Femto Bolt color intrinsics and write RecGen YAML.

RecGen expects fu/fv/pu/pv for the camera frame that matches your RGB and depth.
If depth is aligned to color (same HxW as RGB), use color intrinsics.

Requires: pip install pyorbbecsdk  (and Orbbec udev rules on Linux)

Example:
    python scripts/dump_orbbec_intrinsics.py --width 1280 --height 720 \\
        --output frame_intrinsics.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dump Orbbec color intrinsics for RecGen")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--output", type=Path, default=Path("intrinsics.yaml"))
    p.add_argument(
        "--sensor",
        choices=("color", "depth"),
        default="color",
        help="Use color when RGB/depth are aligned to color frame (default).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from pyorbbecsdk import Config, OBFormat, OBSensorType, Pipeline
    except ImportError as exc:
        raise SystemExit(
            "pyorbbecsdk is not installed.\n"
            "Install from: https://github.com/orbbec/pyorbbecsdk\n"
            "Or read intrinsics from Orbbec Viewer while streaming 1280x720."
        ) from exc

    pipeline = Pipeline()
    config = Config()
    sensor = OBSensorType.COLOR_SENSOR if args.sensor == "color" else OBSensorType.DEPTH_SENSOR
    profiles = pipeline.get_stream_profile_list(sensor)

    try:
        profile = profiles.get_video_stream_profile(args.width, args.height, OBFormat.RGB, args.fps)
    except Exception:
        profile = profiles.get_default_video_stream_profile()

    config.enable_stream(profile)
    pipeline.start(config)

    try:
        intr = profile.get_intrinsic()
        fu, fv, pu, pv = float(intr.fx), float(intr.fy), float(intr.cx), float(intr.cy)
        w, h = int(intr.width), int(intr.height)
    finally:
        pipeline.stop()

    yaml_text = (
        f"# Orbbec Femto Bolt {args.sensor} intrinsics @ {w}x{h}\n"
        f"fu: {fu}\n"
        f"fv: {fv}\n"
        f"pu: {pu}\n"
        f"pv: {pv}\n"
    )
    args.output.write_text(yaml_text)
    print(yaml_text)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
