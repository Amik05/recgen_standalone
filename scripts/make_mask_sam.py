#!/usr/bin/env python3
"""Create a RecGen object mask from an RGB image using SAM 2.

RecGen expects a single-channel mask PNG: 0 = background, 255 = object.
RGB and mask must have the same HxW resolution.

Examples:
    # Click on the object in an interactive window (needs a local display)
    python scripts/make_mask_sam.py --rgb frame_0145_rgb.png --interactive

    # Remote SSH: port-forward then click in your local browser
    #   ssh -L 8765:localhost:8765 user@remote-host
    python scripts/make_mask_sam.py --rgb frame_0145_rgb.png --interactive-web --port 8765

    # Point prompt (x y in pixel coords; can repeat --point)
    python scripts/make_mask_sam.py --rgb frame_0145_rgb.png --point 420 480 \\
        --output frame_0145_mask.png

    # Box prompt (x1 y1 x2 y2)
    python scripts/make_mask_sam.py --rgb frame_0145_rgb.png --box 300 200 550 650

    # Refine with depth (keeps only mask pixels where depth > 0)
    python scripts/make_mask_sam.py --rgb frame_0145_rgb.png --depth frame_0145_depth.png \\
        --point 420 480 --refine-depth
"""

from __future__ import annotations

import argparse
import io
import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
from PIL import Image


def _default_output(rgb_path: Path) -> Path:
    return rgb_path.with_name(rgb_path.stem.replace("_rgb", "") + "_mask.png")


def _load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _refine_with_depth(mask: np.ndarray, depth_path: Path) -> np.ndarray:
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(depth_path)
    if depth.shape != mask.shape:
        raise ValueError(f"depth {depth.shape} and mask {mask.shape} must match")
    valid = depth > 0
    refined = np.where(valid, mask, 0).astype(np.uint8)
    return refined


def _pick_points_interactive(rgb: np.ndarray) -> tuple[list[tuple[int, int]], list[int]]:
    import matplotlib.pyplot as plt

    points: list[tuple[int, int]] = []

    def onclick(event):
        if event.xdata is None or event.ydata is None:
            return
        x, y = int(round(event.xdata)), int(round(event.ydata))
        points.append((x, y))
        ax.plot(x, y, "go", markersize=8)
        ax.text(x + 6, y - 6, str(len(points)), color="lime", fontsize=10)
        fig.canvas.draw_idle()

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(rgb)
    ax.set_title("Click object foreground (close window when done)")
    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()

    if not points:
        raise SystemExit("No points selected.")
    return points, [1] * len(points)


def _pick_points_web(
    rgb: np.ndarray,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> tuple[list[tuple[int, int]], list[int]]:
    """Collect foreground/background clicks via a local web UI (SSH port-forward friendly)."""
    done = threading.Event()
    result: dict[str, list] = {"points": [], "labels": []}
    h, w = rgb.shape[:2]
    jpeg_bytes = io.BytesIO()
    Image.fromarray(rgb).save(jpeg_bytes, format="JPEG", quality=92)
    image_payload = jpeg_bytes.getvalue()

    html_page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>SAM mask — click to prompt</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 1.5rem; max-width: 1400px; }}
    h1 {{ font-size: 1.25rem; margin-bottom: 0.25rem; }}
    p {{ color: #444; margin-top: 0; }}
  .legend span {{ margin-right: 1rem; }}
  .legend .fg::before {{ content: "●"; color: limegreen; margin-right: 0.25rem; }}
  .legend .bg::before {{ content: "●"; color: crimson; margin-right: 0.25rem; }}
    #wrap {{ position: relative; display: inline-block; max-width: 100%; }}
    #img {{ max-width: 100%; height: auto; display: block; cursor: crosshair; }}
    #canvas {{ position: absolute; left: 0; top: 0; pointer-events: none; }}
    button {{ margin-top: 0.75rem; margin-right: 0.5rem; padding: 0.4rem 0.9rem; }}
    #status {{ margin-top: 0.75rem; font-weight: 600; }}
  </style>
</head>
<body>
  <h1>SAM mask prompts</h1>
  <p class="legend">
    <span class="fg">Left-click = object (foreground)</span>
    <span class="bg">Right-click = background</span>
    Image size: {w}×{h} px
  </p>
  <p>On SSH: run <code>ssh -L {port}:localhost:{port} user@host</code> then open
     <a href="http://localhost:{port}">http://localhost:{port}</a> on your laptop.</p>
  <div id="wrap">
    <img id="img" src="/image" alt="RGB" width="{w}" height="{h}" />
    <canvas id="canvas"></canvas>
  </div>
  <div>
    <button type="button" id="undo">Undo last point</button>
    <button type="button" id="clear">Clear all</button>
    <button type="button" id="submit">Generate mask</button>
  </div>
  <div id="status"></div>
  <script>
    const img = document.getElementById("img");
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    const points = [];

    function syncCanvas() {{
      const rect = img.getBoundingClientRect();
      canvas.width = rect.width;
      canvas.height = rect.height;
      canvas.style.width = rect.width + "px";
      canvas.style.height = rect.height + "px";
      redraw();
    }}

    function imageCoords(evt) {{
      const rect = img.getBoundingClientRect();
      const x = Math.round((evt.clientX - rect.left) * (img.naturalWidth / rect.width));
      const y = Math.round((evt.clientY - rect.top) * (img.naturalHeight / rect.height));
      return [x, y];
    }}

    function displayCoords(x, y) {{
      const rect = img.getBoundingClientRect();
      return [
        (x / img.naturalWidth) * rect.width,
        (y / img.naturalHeight) * rect.height,
      ];
    }}

    function redraw() {{
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      points.forEach((p, i) => {{
        const [dx, dy] = displayCoords(p.x, p.y);
        ctx.beginPath();
        ctx.arc(dx, dy, 7, 0, Math.PI * 2);
        ctx.fillStyle = p.label === 1 ? "limegreen" : "crimson";
        ctx.fill();
        ctx.strokeStyle = "black";
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.fillStyle = "white";
        ctx.font = "11px sans-serif";
        ctx.fillText(String(i + 1), dx + 9, dy - 9);
      }});
    }}

    img.addEventListener("load", syncCanvas);
    window.addEventListener("resize", syncCanvas);

    img.addEventListener("contextmenu", (e) => e.preventDefault());

    img.addEventListener("mousedown", (evt) => {{
      const [x, y] = imageCoords(evt);
      const label = evt.button === 2 ? 0 : 1;
      points.push({{ x, y, label }});
      redraw();
    }});

    document.getElementById("undo").onclick = () => {{
      points.pop();
      redraw();
    }};

    document.getElementById("clear").onclick = () => {{
      points.length = 0;
      redraw();
    }};

    document.getElementById("submit").onclick = async () => {{
      if (!points.length) {{
        document.getElementById("status").textContent = "Add at least one foreground click.";
        return;
      }}
      document.getElementById("status").textContent = "Running SAM…";
      const res = await fetch("/submit", {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify({{ points }}),
      }});
      const data = await res.json();
      document.getElementById("status").textContent = data.message || data.error || "Done.";
    }};
  </script>
</body>
</html>"""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            print(f"[sam-web] {self.address_string()} - {fmt % args}")

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                self._send(HTTPStatus.OK, html_page.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/image":
                self._send(HTTPStatus.OK, image_payload, "image/jpeg")
            else:
                self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")

        def do_POST(self):
            if urlparse(self.path).path != "/submit":
                self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
                return
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length))
            clicks = payload.get("points", [])
            if not clicks:
                self._send(
                    HTTPStatus.BAD_REQUEST,
                    json.dumps({"error": "No points submitted"}).encode(),
                    "application/json",
                )
                return
            pts, lbs = [], []
            for p in clicks:
                pts.append((int(p["x"]), int(p["y"])))
                lbs.append(int(p.get("label", 1)))
            if not any(lb == 1 for lb in lbs):
                self._send(
                    HTTPStatus.BAD_REQUEST,
                    json.dumps({"error": "Need at least one foreground (left-click) point."}).encode(),
                    "application/json",
                )
                return
            result["points"] = pts
            result["labels"] = lbs
            done.set()
            self._send(
                HTTPStatus.OK,
                json.dumps({"message": "Points received — generating mask…"}).encode(),
                "application/json",
            )

    server = ThreadingHTTPServer((host, port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    print(f"[sam-web] Open http://localhost:{port} in your browser.")
    if host == "127.0.0.1":
        print(f"[sam-web] Over SSH: ssh -L {port}:localhost:{port} <user>@<remote-host>")
    print("[sam-web] Left-click = foreground, right-click = background, then Generate mask.")

    try:
        done.wait()
    except KeyboardInterrupt:
        raise SystemExit("Cancelled.") from None
    finally:
        server.shutdown()

    return result["points"], result["labels"]


def _segment_sam2(
    rgb: np.ndarray,
    points: list[tuple[int, int]],
    labels: list[int],
    box: tuple[int, int, int, int] | None,
    model_id: str,
) -> np.ndarray:
    import torch
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    predictor = SAM2ImagePredictor.from_pretrained(model_id, device="cuda" if torch.cuda.is_available() else "cpu")
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
    mask = (masks[best] > 0).astype(np.uint8) * 255
    return mask


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create a RecGen mask with SAM 2")
    p.add_argument("--rgb", required=True, type=Path, help="RGB image path")
    p.add_argument("--depth", type=Path, help="Optional depth map (for --refine-depth)")
    p.add_argument("--output", type=Path, help="Output mask PNG (default: <stem>_mask.png)")
    p.add_argument(
        "--point",
        action="append",
        nargs=2,
        type=int,
        metavar=("X", "Y"),
        help="Foreground point prompt; repeat for multiple points",
    )
    p.add_argument(
        "--background-point",
        action="append",
        nargs=2,
        type=int,
        metavar=("X", "Y"),
        help="Background point prompt (label 0)",
    )
    p.add_argument(
        "--box",
        nargs=4,
        type=int,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Box prompt around the object",
    )
    p.add_argument(
        "--interactive",
        action="store_true",
        help="Open a matplotlib window and click foreground points (needs local display)",
    )
    p.add_argument(
        "--interactive-web",
        action="store_true",
        help="Click prompts in a browser (use with SSH port forwarding)",
    )
    p.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address for --interactive-web (default: 127.0.0.1)",
    )
    p.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port for --interactive-web (default: 8765)",
    )
    p.add_argument(
        "--refine-depth",
        action="store_true",
        help="Zero mask pixels where depth == 0",
    )
    p.add_argument(
        "--preview",
        type=Path,
        help="Optional overlay preview PNG",
    )
    p.add_argument(
        "--model",
        default="facebook/sam2.1-hiera-small",
        help="Hugging Face SAM 2.1 model id",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rgb = _load_rgb(args.rgb)

    points: list[tuple[int, int]] = []
    labels: list[int] = []

    if args.interactive and args.interactive_web:
        raise SystemExit("Use only one of --interactive or --interactive-web.")
    if args.interactive_web:
        clicked, click_labels = _pick_points_web(rgb, host=args.host, port=args.port)
        points.extend(clicked)
        labels.extend(click_labels)
    elif args.interactive:
        clicked, click_labels = _pick_points_interactive(rgb)
        points.extend(clicked)
        labels.extend(click_labels)
    if args.point:
        for x, y in args.point:
            points.append((x, y))
            labels.append(1)
    if args.background_point:
        for x, y in args.background_point:
            points.append((x, y))
            labels.append(0)

    box = tuple(args.box) if args.box else None
    if not points and box is None:
        raise SystemExit("Provide --interactive, --interactive-web, --point, or --box.")

    mask = _segment_sam2(rgb, points, labels, box, args.model)

    if args.refine_depth:
        if args.depth is None:
            raise SystemExit("--refine-depth requires --depth")
        mask = _refine_with_depth(mask, args.depth)

    out = args.output or _default_output(args.rgb)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(out)

    fg_pct = 100.0 * (mask > 0).mean()
    print(f"Saved mask: {out}  ({mask.shape[1]}x{mask.shape[0]}, {fg_pct:.2f}% foreground)")

    if args.preview:
        overlay = rgb.copy()
        overlay[mask > 0] = (overlay[mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
        Image.fromarray(overlay).save(args.preview)
        print(f"Saved preview: {args.preview}")


if __name__ == "__main__":
    main()
