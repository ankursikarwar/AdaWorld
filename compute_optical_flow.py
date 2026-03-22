"""
Frame-level optical flow annotation pipeline using RAFT-small (torchvision).

For each consecutive frame pair (sampled at --frame_sample_fps=16 fps, capped at
--max_frame_pairs per video), computes a float32 descriptor of shape (8,):

  [0] mean_u            — mean horizontal flow (dominant motion direction)
  [1] mean_v            — mean vertical flow
  [2] mean_magnitude    — mean per-pixel flow magnitude (overall motion speed)
  [3] std_magnitude     — std of magnitudes; low=uniform camera, high=localised agent
  [4] fg_mean_magnitude — residual after RANSAC homography background subtraction
  [5] fg_max_magnitude  — max residual magnitude
  [6] direction_cos     — cos(atan2(mean_v, mean_u))  unit-circle encoding
  [7] direction_sin     — sin(atan2(mean_v, mean_u))

W&B logs a table per video with frame_0, frame_1, flow_viz (HSV colour wheel),
and all 8 descriptor components.

Usage:
  # CPU (slow but works anywhere)
  python compute_optical_flow.py --max_videos 50 --wandb_project mira-flow

  # GPU (much faster)
  python compute_optical_flow.py --device cuda --max_videos 50 --wandb_project mira-flow

  # SLURM array
  python compute_optical_flow.py --device cuda --task_id $SLURM_ARRAY_TASK_ID --n_tasks 8
"""

import os
import cv2
import json
import math
import argparse
import traceback
import numpy as np
from pathlib import Path
from typing import Optional

import torch

# Patch: torch 2.0.x doesn't accept weights_only in load_state_dict_from_url;
# ptlflow passes it unconditionally — strip the kwarg for older torch versions.
_orig_load_url = torch.hub.load_state_dict_from_url
def _patched_load_url(url, *args, weights_only=None, **kwargs):
    return _orig_load_url(url, *args, **kwargs)
torch.hub.load_state_dict_from_url = _patched_load_url

import ptlflow

# Suppress spurious H.264 decoder warnings (mmco: unref short failure, etc.)
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
cv2.setLogLevel(3)  # 3 = ERROR, suppresses WARNING-level ffmpeg messages


# ── SEA-RAFT helpers ───────────────────────────────────────────────────────────

def load_sea_raft(device: str, variant: str = "sea_raft_m"):
    """Load SEA-RAFT via ptlflow. variant: sea_raft_s / sea_raft_m / sea_raft_l."""
    model = ptlflow.get_model(variant, ckpt_path="spring")
    model = model.to(device).eval()
    return model


@torch.no_grad()
def compute_flow(model, f0_bgr, f1_bgr, device: str) -> np.ndarray:
    """
    Run SEA-RAFT on a frame pair.
    Returns flow as (H, W, 2) float32 numpy array [u, v] in original resolution.
    """
    f0_rgb = cv2.cvtColor(f0_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    f1_rgb = cv2.cvtColor(f1_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    t0 = torch.from_numpy(f0_rgb).permute(2, 0, 1)   # (3, H, W)
    t1 = torch.from_numpy(f1_rgb).permute(2, 0, 1)

    # ptlflow expects (B, N, C, H, W) where N=2 (pair of frames)
    images = torch.stack([t0, t1]).unsqueeze(0).to(device)   # (1, 2, 3, H, W)

    output = model({"images": images})
    # flows shape: (B, 1, 2, H, W) — take first batch, first prediction
    flow = output["flows"][0, 0].cpu().numpy().transpose(1, 2, 0)  # (H, W, 2)
    return flow


# ── Descriptor computation ────────────────────────────────────────────────────

def ransac_background_flow(flow_uv: np.ndarray, step: int = 8) -> np.ndarray:
    """
    Fit a homography to the dense flow using RANSAC, then return the
    residual (foreground) flow after subtracting the background model.
    Returns residual flow (H, W, 2).
    """
    H, W, _ = flow_uv.shape
    y_coords, x_coords = np.mgrid[0:H, 0:W]
    src_pts = np.stack([x_coords.ravel(), y_coords.ravel()], axis=1).astype(np.float32)
    u = flow_uv[..., 0].ravel()
    v = flow_uv[..., 1].ravel()
    dst_pts = src_pts + np.stack([u, v], axis=1)

    # Subsample for RANSAC speed
    idx = np.arange(0, len(src_pts), step)
    H_mat, _ = cv2.findHomography(src_pts[idx], dst_pts[idx], cv2.RANSAC, 3.0)

    if H_mat is None:
        return flow_uv  # fallback: treat everything as foreground

    warped = cv2.perspectiveTransform(
        src_pts.reshape(-1, 1, 2), H_mat
    ).reshape(-1, 2)

    bg_u = (warped[:, 0] - src_pts[:, 0]).reshape(H, W)
    bg_v = (warped[:, 1] - src_pts[:, 1]).reshape(H, W)
    residual = flow_uv - np.stack([bg_u, bg_v], axis=-1)
    return residual


def flow_to_descriptor(flow_uv: np.ndarray) -> np.ndarray:
    """
    Compute the 8-d float32 descriptor from a (H, W, 2) flow field.
    """
    u = flow_uv[..., 0]
    v = flow_uv[..., 1]
    magnitude = np.sqrt(u ** 2 + v ** 2)

    mean_u = float(u.mean())
    mean_v = float(v.mean())
    mean_magnitude = float(magnitude.mean())
    std_magnitude = float(magnitude.std())

    residual = ransac_background_flow(flow_uv)
    res_mag = np.sqrt(residual[..., 0] ** 2 + residual[..., 1] ** 2)
    fg_mean_magnitude = float(res_mag.mean())
    fg_max_magnitude = float(res_mag.max())

    angle = math.atan2(mean_v, mean_u)
    direction_cos = math.cos(angle)
    direction_sin = math.sin(angle)

    descriptor = np.array([
        mean_u, mean_v,
        mean_magnitude, std_magnitude,
        fg_mean_magnitude, fg_max_magnitude,
        direction_cos, direction_sin,
    ], dtype=np.float32)
    return descriptor


# ── Flow visualisation ────────────────────────────────────────────────────────

def flow_to_rgb(flow_uv: np.ndarray) -> np.ndarray:
    """
    Render a (H, W, 2) flow field as an HSV colour-wheel image (H, W, 3) uint8.
    Hue = direction, Value = normalised magnitude, Saturation = full.
    """
    u = flow_uv[..., 0]
    v = flow_uv[..., 1]
    magnitude = np.sqrt(u ** 2 + v ** 2)
    angle = np.arctan2(v, u)                          # [-pi, pi]

    mag_norm = (magnitude / (magnitude.max() + 1e-8) * 255).astype(np.uint8)
    hue = ((angle + np.pi) / (2 * np.pi) * 179).astype(np.uint8)  # [0,179]

    hsv = np.stack([hue, np.full_like(hue, 255), mag_norm], axis=-1)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def flow_to_quiver(frame_bgr: np.ndarray, flow_uv: np.ndarray,
                   grid_step: int = 40, scale: float = 5.0) -> np.ndarray:
    """
    Draw flow arrows on top of frame_bgr at a sparse grid.
    Arrow direction = flow direction; arrow length proportional to magnitude.
    Arrows are coloured by direction (HSV hue wheel) so they match flow_viz.
    Only arrows with magnitude above the 50th percentile are drawn to avoid clutter.
    Returns RGB uint8 image.
    """
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).copy()
    H, W = flow_uv.shape[:2]

    u = flow_uv[..., 0]
    v = flow_uv[..., 1]
    magnitude = np.sqrt(u ** 2 + v ** 2)
    max_mag = magnitude.max() + 1e-8

    # Skip arrows below median magnitude to show only meaningful motion
    threshold = float(np.percentile(magnitude, 50))

    ys = np.arange(grid_step // 2, H, grid_step)
    xs = np.arange(grid_step // 2, W, grid_step)

    for y in ys:
        for x in xs:
            fx = float(u[y, x])
            fy = float(v[y, x])
            mag = float(magnitude[y, x])
            if mag < threshold or mag < 0.5:
                continue

            # Colour by direction (hue wheel), brightness by relative magnitude
            angle = math.atan2(fy, fx)                      # [-pi, pi]
            hue = int((angle + math.pi) / (2 * math.pi) * 179)  # [0, 179]
            val = int(mag / max_mag * 200 + 55)              # [55, 255]
            hsv_pixel = np.uint8([[[hue, 255, val]]])
            rgb_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2RGB)[0, 0]
            color = (int(rgb_pixel[0]), int(rgb_pixel[1]), int(rgb_pixel[2]))

            x2 = int(x + fx * scale)
            y2 = int(y + fy * scale)
            x2 = max(0, min(W - 1, x2))
            y2 = max(0, min(H - 1, y2))

            cv2.arrowedLine(img, (x, y), (x2, y2), color,
                            thickness=2, tipLength=0.25)
    return img


# ── Frame utilities ────────────────────────────────────────────────────────────

def extract_frames_at_fps(video_path: str, sample_fps: float) -> list:
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(src_fps / sample_fps))
    frames = []
    for idx in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append((frame, idx / src_fps))
    cap.release()
    return frames


def video_meta(video_path: str) -> tuple:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total / fps, fps, total


def bgr_to_rgb(frame_bgr) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


# ── W&B logger ─────────────────────────────────────────────────────────────────

class FlowWandbLogger:
    """
    Logs optical flow annotations to a W&B table.
    Columns: video_name | pair_idx | t0 | t1 | frame_0 | frame_1 | flow_viz
           | mean_u | mean_v | mean_magnitude | std_magnitude
           | fg_mean_magnitude | fg_max_magnitude | direction_cos | direction_sin

    Rows accumulate in memory; table is rebuilt from scratch on each flush
    (wandb freezes Table objects once logged).
    """
    COLUMNS = [
        "video_name", "pair_idx", "t0", "t1",
        "frame_0", "frame_1", "flow_viz", "flow_arrows",
        "mean_u", "mean_v", "mean_magnitude", "std_magnitude",
        "fg_mean_magnitude", "fg_max_magnitude",
        "direction_cos", "direction_sin",
    ]

    def __init__(self, project: str, run_name: Optional[str] = None, config: dict = None):
        import wandb
        self.wandb = wandb
        self.run = wandb.init(project=project, name=run_name, config=config or {})
        self._rows = []

    def log_pair(self, video_name: str, p: dict):
        """Log a single frame pair immediately after it's computed."""
        desc = p["descriptor"]
        # Store raw numpy arrays — wandb.Image is created fresh on each table
        # rebuild to avoid stale references (W&B uploads image data on first use;
        # reusing the same Image object in a new table results in empty cells).
        self._rows.append([
            video_name,
            p["pair_idx"],
            round(p["t0"], 3),
            round(p["t1"], 3),
            bgr_to_rgb(p["f0_bgr"]),
            bgr_to_rgb(p["f1_bgr"]),
            flow_to_rgb(p["flow_uv"]),
            flow_to_quiver(p["f0_bgr"], p["flow_uv"]),
            float(desc[0]),   # mean_u
            float(desc[1]),   # mean_v
            float(desc[2]),   # mean_magnitude
            float(desc[3]),   # std_magnitude
            float(desc[4]),   # fg_mean_magnitude
            float(desc[5]),   # fg_max_magnitude
            float(desc[6]),   # direction_cos
            float(desc[7]),   # direction_sin
        ])

        # Rebuild full table and log immediately.
        # Images are indices 4-7; wrap in fresh wandb.Image each time.
        IMAGE_COLS = {4, 5, 6, 7}
        table = self.wandb.Table(columns=self.COLUMNS)
        for row in self._rows:
            table.add_data(*[
                self.wandb.Image(v) if i in IMAGE_COLS else v
                for i, v in enumerate(row)
            ])
        self.run.log({"flow_annotations": table})
        print(f"  [W&B] Logged {len(self._rows)} pairs total.", flush=True)

    def finish(self):
        self.run.finish()


# ── Main pipeline ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video_dir",
                   default="/network/scratch/a/ankur.sikarwar/WORLD_MODEL_PROJECT/data/mira/train")
    p.add_argument("--output_dir", default=None,
                   help="Output dir. Defaults to .../flow_annotations/sea_raft_m/")
    p.add_argument("--model_variant", default="sea_raft_m",
                   choices=["sea_raft_s", "sea_raft_m", "sea_raft_l"],
                   help="SEA-RAFT model size.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--frame_sample_fps", type=float, default=4.0,
                   help="Sampling rate matching LAM training granularity.")
    p.add_argument("--max_frame_pairs", type=int, default=15,
                   help="Max pairs per video, uniformly subsampled.")
    p.add_argument("--max_videos", type=int, default=None)
    p.add_argument("--sample_seed", type=int, default=None)
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--task_id", type=int, default=None)
    p.add_argument("--n_tasks", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()

    output_dir = args.output_dir or os.path.join(
        "/network/scratch/a/ankur.sikarwar/WORLD_MODEL_PROJECT/data/mira/flow_annotations",
        args.model_variant
    )
    os.makedirs(output_dir, exist_ok=True)

    video_files = sorted(Path(args.video_dir).glob("*.mp4"))

    if args.max_videos and args.max_videos < len(video_files):
        if args.sample_seed is not None:
            import random
            rng = random.Random(args.sample_seed)
            video_files = sorted(rng.sample(video_files, args.max_videos))
        else:
            stride = len(video_files) / args.max_videos
            video_files = [video_files[int(i * stride)] for i in range(args.max_videos)]

    if args.task_id is not None:
        video_files = [v for i, v in enumerate(video_files)
                       if i % args.n_tasks == args.task_id]

    print(f"Device  : {args.device}")
    print(f"Videos  : {len(video_files)}")
    print(f"Output  : {output_dir}")

    print(f"Loading SEA-RAFT ({args.model_variant}) ...", flush=True)
    model = load_sea_raft(args.device, args.model_variant)
    print(f"SEA-RAFT loaded.", flush=True)

    wb_logger = None
    if args.wandb_project:
        wb_logger = FlowWandbLogger(
            project=args.wandb_project,
            run_name=args.wandb_run_name,
            config=vars(args),
        )

    for video_path in video_files:
        out_path = Path(output_dir) / (video_path.stem + ".json")
        if out_path.exists():
            continue

        try:
            duration_s, fps, total_frames = video_meta(str(video_path))
            all_frames = extract_frames_at_fps(str(video_path), args.frame_sample_fps)

            # Build consecutive pairs then uniformly subsample
            all_pairs = [(all_frames[i], all_frames[i + 1])
                         for i in range(len(all_frames) - 1)]
            if args.max_frame_pairs and len(all_pairs) > args.max_frame_pairs:
                s = len(all_pairs) / args.max_frame_pairs
                all_pairs = [all_pairs[int(i * s)] for i in range(args.max_frame_pairs)]

            result = {
                "video_path": str(video_path),
                "duration_s": round(duration_s, 2),
                "fps": round(fps, 2),
                "total_frames": total_frames,
                "pairs": [],
            }
            wb_pairs = []

            for i, ((f0, t0), (f1, t1)) in enumerate(all_pairs):
                print(f"  pair {i+1}/{len(all_pairs)}  t={t0:.2f}s → {t1:.2f}s", flush=True)

                flow_uv = compute_flow(model, f0, f1, args.device)
                descriptor = flow_to_descriptor(flow_uv)

                result["pairs"].append({
                    "pair_idx": i,
                    "time_start_s": round(t0, 3),
                    "time_end_s": round(t1, 3),
                    "descriptor": descriptor.tolist(),
                    "components": {
                        "mean_u":            float(descriptor[0]),
                        "mean_v":            float(descriptor[1]),
                        "mean_magnitude":    float(descriptor[2]),
                        "std_magnitude":     float(descriptor[3]),
                        "fg_mean_magnitude": float(descriptor[4]),
                        "fg_max_magnitude":  float(descriptor[5]),
                        "direction_cos":     float(descriptor[6]),
                        "direction_sin":     float(descriptor[7]),
                    }
                })
                if wb_logger:
                    wb_logger.log_pair(video_path.stem, {
                        "pair_idx": i, "t0": t0, "t1": t1,
                        "f0_bgr": f0, "f1_bgr": f1,
                        "flow_uv": flow_uv, "descriptor": descriptor,
                    })

            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)

            print(f"OK  {video_path.name}  ({len(all_pairs)} pairs)", flush=True)

        except Exception as e:
            print(f"ERR {video_path.name}: {e}", flush=True)
            traceback.print_exc()

    if wb_logger:
        wb_logger.finish()


if __name__ == "__main__":
    main()
