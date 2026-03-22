"""
Automatic video annotation pipeline.

Generates captions at three hierarchy levels:
  - clip    : overall action across the full clip (multi-frame input)
  - segment : action per temporal segment (multi-frame input)
  - frame   : action between every consecutive frame pair (2-frame input)

Backends:
  - gemini : google-genai SDK with Files API (upload video once, query N times)
  - openai : OpenAI-compatible client (works with local vLLM or OpenAI API)

W&B logging:
  Three tables are maintained (clip / segment / frame_pair).
  Each table row includes the caption + media (video or frame images).
  Tables are flushed to W&B every --wandb_log_every videos to avoid OOM.

Usage examples:
  # Gemini API + W&B logging
  python annotate_videos.py --backend gemini --api_key $GEMINI_API_KEY \
      --wandb_project adaworld-annotations

  # Local vLLM  (serve first: vllm serve Qwen/Qwen2.5-VL-7B-Instruct)
  python annotate_videos.py --backend openai --model Qwen/Qwen2.5-VL-7B-Instruct \
      --api_base http://localhost:8000/v1 --wandb_project adaworld-annotations

  # SLURM sharding
  python annotate_videos.py --backend gemini --task_id 3 --n_tasks 8 \
      --wandb_project adaworld-annotations
"""

import os
import cv2
import json
import base64
import argparse
import traceback
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


# ── Prompts ───────────────────────────────────────────────────────────────────

CLIP_PROMPT = (
    "These frames are uniformly sampled from a video clip. "
    "Describe the overall action or activity being performed in the clip. "
    "Focus on what is happening, not the visual appearance. "
    "Be concise: 1 sentence."
)

SEGMENT_PROMPT = (
    "These frames are sampled from segment {idx} of {total} of a video clip "
    "(covering {start_s:.1f}s - {end_s:.1f}s). "
    "State only the single atomic action or motion that captures the transition across frames. "
    "Use a short verb phrase (e.g. 'car turns left', 'hand reaches for cup').  Be concise."
    "No descriptions of appearance or background."
)

FRAME_PAIR_PROMPT = (
    "These are two consecutive video frames (at {t0:.2f}s and {t1:.2f}s). "
    "State only the single atomic action or motion occurring between them. "
    "Use a short verb phrase (e.g. 'car turns left', 'hand reaches for cup'). "
    "No descriptions of appearance or background. Maximum 5 words."
)


# ── Frame utilities ────────────────────────────────────────────────────────────

def extract_frames_uniform(video_path: str, n: int,
                           start_frac: float = 0.0,
                           end_frac: float = 1.0) -> list:
    """Extract n frames uniformly from [start_frac, end_frac].
    Returns list of (frame_bgr, timestamp_s)."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_idx = int(start_frac * total)
    end_idx = int(end_frac * total)
    span = max(end_idx - start_idx, 1)
    indices = [start_idx + int(i * span / n) for i in range(n)]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append((frame, idx / fps))
    cap.release()
    return frames


def extract_frames_at_fps(video_path: str, sample_fps: float) -> list:
    """Extract frames at a fixed sample rate.
    Returns list of (frame_bgr, timestamp_s)."""
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
    """Returns (duration_s, fps, total_frames)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total / fps, fps, total


def encode_frame(frame_bgr) -> str:
    """Encode BGR frame to base64 JPEG string."""
    _, buf = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode()


def bgr_to_rgb(frame_bgr):
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


# ── Backend interface ──────────────────────────────────────────────────────────

class VideoBackend(ABC):
    @abstractmethod
    def query_frames(self, frame_groups: list, prompts: list) -> list:
        """
        frame_groups : list of frame-groups, each is list of (frame_bgr, ts)
        prompts      : list of prompt strings, same length
        Returns      : list of caption strings
        """
        pass

    def query(self, frames: list, prompt: str) -> str:
        return self.query_frames([frames], [prompt])[0]


# ── Gemini backend ─────────────────────────────────────────────────────────────

class GeminiBackend(VideoBackend):
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        from google import genai
        from google.genai import types
        self.client = genai.Client(api_key=api_key)
        self.types = types
        self.model_name = model

    def upload_video(self, video_path: str):
        import time
        size_mb = Path(video_path).stat().st_size / 1e6
        print(f"  [Gemini] Uploading {Path(video_path).name} ({size_mb:.1f} MB) ...", flush=True)
        t0 = time.time()
        video_file = self.client.files.upload(file=video_path)
        print(f"  [Gemini] Upload done in {time.time()-t0:.1f}s, waiting for processing ...", flush=True)
        timeout = 120  # seconds
        while video_file.state.name == "PROCESSING":
            elapsed = time.time() - t0
            if elapsed > timeout:
                raise RuntimeError(f"Gemini processing timed out after {timeout}s")
            print(f"    processing ... {elapsed:.0f}s", flush=True)
            time.sleep(5)
            video_file = self.client.files.get(name=video_file.name)
        if video_file.state.name != "ACTIVE":
            raise RuntimeError(f"Gemini upload failed: {video_file.state.name}")
        print(f"  [Gemini] Ready in {time.time()-t0:.1f}s total", flush=True)
        return video_file

    def delete_video(self, video_file):
        try:
            self.client.files.delete(name=video_file.name)
        except Exception:
            pass

    def query_frames(self, frame_groups: list, prompts: list) -> list:
        from google.genai import types as gtypes
        captions = []
        for frames, prompt in zip(frame_groups, prompts):
            parts = []
            for frame_bgr, _ in frames:
                _, buf = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                parts.append(self.types.Part.from_bytes(
                    data=bytes(buf),
                    mime_type="image/jpeg",
                ))
            parts.append(prompt)
            print(f"    [Gemini] querying {len(frames)} frames ...", flush=True)
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=parts,
                config=gtypes.GenerateContentConfig(http_options=gtypes.HttpOptions(timeout=300000)),
            )
            captions.append(response.text.strip())
        return captions

    def query_with_video_file(self, video_file, prompt: str) -> str:
        from google.genai import types as gtypes
        print(f"    [Gemini] querying with video file ...", flush=True)
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[video_file, prompt],
            config=gtypes.GenerateContentConfig(http_options=gtypes.HttpOptions(timeout=300000)),
        )
        return response.text.strip()


# ── OpenAI-compatible backend ──────────────────────────────────────────────────

class OpenAIBackend(VideoBackend):
    def __init__(self, model: str, api_key: str = "dummy", api_base: Optional[str] = None):
        from openai import OpenAI
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=api_base)

    def query_frames(self, frame_groups: list, prompts: list) -> list:
        captions = []
        for frames, prompt in zip(frame_groups, prompts):
            content = [{"type": "text", "text": prompt}]
            for frame_bgr, _ in frames:
                b64 = encode_frame(frame_bgr)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                })
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                max_tokens=256,
            )
            captions.append(resp.choices[0].message.content.strip())
        return captions


# ── W&B logger ─────────────────────────────────────────────────────────────────

class WandbLogger:
    """
    Logs annotations to three W&B tables:
      - clip_table    : video_name | video | clip_caption
      - segment_table : video_name | segment_idx | start_s | end_s | frames_grid | caption
      - pair_table    : video_name | pair_idx | t0 | t1 | frame_0 | frame_1 | caption

    Tables are flushed to W&B every `flush_every` videos to avoid OOM.
    """

    def __init__(self, project: str, run_name: Optional[str] = None,
                 flush_every: int = 50, config: dict = None):
        import wandb
        self.wandb = wandb
        self.run = wandb.init(
            project=project,
            name=run_name,
            config=config or {},
        )
        self.flush_every = flush_every
        self._pending = 0
        # Store raw row data — tables are rebuilt from scratch on each flush
        # because wandb freezes Table objects once logged
        self._clip_rows = []
        self._segment_rows = []
        self._pair_rows = []

    def log_annotation(self, result: dict, video_path: str):
        """Add one video's annotations to the in-memory tables."""
        name = Path(video_path).stem

        # ── clip row ─────────────────────────────────────────────────────────
        if "clip" in result:
            self._clip_rows.append([
                name,
                self.wandb.Video(str(video_path), fps=10, format="mp4"),
                result["clip"]["caption"],
            ])

        # ── segment rows ─────────────────────────────────────────────────────
        for seg in result.get("segments", []):
            raw_frames = seg.get("_frames", [])
            if raw_frames:
                rgb_frames = [bgr_to_rgb(f) for f, _ in raw_frames]
                h = 128
                resized = []
                for rf in rgb_frames:
                    ratio = h / rf.shape[0]
                    w = int(rf.shape[1] * ratio)
                    resized.append(cv2.resize(rf, (w, h)))
                frames_img = self.wandb.Image(np.concatenate(resized, axis=1))
            else:
                frames_img = None
            self._segment_rows.append([
                name, seg["segment_idx"], seg["start_s"], seg["end_s"],
                frames_img, seg["caption"],
            ])

        # ── frame-pair rows ───────────────────────────────────────────────────
        for pair in result.get("frames", []):
            f0 = pair.get("_frame_0")
            f1 = pair.get("_frame_1")
            self._pair_rows.append([
                name,
                pair["pair_idx"],
                pair["time_start_s"],
                pair["time_end_s"],
                self.wandb.Image(bgr_to_rgb(f0)) if f0 is not None else None,
                self.wandb.Image(bgr_to_rgb(f1)) if f1 is not None else None,
                pair["caption"],
            ])

        self._pending += 1

    def flush(self):
        """Rebuild tables from accumulated row data and log to W&B."""
        if self._pending == 0:
            return

        clip_table = self.wandb.Table(columns=["video_name", "video", "clip_caption"])
        for row in self._clip_rows:
            clip_table.add_data(*row)

        seg_table = self.wandb.Table(columns=["video_name", "segment_idx", "start_s", "end_s", "frames_grid", "caption"])
        for row in self._segment_rows:
            seg_table.add_data(*row)

        pair_table = self.wandb.Table(columns=["video_name", "pair_idx", "t0", "t1", "frame_0", "frame_1", "caption"])
        for row in self._pair_rows:
            pair_table.add_data(*row)

        self.run.log({
            "clip_annotations": clip_table,
            "segment_annotations": seg_table,
            "frame_pair_annotations": pair_table,
        })
        self._pending = 0
        print(f"  [W&B] Logged {len(self._clip_rows)} videos to tables.", flush=True)

    def finish(self):
        self.flush()
        self.run.finish()


# ── Main annotator ─────────────────────────────────────────────────────────────

def _strip_internal(result: dict) -> dict:
    """Remove _frame* keys before saving to JSON."""
    clean = {}
    for k, v in result.items():
        if k.startswith("_"):
            continue
        if isinstance(v, list):
            clean[k] = [{kk: vv for kk, vv in item.items()
                         if not kk.startswith("_")} for item in v]
        elif isinstance(v, dict):
            clean[k] = {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
        else:
            clean[k] = v
    return clean


class VideoAnnotator:
    def __init__(self, backend: VideoBackend):
        self.backend = backend

    def annotate(
        self,
        video_path: str,
        levels: list = None,
        n_segments: int = 4,
        clip_frames: int = 8,
        segment_frames: int = 16,
        segment_sample_fps: float = None,
        max_segments: int = None,
        frame_sample_fps: float = 2.0,
        max_frame_pairs: int = 15,
    ) -> dict:
        """
        Annotate at requested levels. Internal result contains _frame* keys
        with raw numpy arrays for W&B logging; call _strip_internal() before
        saving to JSON.

        If segment_sample_fps is set, the video is sampled at that FPS and
        consecutive windows of segment_frames frames form each segment
        (n_segments is computed automatically).
        """
        if levels is None:
            levels = ["clip", "segment", "frame"]

        duration_s, fps, total_frames = video_meta(video_path)

        result = {
            "video_path": str(video_path),
            "duration_s": round(duration_s, 2),
            "fps": round(fps, 2),
            "total_frames": total_frames,
            "backend": type(self.backend).__name__,
        }

        # ── Clip level ────────────────────────────────────────────────────────
        if "clip" in levels:
            clip_f = extract_frames_uniform(video_path, n=clip_frames)
            clip_caption = self.backend.query(clip_f, CLIP_PROMPT)
            result["clip"] = {
                "caption": clip_caption,
                "_frames": clip_f,          # kept for W&B, stripped before JSON
            }

        # ── Segment level ─────────────────────────────────────────────────────
        if "segment" in levels:
            segments = []
            if segment_sample_fps is not None:
                # Sample entire video at fixed FPS, chunk into segment_frames-frame windows
                all_seg_frames = extract_frames_at_fps(video_path, sample_fps=segment_sample_fps)
                n_segs = max(1, len(all_seg_frames) // segment_frames)
                seg_indices = list(range(n_segs))
                if max_segments and n_segs > max_segments:
                    stride = n_segs / max_segments
                    seg_indices = [int(i * stride) for i in range(max_segments)]
                    n_segs = max_segments
                for i, seg_i in enumerate(seg_indices):
                    seg_f = all_seg_frames[seg_i * segment_frames: (seg_i + 1) * segment_frames]
                    start_s = seg_f[0][1]
                    end_s = seg_f[-1][1]
                    prompt = SEGMENT_PROMPT.format(
                        idx=i + 1, total=n_segs, start_s=start_s, end_s=end_s
                    )
                    caption = self.backend.query(seg_f, prompt)
                    segments.append({
                        "segment_idx": i,
                        "start_s": round(start_s, 2),
                        "end_s": round(end_s, 2),
                        "caption": caption,
                        "_frames": seg_f,
                    })
            else:
                for i in range(n_segments):
                    start_frac = i / n_segments
                    end_frac = (i + 1) / n_segments
                    start_s = start_frac * duration_s
                    end_s = end_frac * duration_s
                    seg_f = extract_frames_uniform(
                        video_path, n=segment_frames,
                        start_frac=start_frac, end_frac=end_frac,
                    )
                    prompt = SEGMENT_PROMPT.format(
                        idx=i + 1, total=n_segments, start_s=start_s, end_s=end_s
                    )
                    caption = self.backend.query(seg_f, prompt)
                    segments.append({
                        "segment_idx": i,
                        "start_s": round(start_s, 2),
                        "end_s": round(end_s, 2),
                        "caption": caption,
                        "_frames": seg_f,       # kept for W&B
                    })
            result["segments"] = segments

        # ── Frame-pair level ──────────────────────────────────────────────────
        if "frame" in levels:
            all_frames = extract_frames_at_fps(video_path, sample_fps=frame_sample_fps)
            # Build all consecutive pairs then uniformly subsample to max_frame_pairs
            all_pairs = [(all_frames[i], all_frames[i + 1]) for i in range(len(all_frames) - 1)]
            if max_frame_pairs and len(all_pairs) > max_frame_pairs:
                stride = len(all_pairs) / max_frame_pairs
                all_pairs = [all_pairs[int(i * stride)] for i in range(max_frame_pairs)]
            frame_pairs = []
            for i, ((f0, t0), (f1, t1)) in enumerate(all_pairs):
                prompt = FRAME_PAIR_PROMPT.format(t0=t0, t1=t1)
                caption = self.backend.query([(f0, t0), (f1, t1)], prompt)
                frame_pairs.append({
                    "pair_idx": i,
                    "time_start_s": round(t0, 3),
                    "time_end_s": round(t1, 3),
                    "caption": caption,
                    "_frame_0": f0,
                    "_frame_1": f1,
                })
            result["frames"] = frame_pairs

        return result

    def annotate_gemini_native(
        self,
        video_path: str,
        n_segments: int = 4,
        clip_frames: int = 8,
        segment_frames: int = 6,
        frame_sample_fps: float = 2.0,
        max_frame_pairs: int = 15,
    ) -> dict:
        """
        Gemini-specific: uploads video once via Files API for clip/segment,
        uses inline frame pairs for frame-pair level.
        """
        assert isinstance(self.backend, GeminiBackend)
        duration_s, fps, total_frames = video_meta(video_path)

        result = {
            "video_path": str(video_path),
            "duration_s": round(duration_s, 2),
            "fps": round(fps, 2),
            "total_frames": total_frames,
            "backend": "GeminiNative",
        }

        video_file = self.backend.upload_video(video_path)
        try:
            # Clip level — use uploaded video file
            clip_caption = self.backend.query_with_video_file(video_file, CLIP_PROMPT)
            clip_f = extract_frames_uniform(video_path, n=clip_frames)
            result["clip"] = {"caption": clip_caption, "_frames": clip_f}

            # Segment level — reference uploaded video with timestamps in prompt
            segments = []
            for i in range(n_segments):
                start_s = (i / n_segments) * duration_s
                end_s = ((i + 1) / n_segments) * duration_s
                prompt = (
                    f"Focus on the video between {start_s:.1f}s and {end_s:.1f}s. "
                    + SEGMENT_PROMPT.format(
                        idx=i + 1, total=n_segments, start_s=start_s, end_s=end_s
                    )
                )
                caption = self.backend.query_with_video_file(video_file, prompt)
                seg_f = extract_frames_uniform(
                    video_path, n=segment_frames,
                    start_frac=i / n_segments, end_frac=(i + 1) / n_segments,
                )
                segments.append({
                    "segment_idx": i,
                    "start_s": round(start_s, 2),
                    "end_s": round(end_s, 2),
                    "caption": caption,
                    "_frames": seg_f,
                })
            result["segments"] = segments

        finally:
            self.backend.delete_video(video_file)

        # Frame-pair level — inline images
        all_frames = extract_frames_at_fps(video_path, sample_fps=frame_sample_fps)
        all_pairs = [(all_frames[i], all_frames[i + 1]) for i in range(len(all_frames) - 1)]
        if max_frame_pairs and len(all_pairs) > max_frame_pairs:
            stride = len(all_pairs) / max_frame_pairs
            all_pairs = [all_pairs[int(i * stride)] for i in range(max_frame_pairs)]
        frame_pairs = []
        for i, ((f0, t0), (f1, t1)) in enumerate(all_pairs):
            prompt = FRAME_PAIR_PROMPT.format(t0=t0, t1=t1)
            caption = self.backend.query([(f0, t0), (f1, t1)], prompt)
            frame_pairs.append({
                "pair_idx": i,
                "time_start_s": round(t0, 3),
                "time_end_s": round(t1, 3),
                "caption": caption,
                "_frame_0": f0,
                "_frame_1": f1,
            })
        result["frames"] = frame_pairs

        return result


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video_dir",
                   default="/network/scratch/a/ankur.sikarwar/WORLD_MODEL_PROJECT/data/mira/train")
    p.add_argument("--output_dir",
                   default=None,
                   help="Output directory. Defaults to .../annotations/<model_name>/")

    # Backend
    p.add_argument("--backend", choices=["gemini", "openai"], default="gemini")
    p.add_argument("--model", default=None,
                   help="Model name. Gemini default: gemini-2.0-flash. "
                        "OpenAI/vLLM: e.g. gpt-4o or Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--api_key", default=None)
    p.add_argument("--api_base", default=None,
                   help="Base URL for local vLLM, e.g. http://localhost:8000/v1")
    p.add_argument("--gemini_native", action="store_true",
                   help="Use Gemini Files API for clip/segment queries.")

    # Annotation settings
    p.add_argument("--levels", nargs="+", choices=["clip", "segment", "frame"],
                   default=["clip", "segment", "frame"],
                   help="Which annotation levels to run. E.g. --levels segment")
    p.add_argument("--n_segments", type=int, default=4)
    p.add_argument("--clip_frames", type=int, default=8)
    p.add_argument("--segment_frames", type=int, default=16)
    p.add_argument("--segment_sample_fps", type=float, default=None,
                   help="If set, sample the video at this FPS and split into segments of "
                        "--segment_frames frames each (n_segments computed automatically).")
    p.add_argument("--max_segments", type=int, default=None,
                   help="Cap the number of segments per video. Segments are sampled uniformly "
                        "when the video has more than this many.")
    p.add_argument("--frame_sample_fps", type=float, default=16.0,
                   help="Frame sampling rate for frame-pair captions. Default 30fps matches LAM training granularity.")
    p.add_argument("--max_frame_pairs", type=int, default=0,
                   help="Max consecutive frame pairs to annotate per video. Sampled uniformly.")

    # W&B
    p.add_argument("--wandb_project", default=None,
                   help="W&B project name. Omit to disable W&B logging.")
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_log_every", type=int, default=50,
                   help="Flush W&B tables every N videos (memory control).")

    # Sampling
    p.add_argument("--max_videos", type=int, default=None,
                   help="Annotate at most N videos, sampled uniformly by stride across the full list.")
    p.add_argument("--sample_seed", type=int, default=None,
                   help="If set, sample --max_videos randomly with this seed instead of uniform stride.")

    # SLURM sharding
    p.add_argument("--task_id", type=int, default=None)
    p.add_argument("--n_tasks", type=int, default=1)

    return p.parse_args()


def build_backend(args) -> VideoBackend:
    if args.backend == "gemini":
        api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Gemini requires --api_key or GEMINI_API_KEY env var.")
        return GeminiBackend(api_key=api_key, model=args.model or "gemini-2.0-flash")
    else:
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "dummy")
        if not args.model:
            raise ValueError("OpenAI backend requires --model.")
        return OpenAIBackend(model=args.model, api_key=api_key, api_base=args.api_base)


def main():
    args = parse_args()

    # Default output dir includes model name so different models don't overwrite each other
    model_name = (args.model or "default").replace("/", "_")
    output_dir = args.output_dir or os.path.join(
        "/network/scratch/a/ankur.sikarwar/WORLD_MODEL_PROJECT/data/mira/annotations",
        model_name
    )
    args.output_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)

    video_files = sorted(Path(args.video_dir).glob("*.mp4"))

    # Sample a diverse subset if requested
    if args.max_videos and args.max_videos < len(video_files):
        if args.sample_seed is not None:
            import random
            rng = random.Random(args.sample_seed)
            video_files = rng.sample(video_files, args.max_videos)
            video_files = sorted(video_files)
        else:
            # Uniform stride — spreads picks evenly across the full sorted list
            stride = len(video_files) / args.max_videos
            video_files = [video_files[int(i * stride)] for i in range(args.max_videos)]

    # SLURM sharding (applied after sampling)
    if args.task_id is not None:
        video_files = [v for i, v in enumerate(video_files)
                       if i % args.n_tasks == args.task_id]

    print(f"Backend : {args.backend}  model={args.model or 'default'}")
    print(f"Videos  : {len(video_files)}")
    print(f"Output  : {output_dir}")

    backend = build_backend(args)
    annotator = VideoAnnotator(backend)

    # W&B setup
    wb_logger = None
    if args.wandb_project:
        config = vars(args).copy()
        config.pop("api_key", None)   # don't log secrets
        wb_logger = WandbLogger(
            project=args.wandb_project,
            run_name=args.wandb_run_name,
            flush_every=args.wandb_log_every,
            config=config,
        )

    for video_path in video_files:
        out_path = Path(args.output_dir) / (video_path.stem + ".json")
        if out_path.exists():
            continue  # resume support

        try:
            if args.gemini_native and args.backend == "gemini":
                result = annotator.annotate_gemini_native(
                    str(video_path),
                    n_segments=args.n_segments,
                    clip_frames=args.clip_frames,
                    segment_frames=args.segment_frames,
                    frame_sample_fps=args.frame_sample_fps,
                    max_frame_pairs=args.max_frame_pairs,
                )
            else:
                result = annotator.annotate(
                    str(video_path),
                    levels=args.levels,
                    n_segments=args.n_segments,
                    clip_frames=args.clip_frames,
                    segment_frames=args.segment_frames,
                    segment_sample_fps=args.segment_sample_fps,
                    max_segments=args.max_segments,
                    frame_sample_fps=args.frame_sample_fps,
                    max_frame_pairs=args.max_frame_pairs,
                )

            # Log to W&B immediately after each video
            if wb_logger:
                wb_logger.log_annotation(result, str(video_path))
                wb_logger.flush()

            # Save JSON (without raw frame arrays)
            with open(out_path, "w") as f:
                json.dump(_strip_internal(result), f, indent=2)

            parts = []
            if "clip" in result:
                parts.append("clip")
            if "segments" in result:
                parts.append(f"{len(result['segments'])} segments")
            if "frames" in result:
                parts.append(f"{len(result['frames'])} pairs")
            print(f"OK  {video_path.name}  ({', '.join(parts)})")

        except Exception as e:
            print(f"ERR {video_path.name}: {e}")
            traceback.print_exc()

    if wb_logger:
        wb_logger.finish()


if __name__ == "__main__":
    main()
