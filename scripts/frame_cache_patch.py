"""Monkey-patch lerobot's decode_video_frames to read precached .npy memmaps.

Pair with precache_videos.py. Once videos are pre-decoded, this replaces the
torchcodec decode in the training __getitem__ path with a np.memmap slice — no
ffmpeg, no fork-inherited handles, no per-decode leak → the DataLoader-worker
SEGV is gone, and reads are ~0.005s (vs torchcodec's 0.05-0.21s).

Correctness: lerobot's own torchcodec path computes
    frame_idx = round(ts * average_fps)
(video_utils.py:283). We use the IDENTICAL formula with the average_fps stored
in each cache sidecar, so the frame returned is bit-identical to what training
would have decoded (verified by precache_videos._validate).

Cache lookup key: the patch maps an mp4 video_path to its cache basename
<video_key>__chunk-{C:03d}__file-{F:03d} by parsing the path, so it works for
any v3.0 dataset whose videos were precached into <root>/frame_cache.

Usage (in launcher, after the get_safe_version patch):
    import frame_cache_patch; frame_cache_patch.apply("<dataset_root>/frame_cache")
"""
import json
import re
from pathlib import Path

import numpy as np
import torch


def apply(cache_dir: str):
    import lerobot.datasets.dataset_reader as dr

    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        raise SystemExit(f"[frame_cache_patch] cache_dir not found: {cache_dir}\n"
                         f"  run precache_videos.py first.")

    # Lazily-populated per-process handle cache: name -> (memmap, fps, num_frames).
    # np.memmap is fork-safe (mmap of a file), so workers can share/reinherit
    # safely — unlike torchcodec decoder objects.
    _handles = {}

    def _path_to_name(video_path) -> str:
        s = str(video_path)
        m = re.search(r"videos/(.+?)/chunk-(\d+)/file-(\d+)\.mp4$", s)
        if not m:
            raise KeyError(f"cannot parse video_key/chunk/file from {s}")
        video_key, chunk_idx, file_idx = m.group(1), int(m.group(2)), int(m.group(3))
        return f"{video_key}__chunk-{chunk_idx:03d}__file-{file_idx:03d}"

    def _get(video_path):
        name = _path_to_name(video_path)
        h = _handles.get(name)
        if h is None:
            meta = json.loads((cache_dir / f"{name}.json").read_text())
            mm = np.load(cache_dir / f"{name}.npy", mmap_mode="r")  # (N,C,H,W) uint8
            h = (mm, float(meta["average_fps"]), int(meta["num_frames"]))
            _handles[name] = h
        return h

    def _decode_from_cache(video_path, timestamps, tolerance_s,
                           backend=None, return_uint8=False):
        mm, fps, n = _get(video_path)
        idx = [min(max(round(ts * fps), 0), n - 1) for ts in timestamps]
        frames = np.ascontiguousarray(mm[idx])           # (T,C,H,W) uint8
        t = torch.from_numpy(frames)
        if return_uint8:
            return t
        return (t.to(torch.float32) / 255.0)

    dr.decode_video_frames = _decode_from_cache
    print(f"[frame_cache_patch] decode_video_frames -> .npy cache at {cache_dir}", flush=True)
