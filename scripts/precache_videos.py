"""Pre-decode a LeRobot v3.0 dataset's videos to .npy memmaps (NCHW uint8).

Why: lerobot's torchcodec decode path SEGVs DataLoader workers on long runs
(fork-inherited ffmpeg handles + a per-decode resource leak that crashes after
~2k decodes/worker). num_workers=0 dodges it but serializes decode (data_s
0.21s >> updt_s 0.032s, ~6x slower). Pre-decoding to flat uint8 arrays removes
torchcodec from the training loop entirely: reads become np.memmap slices
(fork-safe, ~0.005s), so the SEGV is gone AND training is fast.

Layout (v3.0): videos/<video_key>/chunk-{C:03d}/file-{F:03d}.mp4 holds many
episodes concatenated at a constant fps. We cache one .npy per mp4 file, named
<video_key>__chunk-{C:03d}__file-{F:03d}.npy, plus a sidecar .json with
{average_fps, num_frames, shape} so the read patch maps timestamp->frame with
the EXACT same formula lerobot uses: idx = round(ts * average_fps).

Usage:
    python precache_videos.py --dataset_root <lerobot_dir> [--cache_dir DIR]
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

BATCH = 2000  # frames decoded per torchcodec call (≈0.4 GB RAM at 256² × 3ch)


def _video_files(ds_root: Path):
    info = json.loads((ds_root / "meta" / "info.json").read_text())
    tmpl = info["video_path"]  # videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4
    vids_dir = ds_root / "videos"
    out = []
    for key_dir in sorted(vids_dir.iterdir()):
        if not key_dir.is_dir():
            continue
        video_key = key_dir.name  # observation.images.<cam>
        for mp4 in sorted(key_dir.rglob("*.mp4")):
            m = re.search(r"chunk-(\d+)/file-(\d+)\.mp4$", str(mp4))
            chunk_idx, file_idx = int(m.group(1)), int(m.group(2))
            cache_name = f"{video_key}__chunk-{chunk_idx:03d}__file-{file_idx:03d}"
            out.append((mp4, cache_name))
    return out


def _decode_file(mp4: Path, cache_dir: Path, cache_name: str):
    from torchcodec.decoders import VideoDecoder

    npy_path = cache_dir / f"{cache_name}.npy"
    json_path = cache_dir / f"{cache_name}.json"
    if npy_path.exists() and json_path.exists():
        meta = json.loads(json_path.read_text())
        print(f"  skip {cache_name} (cached, {meta['num_frames']} frames)", flush=True)
        return meta

    dec = VideoDecoder(str(mp4), seek_mode="approximate")  # default NCHW output
    n = dec.metadata.num_frames
    fps = float(dec.metadata.average_fps)
    # Probe one frame for (C, H, W)
    sample = dec.get_frames_at(indices=[0]).data  # (1, C, H, W) uint8
    c, h, w = int(sample.shape[1]), int(sample.shape[2]), int(sample.shape[3])
    print(f"  {cache_name}: {n} frames, fps={fps:.4f}, frame=({c},{h},{w}) "
          f"-> {n*c*h*w/1e9:.1f} GB", flush=True)

    mm = np.lib.format.open_memmap(npy_path, mode="w+", dtype=np.uint8, shape=(n, c, h, w))
    for start in range(0, n, BATCH):
        end = min(start + BATCH, n)
        idx = list(range(start, end))
        frames = dec.get_frames_at(indices=idx).data  # (b, C, H, W) uint8
        mm[start:end] = frames.numpy()
        if start % (BATCH * 10) == 0:
            print(f"    {cache_name}: {end}/{n}", flush=True)
    mm.flush()
    del mm
    meta = {"average_fps": fps, "num_frames": n, "shape": [c, h, w]}
    json_path.write_text(json.dumps(meta))
    return meta


def _validate(mp4: Path, cache_dir: Path, cache_name: str, meta: dict, n_samples=5):
    """Confirm cache[round(ts*fps)] is bit-identical to a fresh torchcodec decode
    at the same timestamp — guards against any fps / index-mapping drift."""
    from torchcodec.decoders import VideoDecoder
    dec = VideoDecoder(str(mp4), seek_mode="approximate")
    fps = meta["average_fps"]
    n = meta["num_frames"]
    mm = np.load(cache_dir / f"{cache_name}.npy", mmap_mode="r")
    # sample timestamps spread across the file
    test_indices = np.linspace(0, n - 1, n_samples).astype(int)
    for fi in test_indices:
        ts = fi / fps
        ref = dec.get_frames_played_at([ts]).data[0].numpy()  # (C,H,W) uint8 nearest to ts
        cached = mm[round(ts * fps)]
        if not np.array_equal(ref, cached):
            diff = np.abs(ref.astype(int) - cached.astype(int)).mean()
            raise SystemExit(f"VALIDATION FAILED {cache_name} @ ts={ts:.3f}: mean|diff|={diff:.2f}")
    print(f"  validate {cache_name}: {n_samples} timestamps bit-identical ✓", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--cache_dir", default=None,
                    help="Default: <dataset_root>/../frame_cache")
    args = ap.parse_args()

    ds_root = Path(args.dataset_root).resolve()
    cache_dir = Path(args.cache_dir) if args.cache_dir else ds_root / "frame_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[precache] dataset: {ds_root}\n[precache] cache:   {cache_dir}", flush=True)

    files = _video_files(ds_root)
    print(f"[precache] {len(files)} video files to cache", flush=True)
    for mp4, cache_name in files:
        meta = _decode_file(mp4, cache_dir, cache_name)
        _validate(mp4, cache_dir, cache_name, meta)
    print(f"[precache] done. cache at {cache_dir}", flush=True)


if __name__ == "__main__":
    main()
