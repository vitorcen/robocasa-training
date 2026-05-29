"""Create a small v2.1 subset of the mimicgen OpenCabinet dataset so it can be
converted to v3.0 + precached + trained, matching the human pipeline.

The mimicgen data ships as codebase v2.0 (per-episode parquet + mp4 files) and
is MISSING meta/episodes_stats.jsonl — the only thing v2.0 lacks vs v2.1. The
v21->v30 converter requires that file. We:

  1. Hardlink the first N episodes' data parquet + 3-cam mp4 into a subset dir
     (hardlinks = instant, zero extra disk, same filesystem).
  2. Build meta:
       - copy modality/tasks/embodiment/stats.json as-is
       - episodes.jsonl  -> first N lines
       - info.json       -> totals recomputed for N, codebase_version bumped v2.1
       - episodes_stats.jsonl -> per-episode = a COPY of dataset stats.json with
         count set to that episode's frame length.
     Why copy dataset stats per-episode instead of computing real ones: nothing
     in our pipeline uses per-episode stats (training normalizes with dataset-
     level stats.json; the v3.0 convert just aggregates episodes_stats back to
     ~dataset stats — weighted-averaging identical means reproduces them). This
     avoids decoding any video for stats. Per-episode stat accuracy is irrelevant
     here; correctness of the dataset-level normalization (preserved) is what
     matters for ACT.

Usage:
    python subset_mimicgen.py --src <mimicgen_v20_lerobot_dir> --dst <out_dir> --n 500
"""
import argparse
import json
import os
import shutil
from pathlib import Path


def _hardlink(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    os.link(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="mimicgen v2.0 lerobot dir")
    ap.add_argument("--dst", required=True, help="output subset dir")
    ap.add_argument("--n", type=int, default=500)
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    info = json.loads((src / "meta" / "info.json").read_text())
    chunks_size = info.get("chunks_size", 1000)
    data_tmpl = info["data_path"]
    video_tmpl = info["video_path"]
    video_keys = [p.name for p in sorted((src / "videos" / "chunk-000").iterdir()) if p.is_dir()]
    assert args.n <= chunks_size, "subset spanning multiple chunks not handled; keep n<=chunks_size"

    # episodes.jsonl: first N
    eps = [json.loads(l) for l in (src / "meta" / "episodes.jsonl").read_text().splitlines()]
    eps = eps[: args.n]
    total_frames = sum(e["length"] for e in eps)

    print(f"[subset] {args.n} episodes, {total_frames} frames, video_keys={video_keys}", flush=True)

    # 1) hardlink data + videos
    for i in range(args.n):
        ck = i // chunks_size
        _hardlink(src / data_tmpl.format(episode_chunk=ck, episode_index=i),
                  dst / data_tmpl.format(episode_chunk=ck, episode_index=i))
        for vk in video_keys:
            rel = video_tmpl.format(episode_chunk=ck, video_key=vk, episode_index=i)
            _hardlink(src / rel, dst / rel)
        if (i + 1) % 100 == 0:
            print(f"[subset] linked {i+1}/{args.n}", flush=True)

    # 2) meta
    meta_dst = dst / "meta"
    meta_dst.mkdir(parents=True, exist_ok=True)
    for fn in ["modality.json", "tasks.jsonl", "embodiment.json", "stats.json"]:
        sp = src / "meta" / fn
        if sp.exists():
            shutil.copy(sp, meta_dst / fn)

    # FIX 2: mimicgen never computed image stats, so stats.json lacks the
    # observation.images.* keys. With use_imagenet_stats=True (lerobot default),
    # factory.make_dataset does stats[img_key][...] = imagenet_stats and KeyErrors
    # on the missing key. Add the 3 image keys with imagenet values (which
    # use_imagenet_stats would overwrite anyway — only the keys need to exist).
    stats_path = meta_dst / "stats.json"
    dataset_stats = json.loads(stats_path.read_text())
    IMAGENET_MEAN = [[[0.485]], [[0.456]], [[0.406]]]
    IMAGENET_STD = [[[0.229]], [[0.224]], [[0.225]]]
    ZERO3 = [[[0.0]], [[0.0]], [[0.0]]]
    ONE3 = [[[1.0]], [[1.0]], [[1.0]]]
    for vk in video_keys:  # video_keys are already "observation.images.<cam>"
        if vk not in dataset_stats:
            dataset_stats[vk] = {"min": ZERO3, "max": ONE3, "mean": IMAGENET_MEAN,
                                 "std": IMAGENET_STD, "count": [total_frames]}
    stats_path.write_text(json.dumps(dataset_stats))

    (meta_dst / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in eps))

    info_out = dict(info)
    info_out["codebase_version"] = "v2.1"
    info_out["total_episodes"] = args.n
    info_out["total_frames"] = total_frames
    info_out["total_videos"] = args.n * len(video_keys)
    info_out["total_chunks"] = 1
    info_out["splits"] = {"train": f"0:{args.n}"}
    # FIX 1: mimicgen v2.0 parquet stores state/action as list-of-double but
    # info.json tags them dtype "object", which pyarrow rejects at load
    # (get_hf_features_from_features -> "Neither object nor object_ ..."). The
    # data is fine (list<double>); just correct the declared dtype.
    for fk, fv in info_out.get("features", {}).items():
        if fv.get("dtype") == "object":
            fv["dtype"] = "float64"
    (meta_dst / "info.json").write_text(json.dumps(info_out, indent=4))

    # episodes_stats.jsonl: per-episode copy of the IMAGE-AUGMENTED dataset stats
    # (count = ep length). Must include image keys — the v3.0 convert aggregates
    # episodes_stats into the v3.0 meta/stats.json, so missing image keys here
    # would reproduce the use_imagenet_stats KeyError downstream.
    with open(meta_dst / "episodes_stats.jsonl", "w") as f:
        for e in eps:
            length = e["length"]
            ep_stats = {}
            for feat, s in dataset_stats.items():
                fs = {k: v for k, v in s.items() if k != "count"}
                fs["count"] = [length]
                ep_stats[feat] = fs
            f.write(json.dumps({"episode_index": e["episode_index"], "stats": ep_stats}) + "\n")

    print(f"[subset] done -> {dst}", flush=True)


if __name__ == "__main__":
    main()
