"""Monkey-patch lerobot's torchcodec decoder cache to be fork/PID-safe.

Root cause this fixes
---------------------
lerobot.datasets.video_utils holds a module-global `_default_decoder_cache`
(a VideoDecoderCache) that maps video_path -> (torchcodec.VideoDecoder, fsspec
file handle). When a PyTorch DataLoader spawns workers via fork(), each child
inherits a COPY of this dict whose decoder + file-handle objects reference the
*parent's* open ffmpeg/file state. Reusing those handles in the child is
undefined behaviour and SIGSEGVs the worker — which is exactly the crash we hit
reproducibly around step ~24k (= just past the first epoch boundary at 23003
steps, when persistent_workers=False respawns workers and they re-fork).

The dataset_reader docstring even warns about it:
  "When using data workers (num_workers>0), do not call _query_videos in the
   main process ... It will result in a Segmentation Fault."

Why num_workers=0 'works' but is slow
-------------------------------------
With num_workers=0 everything decodes in the main process (one stable cache,
no fork) — correct but single-threaded, so video decode (data_s≈0.21s) dwarfs
GPU compute (updt_s≈0.032s): ~6x slower than it should be.

The fix
-------
Make the cache notice it's been forked: stamp the owning PID, and on first use
in a different PID, drop the inherited entries (WITHOUT closing their handles —
closing a fork-shared fd from the child would corrupt the parent) and rebuild
fresh decoders local to that worker. This lets num_workers>0 + torchcodec run
safely, restoring parallel CPU decode.

Usage (from a launcher, before constructing the dataset/trainer):
    import pid_safe_decoder_patch; pid_safe_decoder_patch.apply()
"""
import os
import threading


def apply():
    import lerobot.datasets.video_utils as vu

    BaseCache = vu.VideoDecoderCache

    class PidSafeDecoderCache(BaseCache):
        def __init__(self):
            super().__init__()
            self._owner_pid = os.getpid()

        def _reset_if_forked(self):
            pid = os.getpid()
            if pid != self._owner_pid:
                # We're in a freshly-forked DataLoader worker. The inherited
                # (decoder, file_handle) tuples point at the parent's ffmpeg
                # state. Abandon them by reference (do NOT close — that would
                # corrupt the parent's still-open fds). A new lock too, in case
                # the inherited one was held at fork time.
                self._cache = {}
                self._lock = threading.Lock()
                self._owner_pid = pid

        def get_decoder(self, video_path):
            self._reset_if_forked()
            return super().get_decoder(video_path)

    vu._default_decoder_cache = PidSafeDecoderCache()
    # decode_video_frames_torchcodec reads the module-global at call time
    # (default arg decoder_cache=None -> _default_decoder_cache), so reassigning
    # the module attribute is sufficient; no need to patch the function itself.
    print("[pid_safe_decoder_patch] installed PID-aware torchcodec decoder cache", flush=True)
