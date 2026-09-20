"""C5 dump timing shim (2026-09-20 methodology redesign).

Activated only when RG_C5_TIMING=1 (set by run_c56_ab.sh for the guard arm).
Patches KvCacheReader on import and emits per-layer JSONL events:

  d2h_layer   {rank_tag, req_id, layer, d2h_ms, bytes}   per-layer D2H copy
  d2h_summary {rank_tag, req_id, layers, d2h_ms, wall_ms, bytes}  per request
  save        {rank_tag, n, save_ms, bytes, files}       per write_snapshots call

All patching is defensive: any failure degrades to the unpatched product path.
"""
import builtins
import json
import os
import threading
import time

_ENABLED = os.environ.get("RG_C5_TIMING") == "1"
_OUT = os.environ.get("RG_C5_TIMING_OUT") or "/tmp/rg_c5_timing_phase.jsonl"
_LOCK = threading.Lock()
_PATCHED = False


def _emit(ev):
    try:
        ev["t"] = round(time.time(), 3)
        with _LOCK:
            with open(_OUT, "a") as f:
                f.write(json.dumps(ev, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _tensor_bytes(obj, depth=0):
    try:
        import torch
        if depth > 5:
            return 0
        if torch.is_tensor(obj):
            return 0 if bool(obj.is_meta) else int(obj.nbytes)
        if isinstance(obj, dict):
            return sum(_tensor_bytes(v, depth + 1) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return sum(_tensor_bytes(v, depth + 1) for v in obj)
    except Exception:
        pass
    return 0


def _snap_meta(snap):
    tag = ""
    req = ""
    try:
        p = snap.payload
        tag = str(p.get("rank_tag") or "")
        req = str(p.get("req_id") or "")
    except Exception:
        pass
    return tag, req


def _patch(mod):
    global _PATCHED
    cls = getattr(mod, "KvCacheReader", None)
    if cls is None or getattr(cls, "_rg_c5_timed", False):
        return
    orig_iter = cls.iter_request_snapshots
    orig_write = cls.write_snapshots

    def iter_timed(self, *, req_id, block_ids, out_dir):
        gen = orig_iter(self, req_id=req_id, block_ids=block_ids, out_dir=out_dir)
        d2h_ms = 0.0
        layers = 0
        nbytes = 0
        rank_tag = ""
        t0_wall = time.perf_counter()
        while True:
            t0 = time.perf_counter()
            try:
                snap = next(gen)
            except StopIteration:
                break
            dt = (time.perf_counter() - t0) * 1000.0
            d2h_ms += dt
            layers += 1
            nb = _tensor_bytes(getattr(snap, "payload", None))
            nbytes += nb
            if not rank_tag:
                rank_tag, _ = _snap_meta(snap)
            _emit({"event": "d2h_layer", "rank_tag": rank_tag,
                   "req_id": str(req_id), "layer": layers,
                   "d2h_ms": round(dt, 3), "bytes": nb})
            yield snap
        wall_ms = (time.perf_counter() - t0_wall) * 1000.0
        _emit({"event": "d2h_summary", "rank_tag": rank_tag,
               "req_id": str(req_id), "layers": layers,
               "d2h_ms": round(d2h_ms, 3), "wall_ms": round(wall_ms, 3),
               "bytes": nbytes})

    def write_timed(snapshots):
        t0 = time.perf_counter()
        paths = orig_write(snapshots)
        dt = (time.perf_counter() - t0) * 1000.0
        nb = 0
        rank_tag = ""
        for s in snapshots:
            nb += _tensor_bytes(getattr(s, "payload", None))
            tag, _ = _snap_meta(s)
            if tag:
                rank_tag = tag
        _emit({"event": "save", "rank_tag": rank_tag, "n": len(snapshots),
               "save_ms": round(dt, 3), "bytes": nb, "files": len(paths)})
        return paths

    cls.iter_request_snapshots = iter_timed
    cls.write_snapshots = staticmethod(write_timed)
    cls._rg_c5_timed = True
    _PATCHED = True
    _emit({"event": "shim_patched", "pid": os.getpid()})


if _ENABLED:
    _orig_import = builtins.__import__

    def _rg_import(name, *args, **kwargs):
        mod = _orig_import(name, *args, **kwargs)
        if not _PATCHED and name.endswith("runtime_guard.kv_cache_reader"):
            try:
                import vllm_ascend.runtime_guard.kv_cache_reader as m
                _patch(m)
            except Exception as exc:
                _emit({"event": "shim_error", "pid": os.getpid(),
                       "error": repr(exc)})
        return mod

    builtins.__import__ = _rg_import
