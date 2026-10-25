"""Columnar day-frame store on Arrow IPC (M43).

Day frames (the 100 ms ``EventFrame``-layout arrays produced by ``reconstruction.l2.frames``) are written as one
Arrow IPC (Feather v2, uncompressed) file per dataset with one column per array (multi-dimensional arrays are
flattened with their shape in the schema metadata). Reading is zero-copy and memory-mapped: ``open_frames`` returns
numpy views over the mapped file, so a study can slice one execution window of a 864,000-bin day without loading
the day into memory. Non-array entries (counters) are kept as JSON in the metadata.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SCHEMA = "cleolob-v1-frames-1"


def write_frames(frames: dict, path: Path, *, meta: dict | None = None) -> dict:
    import pyarrow as pa
    import pyarrow.ipc as ipc
    arrays = {k: np.ascontiguousarray(v) for k, v in frames.items() if isinstance(v, np.ndarray)}
    extra = {k: v for k, v in frames.items() if not isinstance(v, np.ndarray)}
    shapes = {k: list(v.shape) for k, v in arrays.items()}
    dtypes = {k: v.dtype.str for k, v in arrays.items()}
    n = max(int(np.prod(s)) for s in shapes.values())
    cols, names = [], []
    for k, v in arrays.items():
        flat = v.reshape(-1)
        if v.dtype == np.bool_:
            flat = flat.astype(np.uint8)
        padded = np.zeros(n, dtype=flat.dtype)
        padded[: len(flat)] = flat
        cols.append(pa.array(padded))
        names.append(k)
    md = {"schema": SCHEMA, "shapes": shapes, "dtypes": dtypes, "extra": extra, "meta": meta or {}}
    table = pa.table(cols, names=names).replace_schema_metadata({"cleolob": json.dumps(md, sort_keys=True)})
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    with pa.OSFile(str(tmp), "wb") as sink, ipc.new_file(sink, table.schema) as writer:
        writer.write_table(table)
    tmp.replace(path)
    return {"columns": len(names), "rows": n, "bytes": path.stat().st_size}


class Frames:
    """Memory-mapped view; ``frames[name]`` is a numpy view with the original shape and dtype."""

    def __init__(self, path: Path) -> None:
        import pyarrow as pa
        import pyarrow.ipc as ipc
        self._source = pa.memory_map(str(path), "r")
        self._table = ipc.open_file(self._source).read_all()
        self.meta = json.loads(self._table.schema.metadata[b"cleolob"])
        if self.meta["schema"] != SCHEMA:
            raise ValueError(f"{path}: not a v1 frame file")

    def __getitem__(self, name: str) -> np.ndarray:
        shape = self.meta["shapes"][name]
        dt = np.dtype(self.meta["dtypes"][name])
        col = self._table.column(name).chunk(0)
        flat = np.frombuffer(col.buffers()[1], dtype=np.uint8 if dt == np.bool_ else dt)[: int(np.prod(shape))]
        arr = flat.reshape(shape)
        return arr.view(np.bool_) if dt == np.bool_ else arr

    def keys(self) -> list[str]:
        return list(self.meta["shapes"])

    @property
    def extra(self) -> dict:
        return self.meta["extra"]

    def window(self, start_bin: int, stop_bin: int) -> dict:
        """Bin slice of every per-bin array (views; mid has one more entry)."""
        out = {}
        for k in self.keys():
            a = self[k]
            if k == "mid":
                out[k] = a[start_bin:stop_bin + 1]
            elif a.ndim >= 1 and len(a) == len(self["valid"]):
                out[k] = a[start_bin:stop_bin]
        return out


def open_frames(path: Path) -> Frames:
    return Frames(Path(path))


__all__ = ["write_frames", "open_frames", "Frames", "SCHEMA"]
