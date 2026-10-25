"""Run unchanged v0.9 study functions on ledgered v1 day caches (exact replication of the v0.9 estimators).

The v0.9 study task functions obtain data only through two loaders: ``lob.v09.studies.common.load_cache`` and
``lob.v09.data.store.prepare`` (both bound to the v0.9 ledger). ``using_v1_caches(mapping)`` replaces exactly
those two callables, for the duration of the block, with a lookup into ``mapping`` ({dataset id: v1 cache path}):
every other line of v0.9 code - feature extraction, model fitting, selection rules, scoring - runs unchanged, so a
v1 replication uses the same estimator as v0.9, not a re-implementation.

The mapping only contains datasets whose access the v1 parent process has already ledgered; an unmapped dataset
raises, so the adapter can never read data that the v1 ledger has not recorded. v1 caches carry the v0.9 cache
keys (``lob.v1.data.daycache.load``); the v0.9 quality-decision fields are not used by any v0.9 model code.
"""
from __future__ import annotations

from contextlib import contextmanager
import importlib
from pathlib import Path


class UnmappedDataset(KeyError):
    pass


@contextmanager
def using_v1_caches(mapping: dict[str, str]):
    from ...data import daycache as DC
    common = importlib.import_module("lob.v09.studies.common")
    store = importlib.import_module("lob.v09.data.store")
    saved = (common.load_cache, store.prepare)

    def _load(dataset_id: str):
        if dataset_id not in mapping:
            raise UnmappedDataset(f"{dataset_id} has no ledgered v1 access in this run")
        return DC.load(Path(mapping[dataset_id]))

    def load_cache(dataset_id, root=None, design=None, purpose=None):
        return _load(dataset_id)

    def prepare(dataset_id, *, use=None, design=None, purpose=None, root=None):
        return _load(dataset_id), {"v1_cache": mapping[dataset_id]}

    common.load_cache, store.prepare = load_cache, prepare
    try:
        yield
    finally:
        common.load_cache, store.prepare = saved


def call(fn: str, mapping: dict[str, str], /, **kwargs):
    """Worker entry: run ``module:function`` from lob.v09 under the adapter (importable by worker processes)."""
    from ...runtime import cap_threads
    cap_threads()
    mod, _, name = fn.partition(":")
    with using_v1_caches(mapping):
        return getattr(importlib.import_module(mod), name)(**kwargs)


__all__ = ["using_v1_caches", "call", "UnmappedDataset"]
