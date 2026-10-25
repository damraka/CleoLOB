"""Namespace isolation (M3): which dataset roles may be read in which namespace, and the guarded life cycle.

Dataset roles and the namespaces that may read them:

=============  =====================================================================================
role           readable in
=============  =====================================================================================
consumed       ``dev``, ``explore`` (data used by v0.1-v0.9: engineering and exploration only, never evidence)
development    ``dev``
explore        ``dev``, ``explore``
calibrate      ``explore``, ``calibrate``
confirm        ``confirm`` (opened by a sealed design, after promotion)
replicate      ``replicate`` (opened after ``confirm`` is closed)
vault          ``vault`` (opened once, after ``confirm`` and ``replicate`` are closed; never reopened)
=============  =====================================================================================

Guarded namespaces (``GUARDED``) can only be opened by a sealed design and, once closed, never reopen.
"""
from __future__ import annotations

NAMESPACES = ("dev", "explore", "calibrate", "confirm", "replicate", "vault")
GUARDED = frozenset({"confirm", "replicate", "vault"})
ROLES = ("consumed", "development", "explore", "calibrate", "confirm", "replicate", "vault")
READABLE = {
    "consumed": frozenset({"dev", "explore"}),
    "development": frozenset({"dev"}),
    "explore": frozenset({"dev", "explore"}),
    "calibrate": frozenset({"explore", "calibrate"}),
    "confirm": frozenset({"confirm"}),
    "replicate": frozenset({"replicate"}),
    "vault": frozenset({"vault"}),
}
REQUIRES_CLOSED = {"confirm": (), "replicate": ("confirm",), "vault": ("confirm", "replicate")}
NAMESPACE_IDS = {name: i for i, name in enumerate(NAMESPACES)}


def _err(msg: str):
    from .ledger import LedgerError
    return LedgerError(msg)


def check_access(state, dataset: str, role: str | None, namespace: str, design: str | None, use: str | None) -> None:
    if namespace not in NAMESPACES:
        raise _err(f"unknown namespace {namespace}")
    if role is None:
        if use == "development" and namespace == "dev":
            return
        raise _err(f"{dataset}: no registered role")
    if namespace not in READABLE[role]:
        raise _err(f"{dataset} (role {role}) may not be read in namespace {namespace}")
    if namespace in GUARDED:
        if state.open_namespaces.get(namespace) is None:
            raise _err(f"namespace {namespace} is not open")
        if design is None or design not in state.designs:
            raise _err(f"guarded access to {dataset} needs the sealed design that governs it")
        listed = state.designs[design].get("datasets")
        if listed is not None and dataset not in listed:
            raise _err(f"{dataset} is not listed by sealed design {design}")


def check_open(state, namespace: str, design: str) -> None:
    if namespace not in GUARDED:
        raise _err(f"only guarded namespaces are opened ({sorted(GUARDED)})")
    if design not in state.designs:
        raise _err(f"design {design} is not sealed")
    if namespace in state.open_namespaces:
        raise _err(f"namespace {namespace} is already open")
    if namespace in state.closed_namespaces:
        raise _err(f"namespace {namespace} was closed and never reopens")
    if state.promotion is None:
        raise _err("guarded namespaces open only after the promotion decision")
    for prior in REQUIRES_CLOSED[namespace]:
        if prior not in state.closed_namespaces:
            raise _err(f"{namespace} opens only after {prior} is closed")


__all__ = ["NAMESPACES", "GUARDED", "ROLES", "READABLE", "NAMESPACE_IDS", "check_access", "check_open"]
