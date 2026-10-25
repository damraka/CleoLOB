"""GPU-accelerated batch simulation (M46) with a CPU fallback of identical numerical semantics (M47).

``simulate_torch`` implements the ``lob.v1.engines.accelerated`` kernel vectorized over paths: every event step is
applied to all live paths at once with masked tensor operations, and the data-dependent loops (sweeps, touch shifts,
cancel-level search) iterate until no path needs another pass. All state is int64 and all randomness comes from
the same pre-drawn integer inputs, so on any device the outputs must equal the numba/Python kernel bit for bit -
which the compatibility tests check on CPU always and on CUDA when a CUDA device and a CUDA build of torch exist.
torch is optional; without it ``available()`` is False and callers use the CPU kernel.
"""
from __future__ import annotations

import numpy as np

from .accelerated import SCALE, KernelParams, init_depth


def available() -> dict:
    try:
        import torch
    except ImportError:
        return {"torch": False, "cuda": False}
    cuda = bool(torch.cuda.is_available())
    return {"torch": True, "cuda": cuda, "device": torch.cuda.get_device_name(0) if cuda else None,
            "version": torch.__version__}


def default_device() -> str:
    a = available()
    return "cuda" if a.get("cuda") else "cpu"


def simulate_torch(params: KernelParams, inputs: dict, *, device: str | None = None) -> dict:
    import torch
    dev = torch.device(device or default_device())
    i64 = torch.int64
    marks = torch.as_tensor(inputs["marks"], dtype=i64, device=dev)
    r = torch.as_tensor(inputs["r"], dtype=i64, device=dev)
    refill = torch.as_tensor(inputs["refill"], dtype=i64, device=dev)
    P, n = marks.shape
    L, W = params.levels, params.width
    npool = refill.shape[1]
    add_cdf, cancel_cdf = (torch.as_tensor(x, dtype=i64, device=dev) for x in params.tables())
    improve = torch.as_tensor(params.improve(), dtype=i64, device=dev)
    ib, ia = init_depth(params)
    bid = torch.zeros((P, W), dtype=i64, device=dev)
    ask = torch.zeros((P, W), dtype=i64, device=dev)
    bb = torch.full((P,), W // 2 - 1, dtype=i64, device=dev)
    ba = torch.full((P,), W // 2, dtype=i64, device=dev)
    ks = torch.arange(L, device=dev)
    bid[:, (W // 2 - 1) - ks] = torch.as_tensor(ib, dtype=i64, device=dev)
    ask[:, (W // 2) + ks] = torch.as_tensor(ia, dtype=i64, device=dev)
    rc = torch.zeros(P, dtype=i64, device=dev)
    ok = torch.ones(P, dtype=torch.bool, device=dev)
    rows = torch.arange(P, device=dev)
    out_bid = torch.full((P, n), -1, dtype=i64, device=dev)
    out_ask = torch.full((P, n), -1, dtype=i64, device=dev)
    out_depth = torch.zeros((P, n, 2), dtype=i64, device=dev)
    mm, lm, cf = params.market_mean, params.limit_mean, params.cancel_frac_bp

    def lookup(table, u):
        return (u[:, None] >= table[None, :-1]).sum(1)

    def shift_ask(mask, walked=None):
        """Move depleted ask touches outward until non-empty; refill entering levels; flag overflows."""
        while True:
            need = mask & ok & (ask[rows, ba] <= 0)
            if not bool(need.any()):
                return
            ba[need] += 1
            over = need & ((ba + L - 1 >= W) | (rc >= npool))
            ok[over] = False
            go = need & ~over
            idx = rows[go]
            ask[idx, ba[go] + L - 1] = refill[idx, rc[go]]
            rc[go] += 1
            if walked is not None:
                walked[go] += 1

    def shift_bid(mask, walked=None):
        while True:
            need = mask & ok & (bid[rows, bb] <= 0)
            if not bool(need.any()):
                return
            bb[need] -= 1
            over = need & ((bb - L + 1 < 0) | (rc >= npool))
            ok[over] = False
            go = need & ~over
            idx = rows[go]
            bid[idx, bb[go] - L + 1] = refill[idx, rc[go]]
            rc[go] += 1
            if walked is not None:
                walked[go] += 1

    for t in range(n):
        m = marks[:, t]
        r0, r1, r2 = r[:, t, 0], r[:, t, 1], r[:, t, 2]
        live = ok.clone()
        # ------------------------------------------------------------------ market orders
        for side_mark in (0, 1):
            mask = live & (m == side_mark)
            if not bool(mask.any()):
                continue
            q = torch.where(mask, 1 + r1 % (2 * mm - 1), torch.zeros_like(r1))
            walked = torch.zeros(P, dtype=i64, device=dev)
            while True:
                act = mask & ok & (q > 0) & (walked < params.max_walk)
                if not bool(act.any()):
                    break
                idx = rows[act]
                if side_mark == 0:
                    cur = ask[idx, ba[act]]
                    take = torch.minimum(q[act], cur)
                    ask[idx, ba[act]] = cur - take
                    q[act] -= take
                    shift_ask(act, walked)
                else:
                    cur = bid[idx, bb[act]]
                    take = torch.minimum(q[act], cur)
                    bid[idx, bb[act]] = cur - take
                    q[act] -= take
                    shift_bid(act, walked)
        # ------------------------------------------------------------------ additions
        for side_mark in (2, 3):
            mask = live & (m == side_mark)
            if not bool(mask.any()):
                continue
            size = 1 + r1 % (2 * lm - 1)
            spread = ba - bb
            cls = torch.clamp(torch.clamp(spread, max=4) - 2, min=0)
            imp = mask & (spread > 1) & ((r2 % SCALE) < improve[cls])
            ticks = torch.where(imp, 1 + r0 % torch.clamp(spread - 1, min=1), torch.zeros_like(r0))
            if bool(imp.any()):
                for j in range(int(ticks.max().item())):
                    z = imp & (ticks > j)
                    idx = rows[z]
                    if side_mark == 2:
                        bid[idx, bb[z] - L + 1 + j] = 0
                    else:
                        ask[idx, ba[z] + L - 1 - j] = 0
                idx = rows[imp]
                if side_mark == 2:
                    bb[imp] += ticks[imp]
                    bid[idx, bb[imp]] = size[imp]
                else:
                    ba[imp] -= ticks[imp]
                    ask[idx, ba[imp]] = size[imp]
            plain = mask & ~imp
            if bool(plain.any()):
                lev = lookup(add_cdf, r0)
                idx = rows[plain]
                if side_mark == 2:
                    pos = bb[plain] - lev[plain]
                    bid[idx, pos] += size[plain]
                else:
                    pos = ba[plain] + lev[plain]
                    ask[idx, pos] += size[plain]
        # ------------------------------------------------------------------ cancellations
        for side_mark in (4, 5):
            mask = live & (m == side_mark)
            if not bool(mask.any()):
                continue
            lev = lookup(cancel_cdf, r0)
            if side_mark == 4:
                vis = bid.gather(1, (bb[:, None] - ks[None, :]).clamp(min=0))
            else:
                vis = ask.gather(1, (ba[:, None] + ks[None, :]).clamp(max=W - 1))
            nonempty = vis > 0
            up = nonempty & (ks[None, :] >= lev[:, None])
            down = nonempty & (ks[None, :] <= lev[:, None])
            big = torch.full_like(vis, L)
            j_up = torch.where(up, ks[None, :].expand_as(vis), big).min(1).values
            j_down = torch.where(down, ks[None, :].expand_as(vis), torch.full_like(vis, -1)).max(1).values
            j = torch.where(j_up < L, j_up, j_down)
            go = mask & (j >= 0)
            if bool(go.any()):
                idx = rows[go]
                qv = vis[go, j[go]]
                c = torch.clamp(qv * (r1[go] % (2 * cf + 1)) // SCALE, min=1)
                c = torch.minimum(c, qv)
                if side_mark == 4:
                    bid[idx, bb[go] - j[go]] -= c
                    shift_bid(go)
                else:
                    ask[idx, ba[go] + j[go]] -= c
                    shift_ask(go)
        w = ok
        out_bid[w, t] = bb[w]
        out_ask[w, t] = ba[w]
        out_depth[w, t, 0] = bid[rows[w], bb[w]]
        out_depth[w, t, 1] = ask[rows[w], ba[w]]
    return {"bid": out_bid.cpu().numpy(), "ask": out_ask.cpu().numpy(), "depth": out_depth.cpu().numpy(),
            "ok": ok.cpu().numpy(), "refills": rc.cpu().numpy(), "backend": f"torch:{dev.type}"}


def equal(a: dict, b: dict) -> bool:
    return all(np.array_equal(a[k], b[k]) for k in ("bid", "ask", "depth", "ok", "refills"))


__all__ = ["available", "default_device", "simulate_torch", "equal"]
