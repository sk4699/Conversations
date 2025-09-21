# players/player_1/weight_policy.py
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Runtime default (matches your Player1 call)
DEFAULT_ORACLE_PATH = "players/player_1/data/weights_oracle_index.json"

# ---------------------------
# Utilities
# ---------------------------
def _clamp_norm(ws: Iterable[float]) -> Tuple[float, ...]:
    w = [max(0.0, min(1.0, float(x))) for x in ws]
    s = sum(w)
    if s <= 0:
        return (0.3, 0.3, 0.2, 0.1, 0.1)
    return tuple(x / s for x in w)

def _heuristic_weights(P: int, T: int, M: int, S: int) -> Tuple[float, ...]:
    print("Using Heuristics")
    # Simple, safe fallback mirroring your earlier logic
    w_coh, w_imp, w_pref, w_nonmon, w_fresh = 0.40, 0.30, 0.20, 0.10, 0.00
    if T <= 12:
        w_coh, w_imp, w_pref, w_nonmon, w_fresh = 0.30, 0.45, 0.20, 0.05, 0.00
    elif T >= 31:
        w_coh, w_imp, w_pref, w_nonmon, w_fresh = 0.50, 0.20, 0.15, 0.15, 0.00
    if P <= 3:
        w_coh += 0.05; w_imp -= 0.05
    elif P >= 6:
        w_coh -= 0.10; w_imp += 0.10; w_pref = max(w_pref - 0.05, 0.10)
    if M <= 8:
        w_coh += 0.05; w_imp -= 0.05
    elif M >= 16:
        w_imp += 0.05; w_coh -= 0.05
    w = _clamp_norm((w_coh, w_imp, w_pref, w_nonmon, w_fresh))
    if T <= 12 and w[2] > 0.18:
        w = _clamp_norm((w[0], w[1], 0.18, w[3], w[4]))
    elif T >= 31 and w[2] > 0.15:
        w = _clamp_norm((w[0], w[1], 0.15, w[3], w[4]))
    return w

def _heuristic_mix_ab(P: int, T: int, M: int, S: int) -> Tuple[float, float]:
    # Favor feature-weighted score by default; normalize to a+b=1
    a, b = 0.90, 0.10
    s = a + b
    return (a / s, b / s)

@dataclass
class Scenario:
    P: int; T: int; M: int; S: int

def _scenario_key(P: int, T: int, M: int, S: int) -> Tuple[int, int, int, int]:
    return (int(P), int(T), int(M), int(S))

def _load_index(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _collect_stats(entries: List[Dict[str, Any]]) -> Dict[str, Tuple[float, float]]:
    import statistics as st
    Ps = [e["P"] for e in entries] or [1]
    Ts = [e["T"] for e in entries] or [1]
    Ms = [e["M"] for e in entries] or [1]
    Ss = [e["S"] for e in entries] or [1]
    def ms(x): return (st.fmean(x), (st.pstdev(x) if len(x) > 1 else 1.0))
    return {"P": ms(Ps), "T": ms(Ts), "M": ms(Ms), "S": ms(Ss)}

def _zscore_distance(a: Scenario, b: Scenario, stats: Dict[str, Tuple[float, float]]) -> float:
    tot = 0.0
    for dim, va, vb in (("P", a.P, b.P), ("T", a.T, b.T), ("M", a.M, b.M), ("S", a.S, b.S)):
        mu, sd = stats.get(dim, (0.0, 1.0))
        d = 0.0 if sd <= 0 else abs((va - mu) / sd - (vb - mu) / sd)
        tot += d * d
    return math.sqrt(tot)

def _blend(vs: List[Tuple[float, ...]], ws: List[float]) -> Tuple[float, ...]:
    out = [0.0] * len(vs[0])
    for vec, w in zip(vs, ws):
        for i, x in enumerate(vec):
            out[i] += w * x
    return tuple(out)


# Public Method

def compute_initial_weights(
    ctx,
    snapshot,
    *,
    oracle_path: Optional[str] = None,
    alpha: float = 0.7,
    nn_k: int = 3,
) -> Tuple[float, float, float, float, float, float, float]:
    """
    Returns (w_coh, w_imp, w_pref, w_nonmon, w_fresh, a_weighted, b_raw),
    reading 'weights' and 'mix_ab' from the index file; falls back to heuristics.
    """
    P = int(getattr(ctx, "number_of_players", 0))
    T = int(getattr(ctx, "conversation_length", 0))
    M = int(len(getattr(snapshot, "memory_bank", []) or []))
    S = int(len(getattr(snapshot, "preferences", []) or []))
    here = Scenario(P, T, M, S)

    path = oracle_path or os.getenv("WEIGHTS_ORACLE_PATH", DEFAULT_ORACLE_PATH)
    data = _load_index(path)
    entries: List[Dict[str, Any]] = list(data.get("index", []))

    # Exact match
    for e in entries:
        if (e.get("P"), e.get("T"), e.get("M"), e.get("S")) == _scenario_key(P, T, M, S):
            w = tuple(e.get("best", {}).get("weights", [])) or _heuristic_weights(P, T, M, S)
            w = _clamp_norm(w)
            mix = tuple(e.get("best", {}).get("mix_ab", [])) or _heuristic_mix_ab(P, T, M, S)
            if len(mix) != 2:
                mix = _heuristic_mix_ab(P, T, M, S)
            a, b = mix
            s = max(1e-9, a + b)
            print("Found Weights")
            return (*w, a / s, b / s)

    # Nearest neighbors (z-score space)
    if entries:
        stats = _collect_stats(entries)
        cand = []
        for e in entries:
            there = Scenario(int(e["P"]), int(e["T"]), int(e["M"]), int(e["S"]))
            d = _zscore_distance(here, there, stats)
            cand.append((d, e))
        cand.sort(key=lambda t: t[0])
        top = cand[: max(1, int(nn_k))]

        inv = [1.0 / (d + 1e-9) for d, _ in top]
        s = sum(inv)
        ws_norm = [x / s for x in inv]

        W_vecs, M_vecs = [], []
        for _, e in top:
            w = tuple(e.get("best", {}).get("weights", [])) or _heuristic_weights(P, T, M, S)
            W_vecs.append(_clamp_norm(w))
            mix = tuple(e.get("best", {}).get("mix_ab", [])) or _heuristic_mix_ab(P, T, M, S)
            if len(mix) != 2:
                mix = _heuristic_mix_ab(P, T, M, S)
            a, b = mix
            t = max(1e-9, a + b)
            M_vecs.append((a / t, b / t))

        W = _blend(W_vecs, ws_norm)
        A, B = _blend(M_vecs, ws_norm)
        s2 = max(1e-9, A + B)
        print("Computing K Nearest Neighbors")
        return (*_clamp_norm(W), A / s2, B / s2)

    # Nothing on disk → heuristics
    W = _heuristic_weights(P, T, M, S)
    a, b = _heuristic_mix_ab(P, T, M, S)
    s3 = max(1e-9, a + b)
    return (*W, a / s3, b / s3)
