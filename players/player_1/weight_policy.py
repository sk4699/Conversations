# players/player_1/weight_policy.py
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

# Runtime default (matches your Player1 call)
DEFAULT_ORACLE_PATH = "players/player_1/data/weights_oracle_index.json"

# ---------------------------
# Utilities
# ---------------------------
def _clamp_nonneg(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        return 0.0
    return 0.0 if x < 0 or math.isnan(x) else x

def _clamp01(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        return 0.0
    if math.isnan(x):
        return 0.0
    if x < 0:
        return 0.0
    if x > 1:
        return 1.0
    return x

def _clamp_norm(ws: Iterable[float]) -> Tuple[float, ...]:
    w = [_clamp_nonneg(x) for x in ws]
    s = sum(w)
    if s <= 0:
        # safe default triplet
        if len(w) == 3:
            return (0.40, 0.35, 0.25)
        # generic equal partition
        return tuple(1.0 / max(1, len(w)) for _ in w)
    return tuple(x / s for x in w)

def _safe_mix_ab(a: float, b: float) -> Tuple[float, float]:
    a = _clamp01(a)
    b = _clamp01(b)
    s = a + b
    if s <= 0:
        return (0.65, 0.35)
    return (a / s, b / s)

def _heuristic_triplet(P: int, T: int, M: int, S: int) -> Tuple[float, float, float]:
    """
    Heuristic defaults for (w_coh, w_imp, w_pref) ONLY.
    Slightly more coherence for short games, more importance for long games.
    Preference steady.
    """
    if T <= 12:
        w = (0.45, 0.30, 0.25)
    elif T <= 24:
        w = (0.40, 0.34, 0.26)
    else:
        w = (0.36, 0.38, 0.26)
    return _clamp_norm(w)

def _heuristic_mix_ab(P: int, T: int, M: int, S: int) -> Tuple[float, float]:
    # Favor feature-weighted score; normalize to a+b=1
    return _safe_mix_ab(0.90, 0.10)

@dataclass
class Scenario:
    P: int; T: int; M: int; S: int

def _scenario_key(P: int, T: int, M: int, S: int) -> Tuple[int, int, int, int]:
    return (int(P), int(T), int(M), int(S))

def _load_json(path: str) -> Union[dict, list, None]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _collect_stats(entries: List[Dict[str, Any]]) -> Dict[str, Tuple[float, float]]:
    import statistics as st
    Ps = [int(e["P"]) for e in entries] or [1]
    Ts = [int(e["T"]) for e in entries] or [1]
    Ms = [int(e["M"]) for e in entries] or [1]
    Ss = [int(e["S"]) for e in entries] or [1]
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

# ---------- Oracle readers (support multiple shapes) ----------

def _read_entries_from_per_scenario_payload(data: dict) -> List[Dict[str, Any]]:
    """
    New generator format:
    {
      "meta": {...},
      "index": [
        {
          "P": ..., "T": ..., "M": ..., "S": ...,
          "best": { "weights": [w_coh,w_imp,w_pref], "mix_ab": [a,b], ... },
          "alts": [...]
        }, ...
      ]
    }
    """
    idx = data.get("index")
    return list(idx) if isinstance(idx, list) else []

def _read_entries_from_flat_list(data: list) -> List[Dict[str, Any]]:
    """
    Back-compat: list of rows like
    [{"P":..,"T":..,"M":..,"S":..,"w_coh":..,"w_imp":..,"w_pref":..,"a":..}, ...]
    """
    rows: List[Dict[str, Any]] = []
    for rec in data:
        if not isinstance(rec, dict):
            continue
        if all(k in rec for k in ("P", "T", "M", "S")) and \
           all(k in rec for k in ("w_coh", "w_imp", "w_pref", "a")):
            rows.append({
                "P": rec["P"], "T": rec["T"], "M": rec["M"], "S": rec["S"],
                "best": {"weights": [rec["w_coh"], rec["w_imp"], rec["w_pref"]],
                         "mix_ab": [rec.get("a", 0.65), 1.0 - float(rec.get("a", 0.65))]}
            })
    return rows

def _normalize_entry_weights_and_mix(e: Dict[str, Any]) -> Tuple[Tuple[float,float,float], Tuple[float,float]]:
    """
    Returns ((w_coh,w_imp,w_pref), (a,b)) from an entry regardless of legacy/new shape.
    - If best.weights has 5 items, keep first 3.
    - If best.mix_ab missing/invalid, use heuristic.
    """
    best = e.get("best", {}) if isinstance(e.get("best"), dict) else {}
    w_raw = best.get("weights", [])
    # Handle legacy 5-vector by truncating to first 3
    if isinstance(w_raw, list) and len(w_raw) >= 3:
        w3 = _clamp_norm(tuple(float(x) for x in w_raw[:3]))
    else:
        # try flat fields fallback (rare)
        w3 = _clamp_norm((
            float(e.get("w_coh", 0.4)),
            float(e.get("w_imp", 0.35)),
            float(e.get("w_pref", 0.25)),
        ))

    mix = best.get("mix_ab", [])
    if not (isinstance(mix, list) and len(mix) == 2):
        # allow legacy 'a' field at top-level
        a = float(e.get("a", 0.65))
        mix_ab = _safe_mix_ab(a, 1.0 - a)
    else:
        a, b = mix
        mix_ab = _safe_mix_ab(float(a), float(b))

    return w3, mix_ab

# Public Method

def compute_initial_weights(
    ctx,
    snapshot,
    *,
    oracle_path: Optional[str] = None,
    alpha: float = 0.7,
    nn_k: int = 3,
) -> Tuple[float, float, float, float, float]:
    """
    Returns (w_coh, w_imp, w_pref, a_weighted, b_raw).

    Loads the new per-scenario payload format (payload.index[].best.weights triplet + best.mix_ab [a,b])
    and falls back to:
      - legacy flat rows with (w_coh,w_imp,w_pref,a)
      - or heuristics when nothing matches.
    """

    # Try to read runtime scenario from ctx/snapshot (keep loose to avoid import coupling)
    P = int(getattr(ctx, "number_of_players", getattr(ctx, "num_players", 0)) or 0)
    T = int(getattr(ctx, "conversation_length", getattr(ctx, "T", 0)) or 0)
    # Memory: prefer explicit M on ctx; otherwise size of snapshot memory bank
    M = int(getattr(ctx, "M", 0) or len(getattr(snapshot, "memory_bank", []) or []))
    # Subjects: prefer ctx.subjects or snapshot.preferences length
    S = int(getattr(ctx, "subjects", 0) or len(getattr(snapshot, "preferences", []) or []))
    here = Scenario(P, T, M, S)

    path = oracle_path or os.getenv("WEIGHTS_ORACLE_PATH", DEFAULT_ORACLE_PATH)
    raw = _load_json(path)

    # Collect candidate entries in a normalized (per-scenario) shape
    entries: List[Dict[str, Any]] = []
    if isinstance(raw, dict):
        # New payload (with "index") or robust winners; prefer per-scenario
        entries = _read_entries_from_per_scenario_payload(raw)
        if not entries:
            # Could be robust payload or something else; try to coerce
            # Robust payload doesn't directly map to scenario keys, so skip.
            pass
    elif isinstance(raw, list):
        # Flat list of rows from older generator
        entries = _read_entries_from_flat_list(raw)

    # 1) Exact match
    for e in entries:
        key = (int(e.get("P", -1)), int(e.get("T", -1)), int(e.get("M", -1)), int(e.get("S", -1)))
        if key == _scenario_key(P, T, M, S):
            w3, (a, b) = _normalize_entry_weights_and_mix(e)
            print("Weights Found: Exact match from oracle.")
            return (*w3, a, b)

    # 2) Nearest neighbors (z-score over P,T,M,S)
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

        W_vecs: List[Tuple[float,float,float]] = []
        M_vecs: List[Tuple[float,float]] = []
        for _, e in top:
            w3, (a, b) = _normalize_entry_weights_and_mix(e)
            W_vecs.append(w3)
            M_vecs.append((a, b))

        W = _blend(W_vecs, ws_norm)
        A, B = _blend(M_vecs, ws_norm)
        A, B = _safe_mix_ab(A, B)
        print(f"Weights Found: Nearest neighbor blend from oracle (k={len(top)}).")
        return (*_clamp_norm(W), A, B)

    # 3) Nothing on disk → heuristics
    W = _heuristic_triplet(P, T, M, S)
    a, b = _heuristic_mix_ab(P, T, M, S)
    print("Weights Found: Heuristic defaults (no oracle).")
    return (*W, a, b)
