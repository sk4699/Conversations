# players/player_1/weight_policy.py
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

# Runtime default (matches your Player1 call)
DEFAULT_ORACLE_PATH = "players/player_1/data/weights_oracle_index.json"

# ---------------------------
# Utilities
# ---------------------------
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

def _safe_mix_ab(a: float, b: float) -> Tuple[float, float]:
    a = _clamp01(a)
    b = _clamp01(b)
    s = a + b
    if s <= 0:
        return (0.5, 0.5)
    return (a / s, b / s)

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

# ---------- Oracle readers ----------
def _read_entries_from_per_scenario_payload(data: dict) -> List[Dict[str, Any]]:
    """
    New format:
    {
      "meta": {...},
      "index": [
        {
          "P": ..., "T": ..., "M": ..., "S": ...,
          "best": { "a": .., "b": .., "threshold": .., "score": .. },
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
    [{"P":..,"T":..,"M":..,"S":..,"a":..,"b":..,"threshold":..}, ...]
    """
    rows: List[Dict[str, Any]] = []
    for rec in data:
        if not isinstance(rec, dict):
            continue
        if all(k in rec for k in ("P", "T", "M", "S", "a", "b", "threshold")):
            rows.append({
                "P": rec["P"], "T": rec["T"], "M": rec["M"], "S": rec["S"],
                "best": {"a": rec["a"], "b": rec["b"], "threshold": rec["threshold"]}
            })
    return rows

def _normalize_entry_ab_thresh(e: Dict[str, Any]) -> Tuple[float, float, float]:
    """
    Extracts (a, b, threshold) from entry.
    Ensures a+b=1 normalization safety.
    """
    best = e.get("best", {}) if isinstance(e.get("best"), dict) else {}
    a = float(best.get("a", 0.5))
    b = float(best.get("b", 1.0 - a))
    a, b = _safe_mix_ab(a, b)
    threshold = _clamp01(best.get("threshold", 0.5))
    return a, b, threshold

# Public Method
def compute_initial_weights(
    ctx,
    snapshot,
    *,
    oracle_path: Optional[str] = None,
    nn_k: int = 3,
) -> Tuple[float, float, float]:
    """
    Returns (a, b, threshold).

    Loads the per-scenario oracle (payload.index[].best.{a,b,threshold})
    and falls back to:
      - legacy flat rows
      - or heuristics when nothing matches.
    """

    # Extract runtime scenario
    P = int(getattr(ctx, "number_of_players", getattr(ctx, "num_players", 0)) or 0)
    T = int(getattr(ctx, "conversation_length", getattr(ctx, "T", 0)) or 0)
    M = int(getattr(ctx, "M", 0) or len(getattr(snapshot, "memory_bank", []) or []))
    S = int(getattr(ctx, "subjects", 0) or len(getattr(snapshot, "preferences", []) or []))
    here = Scenario(P, T, M, S)

    path = oracle_path or os.getenv("WEIGHTS_ORACLE_PATH", DEFAULT_ORACLE_PATH)
    raw = _load_json(path)

    # Collect candidate entries
    entries: List[Dict[str, Any]] = []
    if isinstance(raw, dict):
        entries = _read_entries_from_per_scenario_payload(raw)
    elif isinstance(raw, list):
        entries = _read_entries_from_flat_list(raw)

    # 1) Exact match
    for e in entries:
        key = (int(e.get("P", -1)), int(e.get("T", -1)), int(e.get("M", -1)), int(e.get("S", -1)))
        if key == _scenario_key(P, T, M, S):
            a, b, thresh = _normalize_entry_ab_thresh(e)
            print("AB/Threshold Found: Exact match from oracle.")
            return a, b, thresh

    # 2) Nearest neighbor fallback
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

        A_vals, B_vals, T_vals = [], [], []
        for _, e in top:
            a, b, thresh = _normalize_entry_ab_thresh(e)
            A_vals.append((a,)); B_vals.append((b,)); T_vals.append((thresh,))

        def blend(vs): return sum(v[0] * w for v, w in zip(vs, ws_norm))

        A = blend(A_vals); B = blend(B_vals); Tt = blend(T_vals)
        A, B = _safe_mix_ab(A, B)
        print(f"AB/Threshold Found: Nearest neighbor blend (k={len(top)}).")
        return A, B, _clamp01(Tt)

    # 3) Heuristic fallback
    print("AB/Threshold Found: Heuristic defaults (no oracle).")
    return (0.5, 0.5, 0.5)
