# players/player_1/weight_policy.py
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Default location for the oracle file (can override via env or arg)
DEFAULT_ORACLE_PATH = os.getenv(
    "WEIGHTS_ORACLE_PATH",
    "players/player_1/data/weights_oracle_index.json",
)

WeightTuple = Tuple[float, float, float, float, float]  # (coh, imp, pref, nonmon, fresh)


@dataclass(frozen=True)
class ScenarioKey:
    P: int  # number_of_players
    T: int  # conversation_length
    M: int  # memory_bank size
    S: int  # subject count (use len(snapshot.preferences))

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.P, self.T, self.M, self.S)


def _clamp_norm(w: Iterable[float],
                lo: float = 0.05,
                hi: float = 0.60) -> WeightTuple:
    """Clamp each component to [lo, hi], then renormalize to sum=1."""
    ws = [max(lo, min(hi, float(x))) for x in w]
    s = sum(ws)
    if s <= 0:
        # fallback to uniform if all got clamped to zeros somehow
        return (0.2, 0.2, 0.2, 0.2, 0.2)
    ws = [x / s for x in ws]
    # Ensure exact 5-tuple
    return (ws[0], ws[1], ws[2], ws[3], ws[4])  # type: ignore[return-value]


def _heuristic_weights(P: int, T: int, M: int, S: int) -> WeightTuple:
    """
    Smooth, context-driven fallback that only relies on (P,T,M,S).
    Intuition:
      - Longer T -> more coherence & some nonmon; reduce freshness.
      - More players P -> more freshness & a bit more nonmon; trim coherence & pref slightly.
      - Larger memory M -> modest boost to coherence/importance; trim freshness/nonmon slightly.
      - Larger S -> don't let preference explode in large subject spaces.
    """
    # Normalize length factor to [0,1] over a reasonable range
    T_min, T_max = 8.0, 32.0
    L = max(0.0, min(1.0, (T - T_min) / (T_max - T_min)))

    # Base trend by T
    w_coh = 0.20 + 0.20 * L       # 0.20 → 0.40
    w_imp = 0.30 - 0.05 * L       # 0.30 → 0.25
    w_pref = 0.22 - 0.05 * L      # 0.22 → 0.17
    w_nonm = 0.08 + 0.07 * L      # 0.08 → 0.15
    w_fre = 0.20 - 0.17 * L       # 0.20 → 0.03

    # Player-count factor in [0,1] over 3..8 players
    P_min, P_max = 3.0, 8.0
    F = max(0.0, min(1.0, (P - P_min) / (P_max - P_min)))

    # More players -> increase freshness/nonmon, reduce coherence/pref
    w_fre += 0.12 * F
    w_nonm += 0.04 * F
    w_coh -= 0.10 * F
    w_pref -= 0.06 * F

    # Memory factor (more memory -> more coherence/importance)
    M_min, M_max = 4.0, 20.0
    MU = max(0.0, min(1.0, (M - M_min) / (M_max - M_min)))
    w_coh += 0.08 * MU
    w_imp += 0.04 * MU
    w_fre -= 0.06 * MU
    w_nonm -= 0.03 * MU

    # Subject-space proxy to temper pref in huge S
    # (larger S -> slightly reduce pref; smaller S -> allow a bit more)
    S_min, S_max = 4.0, 12.0
    SG = max(0.0, min(1.0, (S - S_min) / (S_max - S_min)))
    w_pref -= 0.05 * SG

    return _clamp_norm((w_coh, w_imp, w_pref, w_nonm, w_fre))


def _load_oracle(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_index(oracle: Dict[str, Any]) -> List[Dict[str, Any]]:
    idx = oracle.get("index", [])
    # Ensure minimal schema
    good = []
    for row in idx:
        if all(k in row for k in ("P", "T", "M", "S", "best")) and "weights" in row["best"]:
            good.append(row)
    return good


def _zscore_stats(rows: List[Dict[str, Any]]) -> Tuple[Tuple[float, float, float, float],
                                                       Tuple[float, float, float, float]]:
    """Return per-dimension (mean, std) for (P,T,M,S)."""
    if not rows:
        return (0, 0, 0, 0), (1, 1, 1, 1)
    Ps = [float(r["P"]) for r in rows]
    Ts = [float(r["T"]) for r in rows]
    Ms = [float(r["M"]) for r in rows]
    Ss = [float(r["S"]) for r in rows]

    def mean_std(xs: List[float]) -> Tuple[float, float]:
        if not xs:
            return (0.0, 1.0)
        m = sum(xs) / len(xs)
        v = sum((x - m) ** 2 for x in xs) / max(1, (len(xs) - 1))
        return (m, math.sqrt(v) if v > 0 else 1.0)

    mu = (mean_std(Ps)[0], mean_std(Ts)[0], mean_std(Ms)[0], mean_std(Ss)[0])
    sd = (mean_std(Ps)[1], mean_std(Ts)[1], mean_std(Ms)[1], mean_std(Ss)[1])
    return mu, sd


def _zdist(a: ScenarioKey, b: ScenarioKey,
           mu: Tuple[float, float, float, float],
           sd: Tuple[float, float, float, float]) -> float:
    ap, at, am, as_ = map(float, a.as_tuple())
    bp, bt, bm, bs = map(float, b.as_tuple())
    zp = (ap - mu[0]) / sd[0]
    zt = (at - mu[1]) / sd[1]
    zm = (am - mu[2]) / sd[2]
    zs = (as_ - mu[3]) / sd[3]
    zbp = (bp - mu[0]) / sd[0]
    zbt = (bt - mu[1]) / sd[1]
    zbm = (bm - mu[2]) / sd[2]
    zbs = (bs - mu[3]) / sd[3]
    return math.sqrt((zp - zbp) ** 2 + (zt - zbt) ** 2 + (zm - zbm) ** 2 + (zs - zbs) ** 2)


def _nearest_neighbors(rows: List[Dict[str, Any]], key: ScenarioKey, k: int = 3) -> List[Dict[str, Any]]:
    if not rows:
        return []
    mu, sd = _zscore_stats(rows)
    rows_scored = [
        (row, _zdist(key, ScenarioKey(row["P"], row["T"], row["M"], row["S"]), mu, sd))
        for row in rows
    ]
    rows_scored.sort(key=lambda t: t[1])
    return [r for r, _ in rows_scored[:max(1, k)]]


def _blend_weights(ws: List[WeightTuple], dists: List[float]) -> WeightTuple:
    """
    Distance-weighted average (inverse distance). If any distance is 0, return that weight directly.
    """
    for w, d in zip(ws, dists):
        if d <= 1e-9:
            return w
    inv = [1.0 / max(d, 1e-9) for d in dists]
    s = sum(inv)
    coeffs = [x / s for x in inv]
    blended = [0.0] * 5
    for c, w in zip(coeffs, ws):
        for i in range(5):
            blended[i] += c * w[i]
    return _clamp_norm(blended)


def compute_initial_weights(ctx, snapshot,
                            *,
                            oracle_path: Optional[str] = None,
                            alpha: float = 0.7,
                            nn_k: int = 3) -> WeightTuple:
    """
    Main entry point called by Player1.__init__.
    Tries oracle lookup (exact or nearest) at players/player_1/data/..., else falls back to heuristic.
    """
    P = int(getattr(ctx, "number_of_players", 0))
    T = int(getattr(ctx, "conversation_length", 0))
    M = int(len(getattr(snapshot, "memory_bank", []) or []))
    S = int(len(getattr(snapshot, "preferences", []) or []))
    key = ScenarioKey(P, T, M, S)

    # 1) Try oracle JSON
    path = oracle_path or DEFAULT_ORACLE_PATH
    oracle = _load_oracle(path)
    rows = _parse_index(oracle)
    if rows:
        # Exact match first
        for r in rows:
            if (r["P"], r["T"], r["M"], r["S"]) == key.as_tuple():
                bw = r["best"]["weights"]
                return _clamp_norm((bw[0], bw[1], bw[2], bw[3], bw[4]))

        # Nearest-neighbor fallback (blend top-k)
        mu, sd = _zscore_stats(rows)
        nn = _nearest_neighbors(rows, key, k=max(1, nn_k))
        dists = [
            _zdist(key, ScenarioKey(r["P"], r["T"], r["M"], r["S"]), mu, sd)
            for r in nn
        ]
        wlist = [tuple(r["best"]["weights"]) for r in nn]  # type: ignore[assignment]
        return _blend_weights(wlist, dists)

    # 2) No oracle → heuristic
    return _heuristic_weights(P, T, M, S)
