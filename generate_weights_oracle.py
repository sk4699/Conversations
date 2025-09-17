# AI Generated Code
# Conversations/generate_weights_oracle.py
from __future__ import annotations

import argparse
import json
import math
import os
import random
from itertools import product
from typing import Any, Dict, List, Tuple

import pandas as pd

# Engine & players from your codebase
from core.engine import Engine
from players.player_1.player import Player1
from players.random_player import RandomPlayer
from players.random_pause_player import RandomPausePlayer

# Reuse the heuristic & clamps the player uses
from players.player_1.weight_policy import (
    DEFAULT_ORACLE_PATH,
    _clamp_norm,
    _heuristic_weights,
)

# ------------------------------------------------------------------
# README defaults (always included in sweeps; override via env vars)
# ------------------------------------------------------------------
DEFAULTS_T = int(os.getenv("DEFAULT_T", 10))   # conversation length
DEFAULTS_M = int(os.getenv("DEFAULT_M", 10))   # memory size
DEFAULTS_S = int(os.getenv("DEFAULT_S", 20))   # subjects


# -----------------------------
# Utilities / Sampling helpers
# -----------------------------
def _ensure_dirs(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _dirichlet_near(
    prior: Tuple[float, float, float, float, float], n: int, kappa: float
) -> List[Tuple[float, ...]]:
    """Sample n points near 'prior' on the simplex using Dirichlet(kappa*prior)."""
    try:
        import numpy as np
    except Exception:
        return []
    alpha = np.array(prior, dtype=float) * float(kappa)
    alpha = np.maximum(alpha, 1e-3)
    samples = np.random.default_rng().dirichlet(alpha, size=n)
    return [tuple(map(float, s)) for s in samples]


def _coordinate_bumps(
    prior: Tuple[float, float, float, float, float], delta: float
) -> List[Tuple[float, ...]]:
    """Add +/- delta to each coordinate with re-projection to the simplex."""
    res: List[Tuple[float, ...]] = []
    k = len(prior)
    for i in range(k):
        for sign in (+1.0, -1.0):
            w = list(prior)
            w[i] = w[i] + sign * delta
            # Pay for it by subtracting evenly from others
            dec = (sign * delta) / (k - 1) * (-1.0)
            for j in range(k):
                if j != i:
                    w[j] = w[j] + dec
            res.append(_clamp_norm(w))
    return res


def _balanced_corners() -> List[Tuple[float, ...]]:
    """A few sensible 'corner-ish' baselines that remain safe."""
    return [
        _clamp_norm((0.35, 0.30, 0.15, 0.10, 0.10)),  # coherence/importance heavy
        _clamp_norm((0.15, 0.25, 0.15, 0.10, 0.35)),  # freshness heavy
        _clamp_norm((0.20, 0.20, 0.30, 0.10, 0.20)),  # preference heavy
        _clamp_norm((0.20, 0.25, 0.15, 0.25, 0.15)),  # non-monotonousness bump
    ]


def _build_global_candidates(
    P: int,
    mid_T: int,
    mid_M: int,
    mid_S: int,
    seed: int,
    coord_delta: float,
    n_dirichlet: int,
    dirichlet_kappa: float,
) -> List[Tuple[float, ...]]:
    """
    Build ONE global candidate set (same for all scenarios) from a prior at mid (T,M,S).
    This ensures every candidate is evaluated in every swept scenario for fair aggregation.
    """
    random.seed(seed)
    prior = _heuristic_weights(P, mid_T, mid_M, mid_S)
    cands = [prior]
    cands += _coordinate_bumps(prior, coord_delta)
    cands += _balanced_corners()
    cands += _dirichlet_near(prior, n_dirichlet, dirichlet_kappa)
    # Deduplicate (tuples hashable)
    return list({tuple(round(x, 6) for x in w) for w in cands})


def _build_player_types(P: int, num_p1: int, rpause_frac: float) -> List[type]:
    """
    Build a list of player TYPES for Engine:
    - num_p1 copies of Player1 (clamped to [1, P])
    - remaining seats filled by Random / RandomPause with given fraction.
    """
    num_p1 = max(1, min(P, int(num_p1)))
    n_others = max(0, P - num_p1)
    n_rpause = int(round(rpause_frac * n_others))
    n_random = n_others - n_rpause
    return [Player1] * num_p1 + [RandomPlayer] * n_random + [RandomPausePlayer] * n_rpause


def _eval_once(
    P: int,
    T: int,
    M: int,
    S: int,
    weights: Tuple[float, ...],
    seed: int,
    num_p1: int,
    rpause_frac: float,
) -> Tuple[float, float]:
    """
    Run one game and return (player1_individual, shared_total).
    If multiple Player1s exist, we report the FIRST one's individual score.
    """
    players_types = _build_player_types(P, num_p1=num_p1, rpause_frac=rpause_frac)
    random.seed(seed)

    # Engine expects 'subjects' as INT (range(subjects)), so pass S directly.
    engine = Engine(
        players=players_types,
        player_count=P,
        subjects=S,
        memory_size=M,
        conversation_length=T,
        seed=seed,
    )

    # Inject weights into ALL Player1 instances
    w_coh, w_imp, w_pref, w_nonm, w_fre = weights
    for p in engine.players:
        if isinstance(p, Player1):
            p.w_coh, p.w_imp, p.w_pref, p.w_nonmon, p.w_fresh = (
                w_coh,
                w_imp,
                w_pref,
                w_nonm,
                w_fre,
            )

    output = engine.run(players_types)

    shared_total = float(output["scores"]["shared_score_breakdown"]["total"])

    # Gather all Player1 ids; take the first one's individual score as the metric
    p1_ids = [uid for uid, name in engine.player_names.items() if name == "Player1"]
    p1_individual = float("nan")
    for snap in output["scores"]["player_scores"]:
        if snap["id"] in p1_ids:
            p1_individual = float(snap["scores"]["individual"])
            break

    return (p1_individual, shared_total)


def _objective(ind: float, shared: float, alpha: float) -> float:
    if math.isnan(ind):
        return float("-inf")
    return alpha * ind + (1.0 - alpha) * shared


def _eval_candidate_on_scenario(
    P: int,
    T: int,
    M: int,
    S: int,
    weights: Tuple[float, ...],
    seeds: List[int],
    num_p1: int,
    rpause_frac: float,
    alpha: float,
) -> Dict[str, float]:
    vals: List[Tuple[float, float, float]] = []
    for sd in seeds:
        ind, sh = _eval_once(
            P, T, M, S, weights, sd, num_p1=num_p1, rpause_frac=rpause_frac
        )
        vals.append((ind, sh, _objective(ind, sh, alpha)))
    import statistics as st

    inds = [v[0] for v in vals]
    shrs = [v[1] for v in vals]
    objs = [v[2] for v in vals]
    return {
        "mean_individual": float(st.fmean(inds)),
        "std_individual": float(st.pstdev(inds)) if len(inds) > 1 else 0.0,
        "mean_shared": float(st.fmean(shrs)),
        "std_shared": float(st.pstdev(shrs)) if len(shrs) > 1 else 0.0,
        "mean_obj": float(st.fmean(objs)),
        "std_obj": float(st.pstdev(objs)) if len(objs) > 1 else 0.0,
        "n_reps": len(seeds),
    }


# -----------------------------
# CLI parsing
# -----------------------------
def _parse_int_list(vals: List[str]) -> List[int]:
    out: List[int] = []
    for v in vals:
        if "," in v:
            out.extend(int(x.strip()) for x in v.split(",") if x.strip())
        else:
            out.append(int(v))
    return out


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Generate robust or per-scenario weights. Robust mode: pass only player params; script sweeps T/M/S (README defaults always included)."
    )
    # MODE
    ap.add_argument(
        "--mode",
        choices=["robust", "per-scenario"],
        default="robust",
        help="robust: pick one weight per (P, num_p1, rpause) by aggregating over T/M/S. per-scenario: original behavior.",
    )

    # Minimal required input in robust mode
    ap.add_argument(
        "--players",
        "-P",
        required=True,
        type=int,
        help="Number of players for the game (single value) in robust mode. In per-scenario mode, use --players-grid.",
    )
    ap.add_argument(
        "--num-p1",
        type=str,
        default="1",
        help="How many Player1 instances per game. Integer or 'P' to fill all seats (clamped to [1, P]).",
    )
    ap.add_argument(
        "--p1-only",
        action="store_true",
        help="Shortcut: fill ALL seats with Player1 (equivalent to --num-p1=P).",
    )
    ap.add_argument(
        "--rpause-frac",
        type=float,
        default=0.5,
        help="Fraction of RandomPause among non-Player1 seats (0..1). Ignored if p1-only.",
    )

    # Robust-mode sweep grids (defaults; README defaults auto-included)
    ap.add_argument(
        "--lengths",
        "-T",
        nargs="+",
        type=str,
        default=["12", "16", "20", "24", "28", "32"],
        help="Conversation lengths to sweep (robust mode).",
    )
    ap.add_argument(
        "--mem-sizes",
        "-M",
        nargs="+",
        type=str,
        default=["8", "12", "16", "20"],
        help="Memory bank sizes to sweep (robust mode).",
    )
    ap.add_argument(
        "--subjects",
        "-S",
        nargs="+",
        type=str,
        default=["6", "8", "10", "12"],
        help="Subject counts to sweep (robust mode).",
    )

    # per-scenario (legacy) grid (only used if --mode per-scenario)
    ap.add_argument(
        "--players-grid",
        nargs="+",
        type=str,
        default=["4", "6"],
        help="(per-scenario mode) P grid. Ex: 4 6 8",
    )
    ap.add_argument(
        "--lengths-grid",
        nargs="+",
        type=str,
        default=["12", "20", "28"],
        help="(per-scenario mode) T grid.",
    )
    ap.add_argument(
        "--mem-grid",
        nargs="+",
        type=str,
        default=["8", "12", "16"],
        help="(per-scenario mode) M grid.",
    )
    ap.add_argument(
        "--subjects-grid",
        nargs="+",
        type=str,
        default=["6", "8", "10"],
        help="(per-scenario mode) S grid.",
    )

    # Objective
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Objective blend: alpha*P1Individual + (1-alpha)*Shared.",
    )

    # Candidate set tuning
    ap.add_argument(
        "--coord-delta",
        type=float,
        default=0.07,
        help="Coordinate bump delta for candidate generation.",
    )
    ap.add_argument(
        "--n-dirichlet",
        type=int,
        default=10,
        help="Number of Dirichlet samples around the prior.",
    )
    ap.add_argument(
        "--dirichlet-kappa",
        type=float,
        default=60.0,
        help="Dirichlet concentration around the prior (higher = tighter).",
    )

    # Two-stage evaluation
    ap.add_argument(
        "--stageA-reps", type=int, default=2, help="Reps per scenario for Stage A."
    )
    ap.add_argument(
        "--stageB-reps", type=int, default=4, help="Reps per scenario for Stage B."
    )
    ap.add_argument(
        "--top-k", type=int, default=6, help="Number of finalists for Stage B."
    )

    # Output
    ap.add_argument(
        "--output-json",
        default=DEFAULT_ORACLE_PATH,
        help="per-scenario mode: compact oracle index JSON (as before). Ignored in robust mode.",
    )
    ap.add_argument(
        "--output-table",
        default="players/player_1/data/weights_oracle_full.parquet",
        help="Wide table (Parquet/CSV) with evaluations (both modes).",
    )
    ap.add_argument(
        "--output-robust",
        default="players/player_1/data/weights_oracle_robust.json",
        help="Robust summary JSON (robust mode).",
    )

    # Seeds
    ap.add_argument("--base-seed", type=int, default=4242, help="Base RNG seed.")

    return ap


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = build_argparser()
    args = ap.parse_args()

    # Resolve core player composition
    P = int(args.players)
    p1_only = bool(args.p1_only)
    rpause_frac = 0.0 if p1_only else max(0.0, min(1.0, float(args.rpause_frac)))

    if p1_only:
        num_p1 = P
    else:
        raw_num_p1 = (args.num_p1 or "1").strip()
        if raw_num_p1.lower() == "p":
            num_p1 = P
        else:
            try:
                num_p1 = max(1, min(P, int(raw_num_p1)))
            except ValueError:
                num_p1 = 1

    random.seed(args.base_seed)

    # Ensure outputs path exists
    _ensure_dirs(args.output_table)
    _ensure_dirs(args.output_robust)
    _ensure_dirs(args.output_json)

    # ---------- ROBUST MODE ----------
    if args.mode == "robust":
        GRID_T = _parse_int_list(list(args.lengths))
        GRID_M = _parse_int_list(list(args.mem_sizes))
        GRID_S = _parse_int_list(list(args.subjects))

        # Always include README defaults
        GRID_T = sorted(set(GRID_T + [DEFAULTS_T]))
        GRID_M = sorted(set(GRID_M + [DEFAULTS_M]))
        GRID_S = sorted(set(GRID_S + [DEFAULTS_S]))

        scenarios = list(product([P], GRID_T, GRID_M, GRID_S))

        # Build a single global candidate set using mid values (ensures same set across scenarios)
        mid_T = sorted(GRID_T)[len(GRID_T) // 2]
        mid_M = sorted(GRID_M)[len(GRID_M) // 2]
        mid_S = sorted(GRID_S)[len(GRID_S) // 2]
        cand_seed = random.randrange(10**9)
        candidates = _build_global_candidates(
            P,
            mid_T,
            mid_M,
            mid_S,
            seed=cand_seed,
            coord_delta=float(args.coord_delta),
            n_dirichlet=int(args.n_dirichlet),
            dirichlet_kappa=float(args.dirichlet_kappa),
        )

        # Stage A: evaluate ALL candidates on ALL scenarios with small reps, aggregate
        all_rows: List[Dict[str, Any]] = []
        agg: Dict[Tuple[float, ...], Dict[str, Any]] = {}  # weight -> aggregation stats

        for (_, T, M, S) in scenarios:
            # CRN seeds per scenario
            seeds_A = [random.randrange(10**9) for _ in range(int(args.stageA_reps))]
            for w in candidates:
                stats = _eval_candidate_on_scenario(
                    P,
                    T,
                    M,
                    S,
                    w,
                    seeds_A,
                    num_p1=num_p1,
                    rpause_frac=rpause_frac,
                    alpha=float(args.alpha),
                )
                row = dict(stats)
                row.update(
                    {
                        "P": P,
                        "T": T,
                        "M": M,
                        "S": S,
                        "weights": list(w),
                        "stage": "A",
                        "num_p1": num_p1,
                        "rpause_frac": rpause_frac,
                    }
                )
                all_rows.append(row)

                key = tuple(w)
                if key not in agg:
                    agg[key] = {
                        "sum_mean_obj": 0.0,
                        "count": 0,
                        "min_mean_obj": float("inf"),
                    }
                agg[key]["sum_mean_obj"] += stats["mean_obj"]
                agg[key]["count"] += 1
                agg[key]["min_mean_obj"] = min(
                    agg[key]["min_mean_obj"], stats["mean_obj"]
                )

        # Rank candidates by average performance across scenarios
        avg_ranked = sorted(
            [(w, v["sum_mean_obj"] / max(1, v["count"])) for w, v in agg.items()],
            key=lambda t: t[1],
            reverse=True,
        )
        finalists = [w for w, _ in avg_ranked[: max(1, int(args.top_k))]]

        # Stage B: re-evaluate finalists on ALL scenarios with fresh CRN seeds
        final_stats: Dict[Tuple[float, ...], Dict[str, float]] = {
            w: {"sum_avg": 0.0, "min_avg": float("inf"), "nsc": 0} for w in finalists
        }
        for (_, T, M, S) in scenarios:
            seeds_B = [random.randrange(10**9) for _ in range(int(args.stageB_reps))]
            for w in finalists:
                statsB = _eval_candidate_on_scenario(
                    P,
                    T,
                    M,
                    S,
                    w,
                    seeds_B,
                    num_p1=num_p1,
                    rpause_frac=rpause_frac,
                    alpha=float(args.alpha),
                )
                row = dict(statsB)
                row.update(
                    {
                        "P": P,
                        "T": T,
                        "M": M,
                        "S": S,
                        "weights": list(w),
                        "stage": "B",
                        "num_p1": num_p1,
                        "rpause_frac": rpause_frac,
                    }
                )
                all_rows.append(row)

                final_stats[w]["sum_avg"] += statsB["mean_obj"]
                final_stats[w]["min_avg"] = min(
                    final_stats[w]["min_avg"], statsB["mean_obj"]
                )
                final_stats[w]["nsc"] += 1

        # Pick winners
        best_avg_w = max(
            final_stats.items(),
            key=lambda kv: (kv[1]["sum_avg"] / max(1, kv[1]["nsc"])),
        )[0]
        best_worst_w = max(final_stats.items(), key=lambda kv: kv[1]["min_avg"])[0]

        robust_payload = {
            "meta": {
                "mode": "robust",
                "alpha": float(args.alpha),
                "stageA_reps": int(args.stageA_reps),
                "stageB_reps": int(args.stageB_reps),
                "top_k": int(args.top_k),
                "seed": int(args.base_seed),
                "defaults_included": {
                    "T": DEFAULTS_T,
                    "M": DEFAULTS_M,
                    "S": DEFAULTS_S,
                },
                "notes": "Robust pick aggregates over T/M/S sweep using a single global candidate set.",
            },
            "sweep": {
                "P": P,
                "num_p1": num_p1,
                "rpause_frac": rpause_frac,
                "lengths": sorted(list(set(GRID_T))),
                "mem_sizes": sorted(list(set(GRID_M))),
                "subjects": sorted(list(set(GRID_S))),
            },
            "winners": {
                "best_avg": {
                    "weights": list(best_avg_w),
                    "avg_mean_obj": final_stats[best_avg_w]["sum_avg"]
                    / max(1, final_stats[best_avg_w]["nsc"]),
                    "worst_case_obj": final_stats[best_avg_w]["min_avg"],
                },
                "best_worst": {
                    "weights": list(best_worst_w),
                    "avg_mean_obj": final_stats[best_worst_w]["sum_avg"]
                    / max(1, final_stats[best_worst_w]["nsc"]),
                    "worst_case_obj": final_stats[best_worst_w]["min_avg"],
                },
                "finalists": [
                    {
                        "weights": list(w),
                        "avg_mean_obj": final_stats[w]["sum_avg"]
                        / max(1, final_stats[w]["nsc"]),
                        "worst_case_obj": final_stats[w]["min_avg"],
                    }
                    for w in finalists
                ],
            },
        }

        # Write robust JSON
        with open(args.output_robust, "w", encoding="utf-8") as f:
            json.dump(robust_payload, f, indent=2)

        # Wide table (Parquet → CSV fallback)
        try:
            df = pd.DataFrame(all_rows)
            df.to_parquet(args.output_table, index=False)
        except Exception:
            alt_csv = os.path.splitext(args.output_table)[0] + ".csv"
            pd.DataFrame(all_rows).to_csv(alt_csv, index=False)

        print(
            f"[oracle][robust] Best-Avg weights:  {robust_payload['winners']['best_avg']['weights']}"
        )
        print(
            f"[oracle][robust] Best-Worst weights:{robust_payload['winners']['best_worst']['weights']}"
        )
        print(f"[oracle] Wrote robust summary to {args.output_robust}")
        print(f"[oracle] Wrote table to {args.output_table} (or CSV fallback)")
        return

    # ---------- PER-SCENARIO MODE (legacy) ----------
    GRID_P = _parse_int_list(list(args.players_grid))
    GRID_T = _parse_int_list(list(args.lengths_grid))
    GRID_M = _parse_int_list(list(args.mem_grid))
    GRID_S = _parse_int_list(list(args.subjects_grid))

    # Always include README defaults
    GRID_T = sorted(set(GRID_T + [DEFAULTS_T]))
    GRID_M = sorted(set(GRID_M + [DEFAULTS_M]))
    GRID_S = sorted(set(GRID_S + [DEFAULTS_S]))

    scenarios = list(product(GRID_P, GRID_T, GRID_M, GRID_S))

    all_rows: List[Dict[str, Any]] = []
    index_entries: List[Dict[str, Any]] = []

    for (pP, T, M, S) in scenarios:
        # Build a per-scenario candidate set (consistent with original behavior)
        cand_seed = random.randrange(10**9)
        candidates = _build_global_candidates(
            pP,
            T,
            M,
            S,
            seed=cand_seed,
            coord_delta=float(args.coord_delta),
            n_dirichlet=int(args.n_dirichlet),
            dirichlet_kappa=float(args.dirichlet_kappa),
        )

        # Stage A with CRN seeds
        seeds_A = [random.randrange(10**9) for _ in range(int(args.stageA_reps))]
        stageA: List[Tuple[Tuple[float, ...], float]] = []
        for w in candidates:
            stats = _eval_candidate_on_scenario(
                pP,
                T,
                M,
                S,
                w,
                seeds_A,
                num_p1=(pP if p1_only else num_p1),
                rpause_frac=rpause_frac,
                alpha=float(args.alpha),
            )
            row = dict(stats)
            row.update(
                {
                    "P": pP,
                    "T": T,
                    "M": M,
                    "S": S,
                    "weights": list(w),
                    "stage": "A",
                    "num_p1": (pP if p1_only else num_p1),
                    "rpause_frac": rpause_frac,
                }
            )
            all_rows.append(row)
            stageA.append((w, stats["mean_obj"]))
        stageA.sort(key=lambda t: t[1], reverse=True)
        finalists = [w for w, _ in stageA[: max(1, int(args.top_k))]]

        # Stage B with new CRN seeds
        seeds_B = [random.randrange(10**9) for _ in range(int(args.stageB_reps))]
        stageB_rows: List[Tuple[Tuple[float, ...], float]] = []
        for w in finalists:
            statsB = _eval_candidate_on_scenario(
                pP,
                T,
                M,
                S,
                w,
                seeds_B,
                num_p1=(pP if p1_only else num_p1),
                rpause_frac=rpause_frac,
                alpha=float(args.alpha),
            )
            row = dict(statsB)
            row.update(
                {
                    "P": pP,
                    "T": T,
                    "M": M,
                    "S": S,
                    "weights": list(w),
                    "stage": "B",
                    "num_p1": (pP if p1_only else num_p1),
                    "rpause_frac": rpause_frac,
                }
            )
            all_rows.append(row)
            stageB_rows.append((w, statsB["mean_obj"]))
        stageB_rows.sort(key=lambda t: t[1], reverse=True)

        if stageB_rows:
            best_w, best_obj = stageB_rows[0]
            alts = stageB_rows[1:3]
            index_entries.append(
                {
                    "P": pP,
                    "T": T,
                    "M": M,
                    "S": S,
                    "num_p1": (pP if p1_only else num_p1),
                    "rpause_frac": rpause_frac,
                    "best": {"weights": list(best_w), "mean_obj": float(best_obj)},
                    "alts": [
                        {"weights": list(w), "mean_obj": float(o)} for (w, o) in alts
                    ],
                    "n_reps": int(args.stageB_reps),
                }
            )

        print(
            f"[oracle][per-scenario] (P={pP}, T={T}, M={M}, S={S}) -> {'OK' if stageB_rows else 'NO CANDIDATE'}"
        )

    # Write compact JSON (per-scenario)
    payload = {
        "meta": {
            "mode": "per-scenario",
            "alpha": float(args.alpha),
            "stageA_reps": int(args.stageA_reps),
            "stageB_reps": int(args.stageB_reps),
            "top_k": int(args.top_k),
            "seed": int(args.base_seed),
            "defaults_included": {"T": DEFAULTS_T, "M": DEFAULTS_M, "S": DEFAULTS_S},
            "notes": "Per-scenario best weights by T/M/S grid.",
        },
        "index": index_entries,
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Wide table (Parquet → CSV fallback)
    try:
        df = pd.DataFrame(all_rows)
        df.to_parquet(args.output_table, index=False)
    except Exception:
        alt_csv = os.path.splitext(args.output_table)[0] + ".csv"
        pd.DataFrame(all_rows).to_csv(alt_csv, index=False)

    print(f"[oracle] Wrote per-scenario index to {args.output_json}")
    print(f"[oracle] Wrote table to {args.output_table} (or CSV fallback)")


if __name__ == "__main__":
    main()
