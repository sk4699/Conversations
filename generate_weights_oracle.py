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

from core.engine import Engine
from players.player_1.player import Player1
from players.random_player import RandomPlayer
from players.random_pause_player import RandomPausePlayer

from players.player_1.weight_policy import DEFAULT_ORACLE_PATH, _clamp_norm, _heuristic_weights

DEFAULTS_T = int(os.getenv("DEFAULT_T", 10))
DEFAULTS_M = int(os.getenv("DEFAULT_M", 10))
DEFAULTS_S = int(os.getenv("DEFAULT_S", 20))

# ---------- candidates ----------
def _dirichlet_near(prior: Tuple[float, ...], n: int, kappa: float) -> List[Tuple[float, ...]]:
    try:
        import numpy as np
    except Exception:
        return []
    alpha = (np.array(prior, dtype=float) * float(kappa)).clip(1e-3, None)
    samples = np.random.default_rng().dirichlet(alpha, size=n)
    return [tuple(map(float, s)) for s in samples]

def _coordinate_bumps(prior: Tuple[float, ...], delta: float) -> List[Tuple[float, ...]]:
    res: List[Tuple[float, ...]] = []
    k = len(prior)
    for i in range(k):
        for sign in (+1.0, -1.0):
            w = list(prior)
            w[i] = w[i] + sign * delta
            dec = (sign * delta) / (k - 1) * (-1.0)
            for j in range(k):
                if j != i:
                    w[j] = w[j] + dec
            res.append(_clamp_norm(w))
    return res

def _balanced_corners() -> List[Tuple[float, ...]]:
    return [
        _clamp_norm((0.35, 0.30, 0.15, 0.10, 0.10)),
        _clamp_norm((0.15, 0.25, 0.15, 0.10, 0.35)),
        _clamp_norm((0.20, 0.20, 0.30, 0.10, 0.20)),
        _clamp_norm((0.20, 0.25, 0.15, 0.25, 0.15)),
    ]

def _mix_grid(step: float = 0.1) -> List[Tuple[float, float]]:
    # (a,b) with a+b=1
    N = int(round(1.0 / step))
    out = [(k * step, 1.0 - k * step) for k in range(N + 1)]
    out += [(0.0, 1.0), (1.0, 0.0)]
    return sorted(list({(round(a, 6), round(b, 6)) for a, b in out}), key=lambda t: t[0])

def _build_global_candidates(P: int, mid_T: int, mid_M: int, mid_S: int,
                             *, seed: int, coord_delta: float,
                             n_dirichlet: int, dirichlet_kappa: float,
                             mix_step: float) -> List[Tuple[Tuple[float, ...], Tuple[float, float]]]:
    random.seed(seed)
    prior = _heuristic_weights(P, mid_T, mid_M, mid_S)
    Ws = [prior] + _coordinate_bumps(prior, coord_delta) + _balanced_corners() + _dirichlet_near(prior, n_dirichlet, dirichlet_kappa)
    # Deduplicate by rounding each coordinate inside each weight tuple
    Ws = sorted({tuple(round(v, 6) for v in w) for w in Ws})
    Ms = _mix_grid(step=mix_step)
    return [(w, m) for w in Ws for m in Ms]

# ---------- evaluation ----------
def _build_player_types(P: int, num_p1: int, rpause_frac: float) -> List[type]:
    num_p1 = max(1, min(P, int(num_p1)))
    n_others = max(0, P - num_p1)
    n_rpause = int(round(rpause_frac * n_others))
    n_random = n_others - n_rpause
    return [Player1] * num_p1 + [RandomPlayer] * n_random + [RandomPausePlayer] * n_rpause

def _eval_once(P: int, T: int, M: int, S: int,
               weights5: Tuple[float, ...], mix_ab: Tuple[float, float],
               seed: int, num_p1: int, rpause_frac: float) -> Tuple[float, float]:
    players_types = _build_player_types(P, num_p1, rpause_frac)
    random.seed(seed)
    engine = Engine(players=players_types, player_count=P, subjects=S, memory_size=M, conversation_length=T, seed=seed)

    w_coh, w_imp, w_pref, w_nonm, w_fre = weights5
    a, b = mix_ab
    s = max(1e-9, a + b); a, b = a / s, b / s

    for p in engine.players:
        if isinstance(p, Player1):
            p.w_coh, p.w_imp, p.w_pref, p.w_nonmon, p.w_fresh = (w_coh, w_imp, w_pref, w_nonm, w_fre)
            setattr(p, "weighted", float(a))
            setattr(p, "raw", float(b))

    output = engine.run(players_types)
    shared_total = float(output["scores"]["shared_score_breakdown"]["total"])
    p1_ids = [uid for uid, name in engine.player_names.items() if name == "Player1"]
    p1_individual = float("nan")
    for snap in output["scores"]["player_scores"]:
        if snap["id"] in p1_ids:
            p1_individual = float(snap["scores"]["individual"]); break
    return (p1_individual, shared_total)

def _objective(ind: float, shared: float, alpha: float) -> float:
    if math.isnan(ind): return float("-inf")
    return alpha * ind + (1.0 - alpha) * shared

def _eval_cand_on_scenario(P: int, T: int, M: int, S: int,
                           w5: Tuple[float, ...], ab: Tuple[float, float],
                           seeds: List[int], num_p1: int, rpause_frac: float,
                           alpha: float) -> Dict[str, float]:
    vals = []
    for sd in seeds:
        ind, sh = _eval_once(P, T, M, S, w5, ab, sd, num_p1, rpause_frac)
        vals.append(_objective(ind, sh, alpha))
    import statistics as st
    return {"mean_obj": float(st.fmean(vals)),
            "std_obj": float(st.pstdev(vals)) if len(vals) > 1 else 0.0,
            "n_reps": len(vals)}

# ---------- CLI ----------
def _parse_int_list(vals: List[str]) -> List[int]:
    out: List[int] = []
    for v in vals:
        if "," in v: out.extend(int(x.strip()) for x in v.split(",") if x.strip())
        else: out.append(int(v))
    return out

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Learn (w1..w5) and (a,b) with a+b=1 for Player1.")
    # --- Expanded single-run knobs ---
    ap.add_argument("--mode", choices=["robust", "per-scenario"], default="robust")
    ap.add_argument("-P", "--players", required=True, type=int)

    # Allow multiple Player1s if your harness supports it
    ap.add_argument("--num-p1", type=str, default="1")
    ap.add_argument("--p1-only", action="store_true")
    ap.add_argument("--rpause-frac", type=float, default=0.5)

    # Extend run-lengths up to 100
    ap.add_argument("-T", "--lengths", nargs="+", type=str,
                    default=["12","16","20","24","28","32","40","60","80","100"])

    # Scale memory sizes with longer runs (keep <= T in sanity checks below)
    ap.add_argument("-M", "--mem-sizes", nargs="+", type=str,
                    default=["8","12","16","20","24","32","40","48","64"])

    # Allow more subjects as conversations get longer
    ap.add_argument("-S", "--subjects", nargs="+", type=str,
                    default=["6","8","10","12","16","20"])

    # --- Expanded grids for search/sweep ---
    ap.add_argument("--players-grid", nargs="+", type=str,
                    default=["4","6","8","10"])
    ap.add_argument("--lengths-grid", nargs="+", type=str,
                    default=["12","20","28","40","60","80","100"])
    ap.add_argument("--mem-grid", nargs="+", type=str,
                    default=["8","12","16","20","24","32","40","48","64"])
    ap.add_argument("--subjects-grid", nargs="+", type=str,
                    default=["6","8","10","12","16","20"])

    # Tuning knobs
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--coord-delta", type=float, default=0.07)
    ap.add_argument("--n-dirichlet", type=int, default=10)
    ap.add_argument("--dirichlet-kappa", type=float, default=60.0)
    ap.add_argument("--mix-step", type=float, default=0.1)
    ap.add_argument("--stageA-reps", type=int, default=2)
    ap.add_argument("--stageB-reps", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=8)

    # Outputs
    ap.add_argument("--output-json", default=DEFAULT_ORACLE_PATH)
    ap.add_argument("--output-robust", default="players/player_1/data/weights_oracle_robust.json")
    ap.add_argument("--output-table", default="players/player_1/data/weights_oracle_full.parquet")

    ap.add_argument("--base-seed", type=int, default=4242)

    return ap


def _ensure_dirs(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d): os.makedirs(d, exist_ok=True)

# ---------- main ----------
def main() -> None:
    ap = build_argparser()
    args = ap.parse_args()

    P = int(args.players)
    p1_only = bool(args.p1_only)
    rpause_frac = 0.0 if p1_only else max(0.0, min(1.0, float(args.rpause_frac)))
    if p1_only:
        num_p1 = P
    else:
        raw_num = (args.num_p1 or "1").strip()
        if raw_num.lower() == "p": num_p1 = P
        else:
            try: num_p1 = max(1, min(P, int(raw_num)))
            except ValueError: num_p1 = 1

    random.seed(args.base_seed)
    _ensure_dirs(args.output_json); _ensure_dirs(args.output_robust); _ensure_dirs(args.output_table)

    if args.mode == "robust":
        GRID_T = sorted(set(_parse_int_list(list(args.lengths)) + [DEFAULTS_T]))
        GRID_M = sorted(set(_parse_int_list(list(args.mem_sizes)) + [DEFAULTS_M]))
        GRID_S = sorted(set(_parse_int_list(list(args.subjects)) + [DEFAULTS_S]))
        scenarios = list(product([P], GRID_T, GRID_M, GRID_S))

        mid_T, mid_M, mid_S = GRID_T[len(GRID_T)//2], GRID_M[len(GRID_M)//2], GRID_S[len(GRID_S)//2]
        cand_seed = random.randrange(10**9)
        cands = _build_global_candidates(P, mid_T, mid_M, mid_S,
                                         seed=cand_seed, coord_delta=float(args.coord_delta),
                                         n_dirichlet=int(args.n_dirichlet), dirichlet_kappa=float(args.dirichlet_kappa),
                                         mix_step=float(args.mix_step))

        agg: Dict[Tuple[Tuple[float, ...], Tuple[float, float]], Dict[str, float]] = {}
        all_rows: List[Dict[str, Any]] = []

        # Stage A
        for (_, T, M, S) in scenarios:
            seeds_A = [random.randrange(10**9) for _ in range(int(args.stageA_reps))]
            for w5, ab in cands:
                r = _eval_cand_on_scenario(P, T, M, S, w5, ab, seeds_A, num_p1, rpause_frac, float(args.alpha))
                all_rows.append({"P": P, "T": T, "M": M, "S": S, "weights": list(w5), "mix_ab": list(ab), "stage": "A", **r})
                key = (w5, ab)
                agg.setdefault(key, {"sum": 0.0, "cnt": 0, "min": float("inf")})
                agg[key]["sum"] += r["mean_obj"]; agg[key]["cnt"] += 1; agg[key]["min"] = min(agg[key]["min"], r["mean_obj"])

        ranked = sorted(agg.items(), key=lambda kv: kv[1]["sum"]/max(1, kv[1]["cnt"]), reverse=True)
        finalists = [k for k, _ in ranked[:max(1, int(args.top_k))]]

        # Stage B
        final_stats = {k: {"sum": 0.0, "min": float("inf"), "n": 0} for k in finalists}
        for (_, T, M, S) in scenarios:
            seeds_B = [random.randrange(10**9) for _ in range(int(args.stageB_reps))]
            for k in finalists:
                w5, ab = k
                r = _eval_cand_on_scenario(P, T, M, S, w5, ab, seeds_B, num_p1, rpause_frac, float(args.alpha))
                all_rows.append({"P": P, "T": T, "M": M, "S": S, "weights": list(w5), "mix_ab": list(ab), "stage": "B", **r})
                final_stats[k]["sum"] += r["mean_obj"]; final_stats[k]["min"] = min(final_stats[k]["min"], r["mean_obj"]); final_stats[k]["n"] += 1

        best_avg = max(final_stats.items(), key=lambda kv: kv[1]["sum"]/max(1, kv[1]["n"]))[0]
        best_worst = max(final_stats.items(), key=lambda kv: kv[1]["min"])[0]

        robust = {
            "meta": {"mode": "robust", "alpha": float(args.alpha),
                     "stageA_reps": int(args.stageA_reps), "stageB_reps": int(args.stageB_reps),
                     "top_k": int(args.top_k), "seed": int(args.base_seed),
                     "defaults_included": {"T": DEFAULTS_T, "M": DEFAULTS_M, "S": DEFAULTS_S},
                     "notes": "Joint search over (w1..w5) and (a,b) with a+b=1"},
            "sweep": {"P": P, "num_p1": num_p1, "rpause_frac": rpause_frac,
                      "lengths": GRID_T, "mem_sizes": GRID_M, "subjects": GRID_S},
            "winners": {"best_avg": {"weights": list(best_avg[0]), "mix_ab": list(best_avg[1])},
                        "best_worst": {"weights": list(best_worst[0]), "mix_ab": list(best_worst[1])}},
        }
        with open(args.output_robust, "w", encoding="utf-8") as f: json.dump(robust, f, indent=2)
        try: pd.DataFrame(all_rows).to_parquet(args.output_table, index=False)
        except Exception:
            alt = os.path.splitext(args.output_table)[0] + ".csv"
            pd.DataFrame(all_rows).to_csv(alt, index=False)
        print("[oracle][robust] best_avg :", robust["winners"]["best_avg"])
        print("[oracle][robust] best_worst:", robust["winners"]["best_worst"])
        return

    # per-scenario (writes the index JSON Player1 reads)
    GRID_P = _parse_int_list(list(args.players_grid))
    GRID_T = sorted(set(_parse_int_list(list(args.lengths_grid)) + [DEFAULTS_T]))
    GRID_M = sorted(set(_parse_int_list(list(args.mem_grid)) + [DEFAULTS_M]))
    GRID_S = sorted(set(_parse_int_list(list(args.subjects_grid)) + [DEFAULTS_S]))
    scenarios = list(product(GRID_P, GRID_T, GRID_M, GRID_S))

    index_entries: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []

    for (pP, T, M, S) in scenarios:
        cand_seed = random.randrange(10**9)
        cands = _build_global_candidates(pP, T, M, S, seed=cand_seed, coord_delta=float(args.coord_delta),
                                         n_dirichlet=int(args.n_dirichlet), dirichlet_kappa=float(args.dirichlet_kappa),
                                         mix_step=float(args.mix_step))
        # Stage A
        seeds_A = [random.randrange(10**9) for _ in range(int(args.stageA_reps))]
        stageA = []
        for w5, ab in cands:
            r = _eval_cand_on_scenario(pP, T, M, S, w5, ab, seeds_A, (pP if p1_only else num_p1), rpause_frac, float(args.alpha))
            all_rows.append({"P": pP, "T": T, "M": M, "S": S, "weights": list(w5), "mix_ab": list(ab), "stage": "A", **r})
            stageA.append(((w5, ab), r["mean_obj"]))
        stageA.sort(key=lambda t: t[1], reverse=True)
        finalists = [k for k, _ in stageA[:max(1, int(args.top_k))]]

        # Stage B
        seeds_B = [random.randrange(10**9) for _ in range(int(args.stageB_reps))]
        stageB = []
        for w5, ab in finalists:
            r = _eval_cand_on_scenario(pP, T, M, S, w5, ab, seeds_B, (pP if p1_only else num_p1), rpause_frac, float(args.alpha))
            all_rows.append({"P": pP, "T": T, "M": M, "S": S, "weights": list(w5), "mix_ab": list(ab), "stage": "B", **r})
            stageB.append(((w5, ab), r["mean_obj"]))
        stageB.sort(key=lambda t: t[1], reverse=True)

        if stageB:
            (best_w5, best_ab), best_obj = stageB[0]
            alts = stageB[1:3]
            index_entries.append({
                "P": pP, "T": T, "M": M, "S": S,
                "num_p1": (pP if p1_only else num_p1),
                "rpause_frac": rpause_frac,
                "best": {"weights": list(best_w5), "mix_ab": list(best_ab), "mean_obj": float(best_obj)},
                "alts": [{"weights": list(w5), "mix_ab": list(ab), "mean_obj": float(obj)}
                         for ((w5, ab), obj) in alts],
                "n_reps": int(args.stageB_reps),
            })

        print(f"[oracle][per-scenario] (P={pP},T={T},M={M},S={S}) -> {'OK' if stageB else 'NO CANDIDATE'}")

    payload = {"meta": {"mode": "per-scenario", "alpha": float(args.alpha),
                        "stageA_reps": int(args.stageA_reps), "stageB_reps": int(args.stageB_reps),
                        "top_k": int(args.top_k), "seed": int(args.base_seed),
                        "defaults_included": {"T": DEFAULTS_T, "M": DEFAULTS_M, "S": DEFAULTS_S},
                        "notes": "Index stores both weights and mix_ab=[a,b]."},
               "index": index_entries}
    with open(args.output_json, "w", encoding="utf-8") as f: json.dump(payload, f, indent=2)

    try: pd.DataFrame(all_rows).to_parquet(args.output_table, index=False)
    except Exception:
        alt = os.path.splitext(args.output_table)[0] + ".csv"
        pd.DataFrame(all_rows).to_csv(alt, index=False)

    print(f"[oracle] Wrote index to {args.output_json}")
    print(f"[oracle] Wrote table to {args.output_table} (or CSV fallback)")

if __name__ == "__main__":
    main()



# python generate_weights_oracle.py \
#   --mode per-scenario \
#   -P 10 \
#   --players-grid 4 6 8 10 \
#   --lengths-grid 12 20 28 40 60 80 100 \
#   --mem-grid 8 12 16 20 24 32 40 48 64 \
#   --subjects-grid 6 8 10 12 16 20 \
#   --rpause-frac 0.5 \
#   --output-json players/player_1/data/w_oracle_per_scenario_rpause50.json \
#   --output-table players/player_1/data/w_oracle_per_scenario_rpause50.parquet
