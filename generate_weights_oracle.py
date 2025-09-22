# Conversations/generate_weights_oracle.py
from __future__ import annotations

import argparse
import json
import math
import os
import random
from itertools import product
from typing import Any, Dict, List, Tuple, Type

import pandas as pd

from core.engine import Engine

# Player classes
from players.player_1.player import Player1
from players.player_2.player import Player2
from players.player_3.player import Player3
from players.player_5.player import Player5
from players.player_6.player import Player6
from players.player_7.player import Player7
from players.player_9.player import Player9
from players.player_10.player import Player10

from players.random_player import RandomPlayer
from players.random_pause_player import RandomPausePlayer

# Path to save oracle JSON
from players.player_1.weight_policy import DEFAULT_ORACLE_PATH

DEFAULTS_T = int(os.getenv("DEFAULT_T", 10))
DEFAULTS_M = int(os.getenv("DEFAULT_M", 10))
DEFAULTS_S = int(os.getenv("DEFAULT_S", 20))

# A registry to resolve names passed via --opponents
OPPONENT_REGISTRY: Dict[str, Type] = {
    "player1": Player1,
    "player2": Player2,
    "player3": Player3,
    "player5": Player5,
    "player6": Player6,
    "player7": Player7,
    "player9": Player9,
    "player10": Player10,
    "random": RandomPlayer,
    "random_pause": RandomPausePlayer,
}


# ---------- local helpers ----------
def _clamp01(x: float) -> float:
    if math.isnan(x):
        return 0.0
    x = float(x)
    return 0.0 if x < 0 else (1.0 if x > 1 else x)

def _safe_mix(a: float, b: float) -> Tuple[float, float]:
    a = _clamp01(a)
    b = _clamp01(b)
    s = a + b
    if s <= 0:
        return (0.5, 0.5)
    return (a / s, b / s)

def _mix_grid(step: float = 0.1) -> List[Tuple[float, float]]:
    # (a,b) with a+b=1
    N = int(round(1.0 / step))
    out = [(k * step, 1.0 - k * step) for k in range(N + 1)]
    out += [(0.0, 1.0), (1.0, 0.0)]
    out = [tuple(round(v, 6) for v in _safe_mix(a, b)) for a, b in out]
    return sorted(list(set(out)), key=lambda t: t[0])

def _threshold_grid(step: float = 0.05) -> List[float]:
    N = int(round(1.0 / step))
    out = [round(k * step, 6) for k in range(N + 1)]
    out += [0.0, 1.0]
    return sorted(list(set(out)))

def _build_global_candidates(*, mix_step: float, thresh_step: float) -> List[Tuple[Tuple[float, float], float]]:
    Ms = _mix_grid(step=mix_step)
    Ts = _threshold_grid(step=thresh_step)
    return [(m, t) for m in Ms for t in Ts]


# ---------- opponents / roster building ----------
def _parse_opponents_spec(spec: str | None) -> List[str] | None:
    """
    Returns a list of normalized opponent keys or a sentinel:
      - None  -> use default (random/random_pause mix via rpause_frac)
      - ["player9"] -> fill all opponent slots with Player9
      - ["real_mix"] -> alias we expand later
      - ["player9","player7","random", ...] -> exact order (cycled if needed)
    """
    if not spec:
        return None
    s = spec.strip().lower()
    if s in ("default", "randoms", "rand", "random"):
        return None
    if s in ("player9", "p9"):
        return ["player9"]
    if s in ("real_mix", "mix", "real"):
        return ["real_mix"]
    # explicit comma list
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return parts or None


def _expand_real_mix() -> List[str]:
    # Always include Player9, then rotate the rest
    return ["player9", "player2", "player3", "player5", "player6", "player7", "player10"]



def _resolve_player_class(name: str) -> Type:
    return OPPONENT_REGISTRY.get(name, RandomPlayer)


def _build_player_types(
    P: int,
    num_p1: int,
    rpause_frac: float,
    opponents_spec: str | None,
    rng: random.Random | None = None,
) -> List[Type]:
    """
    Build the roster (list of classes) for Engine. Position/order doesn’t matter
    unless your engine couples identity to index (it shouldn’t).
    """
    rng = rng or random
    num_p1 = max(1, min(P, int(num_p1)))
    n_others = max(0, P - num_p1)

    # Always place Player1s
    roster: List[Type] = [Player1] * num_p1

    parsed = _parse_opponents_spec(opponents_spec)

    if n_others <= 0:
        return roster

    if parsed is None:
        # Default behavior: mix of Random and RandomPause by rpause_frac
        n_rpause = int(round(rpause_frac * n_others))
        n_random = n_others - n_rpause
        roster += [RandomPlayer] * n_random + [RandomPausePlayer] * n_rpause
        return roster

    # Special alias -> expand to a rotating slate
    if parsed == ["real_mix"]:
        pool = _expand_real_mix()  # ['player9', ...]
        # Guarantee Player9 if there is at least one opponent slot
        if n_others > 0:
            roster.append(_resolve_player_class("player9"))
            n_others -= 1
            # remove the first 'player9' from the cycling pool so we don't double-add
            pool = [tok for tok in pool if tok != "player9"]
            if not pool:
                pool = ["player2", "player3", "player5", "player6", "player7", "player10"]

        # Fill remaining opponent slots by cycling the rest
        i = 0
        while n_others > 0:
            roster.append(_resolve_player_class(pool[i % len(pool)]))
            i += 1
            n_others -= 1
        return roster


    # If a single token (e.g., ["player9"]), fill all with that
    if len(parsed) == 1:
        cls = _resolve_player_class(parsed[0])
        roster += [cls] * n_others
        return roster

    # Otherwise, cycle through the explicit list until we fill
    pool = [ _resolve_player_class(tok) for tok in parsed ]
    i = 0
    while len(roster) < P:
        roster.append(pool[i % len(pool)])
        i += 1
    return roster


# ---------- evaluation ----------
def _eval_once(
    P: int, T: int, M: int, S: int,
    ab: Tuple[float, float], threshold: float,
    seed: int, num_p1: int, rpause_frac: float,
    opponents_spec: str | None,
) -> float:
    rng = random.Random(seed)
    players_types = _build_player_types(P, num_p1, rpause_frac, opponents_spec, rng)

    random.seed(seed)
    engine = Engine(
        players=players_types,
        player_count=P,
        subjects=S,
        memory_size=M,
        conversation_length=T,
        seed=seed,
    )

    a, b = _safe_mix(*ab)
    for p in engine.players:
        if isinstance(p, Player1):
            setattr(p, "a", float(a))
            setattr(p, "b", float(b))
            setattr(p, "threshold", float(threshold))

    output = engine.run(players_types)
    p1_ids = [uid for uid, name in engine.player_names.items() if name == "Player1"]

    # Return the best individual score (not averaged)
    p1_individual = float("nan")
    for snap in output["scores"]["player_scores"]:
        if snap["id"] in p1_ids:
            p1_individual = float(snap["scores"]["individual"])
            break
    return p1_individual


def _eval_cand_on_scenario(
    P: int, T: int, M: int, S: int,
    ab: Tuple[float, float], threshold: float,
    seeds: List[int], num_p1: int, rpause_frac: float,
    opponents_spec: str | None,
) -> Dict[str, float]:
    vals = []
    for sd in seeds:
        ind = _eval_once(P, T, M, S, ab, threshold, sd, num_p1, rpause_frac, opponents_spec)
        vals.append(ind)
    import statistics as st
    return {
        "mean_score": float(st.fmean(vals)),
        "max_score": float(max(vals)),
        "std_score": float(st.pstdev(vals)) if len(vals) > 1 else 0.0,
        "n_reps": len(vals),
    }


# ---------- CLI ----------
def _parse_int_list(vals: List[str]) -> List[int]:
    out: List[int] = []
    for v in vals:
        if "," in v:
            out.extend(int(x.strip()) for x in v.split(",") if x.strip())
        else:
            out.append(int(v))
    return out


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Learn (a,b,threshold) with a+b=1 for Player1.")
    ap.add_argument("--mode", choices=["robust", "per-scenario"], default="per-scenario")
    ap.add_argument("-P", "--players", required=True, type=int)

    # Roster control
    ap.add_argument("--num-p1", type=str, default="1")
    ap.add_argument("--p1-only", action="store_true")
    ap.add_argument("--rpause-frac", type=float, default=0.5)
    ap.add_argument(
        "--opponents",
        type=str,
        default=None,
        help=(
            "Who to play against. Options:\n"
            "  - not set / 'random' / 'default' : Random + RandomPause controlled by --rpause-frac\n"
            "  - 'player9' (or 'p9')            : all opponents are Player9\n"
            "  - 'real_mix'                      : cycle [Player2,3,5,6,7,9,10]\n"
            "  - comma list (e.g. 'player9,player7,random') : exact repeating roster"
        ),
    )

    # Scenario space
    ap.add_argument("-T", "--lengths", nargs="+", type=str, default=["12","16","20","24","28","32","40","60","80","100"])
    ap.add_argument("-M", "--mem-sizes", nargs="+", type=str, default=["8","12","16","20","24","32","40","48","64"])
    ap.add_argument("-S", "--subjects", nargs="+", type=str, default=["6","8","10","12","16","20"])

    ap.add_argument("--players-grid", nargs="+", type=str, default=["4","6","8","10"])
    ap.add_argument("--lengths-grid", nargs="+", type=str, default=["12","20","28","40","60","80","100"])
    ap.add_argument("--mem-grid", nargs="+", type=str, default=["8","12","16","20","24","32","40","48","64"])
    ap.add_argument("--subjects-grid", nargs="+", type=str, default=["6","8","10","12","16","20"])

    # Search params
    ap.add_argument("--mix-step", type=float, default=0.1)
    ap.add_argument("--thresh-step", type=float, default=0.05)
    ap.add_argument("--stageA-reps", type=int, default=2)
    ap.add_argument("--stageB-reps", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=8)

    # Outputs
    ap.add_argument("--output-json", default=DEFAULT_ORACLE_PATH)
    ap.add_argument("--output-table", default="players/player_1/data/ab_threshold_oracle_full.parquet")
    ap.add_argument("--base-seed", type=int, default=4242)
    return ap


def _ensure_dirs(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


# ---------- main ----------
def main() -> None:
    ap = build_argparser()
    args = ap.parse_args()

    P = int(args.players)
    p1_only = bool(args.p1_only)
    opponents_spec = args.opponents

    rpause_frac = 0.0 if p1_only else max(0.0, min(1.0, float(args.rpause_frac)))
    if p1_only:
        num_p1 = P
    else:
        raw_num = (args.num_p1 or "1").strip()
        if raw_num.lower() == "p":
            num_p1 = P
        else:
            try:
                num_p1 = max(1, min(P, int(raw_num)))
            except ValueError:
                num_p1 = 1

    random.seed(args.base_seed)
    _ensure_dirs(args.output_json)
    _ensure_dirs(args.output_table)

    # Always per-scenario in this script (robust mode omitted here)
    GRID_P = _parse_int_list(list(args.players_grid))
    GRID_T = sorted(set(_parse_int_list(list(args.lengths_grid)) + [DEFAULTS_T]))
    GRID_M = sorted(set(_parse_int_list(list(args.mem_grid)) + [DEFAULTS_M]))
    GRID_S = sorted(set(_parse_int_list(list(args.subjects_grid)) + [DEFAULTS_S]))
    scenarios = list(product(GRID_P, GRID_T, GRID_M, GRID_S))

    index_entries: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []

    for (pP, T, M, S) in scenarios:
        # Candidate grid for (a,b,threshold)
        cands = _build_global_candidates(mix_step=float(args.mix_step), thresh_step=float(args.thresh_step))

        # Stage A
        seeds_A = [random.randrange(10**9) for _ in range(int(args.stageA_reps))]
        stageA = []
        for ab, thresh in cands:
            r = _eval_cand_on_scenario(
                pP, T, M, S, ab, thresh, seeds_A,
                (pP if p1_only else num_p1),
                rpause_frac,
                opponents_spec,
            )
            all_rows.append({
                "P": pP, "T": T, "M": M, "S": S,
                "a": ab[0], "b": ab[1], "threshold": thresh,
                "stage": "A", **r
            })
            # Rank by BEST individual score, not mean
            stageA.append(((ab, thresh), r["max_score"]))
        stageA.sort(key=lambda t: t[1], reverse=True)
        finalists = [k for k, _ in stageA[:max(1, int(args.top_k))]]

        # Stage B
        seeds_B = [random.randrange(10**9) for _ in range(int(args.stageB_reps))]
        stageB = []
        for ab, thresh in finalists:
            r = _eval_cand_on_scenario(
                pP, T, M, S, ab, thresh, seeds_B,
                (pP if p1_only else num_p1),
                rpause_frac,
                opponents_spec,
            )
            all_rows.append({
                "P": pP, "T": T, "M": M, "S": S,
                "a": ab[0], "b": ab[1], "threshold": thresh,
                "stage": "B", **r
            })
            stageB.append(((ab, thresh), r["max_score"]))
        stageB.sort(key=lambda t: t[1], reverse=True)

        if stageB:
            (best_ab, best_thresh), best_score = stageB[0]
            alts = stageB[1:3]
            index_entries.append({
                "P": pP, "T": T, "M": M, "S": S,
                "num_p1": (pP if p1_only else num_p1),
                "rpause_frac": rpause_frac,
                "opponents": opponents_spec if opponents_spec else "default",
                "best": {"a": best_ab[0], "b": best_ab[1], "threshold": best_thresh, "score": float(best_score)},
                "alts": [{"a": ab[0], "b": ab[1], "threshold": t, "score": float(obj)}
                         for ((ab, t), obj) in alts],
                "n_reps": int(args.stageB_reps),
            })

        print(f"[oracle][per-scenario] (P={pP},T={T},M={M},S={S}) -> {'OK' if stageB else 'NO CANDIDATE'}")

    payload = {
        "meta": {
            "mode": "per-scenario",
            "stageA_reps": int(args.stageA_reps),
            "stageB_reps": int(args.stageB_reps),
            "top_k": int(args.top_k),
            "seed": int(args.base_seed),
            "defaults_included": {"T": DEFAULTS_T, "M": DEFAULTS_M, "S": DEFAULTS_S},
            "notes": "Index stores (a,b,threshold) maximizing best individual score; opponents selectable via --opponents"
        },
        "index": index_entries
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    try:
        pd.DataFrame(all_rows).to_parquet(args.output_table, index=False)
    except Exception:
        alt = os.path.splitext(args.output_table)[0] + ".csv"
        pd.DataFrame(all_rows).to_csv(alt, index=False)

    print(f"[oracle] Wrote index to {args.output_json}")
    print(f"[oracle] Wrote table to {args.output_table} (or CSV fallback)")


if __name__ == "__main__":
    main()
