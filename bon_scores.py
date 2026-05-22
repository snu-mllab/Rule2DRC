#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import Any

from tqdm import tqdm

EVAL_RESULTS = "eval_results.json"


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def _read_json_any(path: Path) -> Any | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _cand_dirs(problem_dir: Path, cand_min: int, cand_max: int | None) -> list[Path]:
    return [
        d for d in sorted(problem_dir.iterdir())
        if d.is_dir()
        and d.name.startswith("cand_")
        and d.name[5:].isdigit()
        and cand_min <= int(d.name[5:]) and (cand_max is None or int(d.name[5:]) < cand_max)
    ]


def _gt_rewards(cand_dir: Path) -> tuple[float | None, float | None]:
    payload = _read_json(cand_dir / EVAL_RESULTS)
    if not payload:
        return None, None
    cr = payload.get("compile_rate")
    cr_f = float(cr) if isinstance(cr, (int, float)) else 0.0
    return (1.0 if payload.get("success") else 0.0, 1.0 if cr_f >= 1.0 else 0.0)


def _gt_from_cache(cand_dir: Path, score_rel: str) -> tuple[float, float] | None:
    data = _read_json(cand_dir / score_rel)
    if isinstance(data, dict):
        gt_success = data.get("gt_success")
        gt_compile = data.get("gt_compile")
        if isinstance(gt_success, (int, float)) and isinstance(gt_compile, (int, float)):
            return float(gt_success), float(gt_compile)
    gt_success, gt_compile = _gt_rewards(cand_dir)
    if gt_success is None or gt_compile is None:
        return None
    return float(gt_success), float(gt_compile)


def _score_file_for_pool(template: str, lo: int, hi_excl: int, pool_size: int) -> str:
    tag = f"c{lo:04d}-{hi_excl:04d}"
    if "{" not in template:
        return template
    return template.format(lo=lo, hi=hi_excl, tag=tag, pool_size=pool_size)


def _get_score(data: dict | None, method: str) -> float | None:
    if not data:
        return None
    scores = data.get("scores")
    if not isinstance(scores, dict):
        return None
    v = scores.get(method)
    return float(v) if isinstance(v, (int, float)) else None


def _pool_mean(vals: list[float]) -> float | None:
    return (sum(vals) / len(vals)) if vals else None


def _pool_std(vals: list[float]) -> float | None:
    if not vals:
        return None
    mean = _pool_mean(vals)
    if mean is None:
        return None
    return math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))


def _means_by_key(data: dict[int, list[float]]) -> list[float]:
    return [_pool_mean(data[k]) for k in sorted(data) if data[k]]


def _merge_metric_lists(dst: dict[int, list[float]], src: dict[int, list[float]]) -> None:
    for key, values in src.items():
        dst.setdefault(key, []).extend(values)


def _process_problem(
    pdir: Path,
    cand_min: int,
    cand_max: int | None,
    pool_size: int,
    score_method: str,
    score_file: str,
) -> dict[str, Any]:
    pass_by_pool: dict[int, list[float]] = {}
    compile_by_pool: dict[int, list[float]] = {}
    bon_success_by_pool: dict[int, list[float]] = {}
    bon_compile_by_pool: dict[int, list[float]] = {}
    single_success_by_pos: dict[int, list[float]] = {}
    single_compile_by_pos: dict[int, list[float]] = {}
    pools_pass_total = pools_bon_total = pools_skipped_gt = pools_skipped_score = 0
    has_pass = False
    has_bon = False

    cand_dirs = _cand_dirs(pdir, cand_min, cand_max)
    if len(cand_dirs) < pool_size:
        return {
            "pass_by_pool": pass_by_pool,
            "compile_by_pool": compile_by_pool,
            "bon_success_by_pool": bon_success_by_pool,
            "bon_compile_by_pool": bon_compile_by_pool,
            "single_success_by_pos": single_success_by_pos,
            "single_compile_by_pos": single_compile_by_pos,
            "pools_pass_total": pools_pass_total,
            "pools_bon_total": pools_bon_total,
            "pools_skipped_gt": pools_skipped_gt,
            "pools_skipped_score": pools_skipped_score,
            "has_pass": has_pass,
            "has_bon": has_bon,
        }

    for cand_dir in cand_dirs:
        cand_i = int(cand_dir.name[5:])
        pool_lo = cand_i - ((cand_i - cand_min) % pool_size)
        score_rel = _score_file_for_pool(score_file, pool_lo, pool_lo + pool_size, pool_size)
        gt = _gt_from_cache(cand_dir, score_rel)
        if gt is None:
            continue
        pos = cand_i - cand_min
        single_success_by_pos.setdefault(pos, []).append(gt[0])
        single_compile_by_pos.setdefault(pos, []).append(gt[1])

    for i in range(0, len(cand_dirs) - pool_size + 1, pool_size):
        pool = cand_dirs[i:i + pool_size]
        if len(pool) != pool_size:
            continue

        lo = int(pool[0].name[5:])
        hi_excl = lo + pool_size
        score_rel = _score_file_for_pool(score_file, lo, hi_excl, pool_size)

        gts: list[tuple[float, float]] = []
        for cand_dir in pool:
            gt = _gt_from_cache(cand_dir, score_rel)
            if gt is None:
                gts = []
                break
            gts.append(gt)

        if not gts:
            pools_skipped_gt += 1
            continue

        pools_pass_total += 1
        pass_val = 1.0 if any(s >= 1.0 for s, _c in gts) else 0.0
        compile_val = 1.0 if any(c >= 1.0 for _s, c in gts) else 0.0
        pool_idx = (lo - cand_min) // pool_size
        pass_by_pool.setdefault(pool_idx, []).append(pass_val)
        compile_by_pool.setdefault(pool_idx, []).append(compile_val)
        has_pass = True

        if score_method == "pass_k":
            scored = [
                (-1.0 if c < 1.0 else s, int(d.name[5:]), s, c)
                for d, (s, c) in zip(pool, gts)
            ]
        else:
            scored = []
            for d, (s, c) in zip(pool, gts):
                data = _read_json(d / score_rel)
                sc = _get_score(data, score_method)
                if sc is None:
                    scored = []
                    break
                scored.append((float(sc), int(d.name[5:]), s, c))

        if not scored:
            pools_skipped_score += 1
            pools_bon_total += 1
            bon_success_by_pool.setdefault(pool_idx, []).append(0.0)
            bon_compile_by_pool.setdefault(pool_idx, []).append(0.0)
            has_bon = True
            continue

        pools_bon_total += 1
        best = max(scored, key=lambda t: (t[0], -t[1]))
        bon_s = 1.0 if best[2] >= 1.0 else 0.0
        bon_c = 1.0 if best[3] >= 1.0 else 0.0
        bon_success_by_pool.setdefault(pool_idx, []).append(bon_s)
        bon_compile_by_pool.setdefault(pool_idx, []).append(bon_c)
        has_bon = True

    return {
        "pass_by_pool": pass_by_pool,
        "compile_by_pool": compile_by_pool,
        "bon_success_by_pool": bon_success_by_pool,
        "bon_compile_by_pool": bon_compile_by_pool,
        "single_success_by_pos": single_success_by_pos,
        "single_compile_by_pos": single_compile_by_pos,
        "pools_pass_total": pools_pass_total,
        "pools_bon_total": pools_bon_total,
        "pools_skipped_gt": pools_skipped_gt,
        "pools_skipped_score": pools_skipped_score,
        "has_pass": has_pass,
        "has_bon": has_bon,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compute pool-size metrics (pass@N and BoN@N) by partitioning candidates into pools."
    )
    ap.add_argument("run_dir", help="BON run directory (contains per-problem cand_#### folders).")
    ap.add_argument("--pool-size", type=int, required=True, help="Pool size N (e.g. 10 => cand_0000-0010, 0010-0020, ...).")
    ap.add_argument("--score-method", default="pass_k", help="Selection score to maximize (pass_k or cache key).")
    ap.add_argument("--score-file", default="scores.json",
                    help="Relative under each cand_dir; can use {tag}/{lo}/{hi}/{pool_size} formatting.")
    ap.add_argument("--cand-min", type=int, default=0, help="Only use candidates cand_<cand-min>..")
    ap.add_argument("--cand-max", type=int, default=None, help="Only use candidates cand_0000..cand_<cand-max-1>")
    ap.add_argument("--jobs", type=int, default=100, help="Number of worker threads to use across problems.")
    ap.add_argument("--out", default=None, help="Output JSON path (default: <run-dir>/bon_pool<N>__<method>.json).")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise SystemExit(f"run_dir not found: {run_dir}")
    if args.pool_size <= 0:
        raise SystemExit(f"--pool-size must be positive, got {args.pool_size}")
    if args.cand_min < 0:
        raise SystemExit(f"--cand-min must be >= 0, got {args.cand_min}")
    if args.cand_max is not None and args.cand_max <= args.cand_min:
        raise SystemExit(f"--cand-max must be > --cand-min, got {args.cand_max} <= {args.cand_min}")
    if args.jobs <= 0:
        raise SystemExit(f"--jobs must be positive, got {args.jobs}")

    problem_dirs = sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name != "judge_prompts"], key=lambda p: p.name)
    if not problem_dirs:
        raise SystemExit(f"No problem dirs found under: {run_dir}")

    pass_by_pool: dict[int, list[float]] = {}
    compile_by_pool: dict[int, list[float]] = {}
    bon_success_by_pool: dict[int, list[float]] = {}
    bon_compile_by_pool: dict[int, list[float]] = {}
    single_success_by_pos: dict[int, list[float]] = {}
    single_compile_by_pos: dict[int, list[float]] = {}
    n_problems_used_pass = 0
    n_problems_used_bon = 0
    pools_pass_total = pools_bon_total = pools_skipped_gt = pools_skipped_score = 0

    if args.jobs == 1:
        problem_metrics_iter = (
            _process_problem(
                pdir,
                args.cand_min,
                args.cand_max,
                args.pool_size,
                args.score_method,
                args.score_file,
            )
            for pdir in problem_dirs
        )
    else:
        ex = ThreadPoolExecutor(max_workers=args.jobs)
        problem_metrics_iter = ex.map(
            _process_problem,
            problem_dirs,
            repeat(args.cand_min),
            repeat(args.cand_max),
            repeat(args.pool_size),
            repeat(args.score_method),
            repeat(args.score_file),
        )

    try:
        for metrics in tqdm(problem_metrics_iter, total=len(problem_dirs), desc="Pooling", unit="problem"):
            _merge_metric_lists(pass_by_pool, metrics["pass_by_pool"])
            _merge_metric_lists(compile_by_pool, metrics["compile_by_pool"])
            _merge_metric_lists(bon_success_by_pool, metrics["bon_success_by_pool"])
            _merge_metric_lists(bon_compile_by_pool, metrics["bon_compile_by_pool"])
            _merge_metric_lists(single_success_by_pos, metrics["single_success_by_pos"])
            _merge_metric_lists(single_compile_by_pos, metrics["single_compile_by_pos"])
            n_problems_used_pass += int(metrics["has_pass"])
            n_problems_used_bon += int(metrics["has_bon"])
            pools_pass_total += int(metrics["pools_pass_total"])
            pools_bon_total += int(metrics["pools_bon_total"])
            pools_skipped_gt += int(metrics["pools_skipped_gt"])
            pools_skipped_score += int(metrics["pools_skipped_score"])
    finally:
        if args.jobs != 1:
            ex.shutdown()

    pass_pool_means = _means_by_key(pass_by_pool)
    compile_pool_means = _means_by_key(compile_by_pool)
    bon_success_pool_means = _means_by_key(bon_success_by_pool)
    bon_compile_pool_means = _means_by_key(bon_compile_by_pool)
    single_success_pos_means = _means_by_key(single_success_by_pos)
    single_compile_pos_means = _means_by_key(single_compile_by_pos)

    out = {
        "run_dir": str(run_dir),
        "pool_size": int(args.pool_size),
        "cand_min": int(args.cand_min),
        "cand_max": int(args.cand_max) if args.cand_max is not None else None,
        "score_method": args.score_method,
        "score_file": args.score_file,
        "n_problems_total": len(problem_dirs),
        "n_problems_used_pass": n_problems_used_pass,
        "n_problems_used_bon": n_problems_used_bon,
        "pools_pass_total": int(pools_pass_total),
        "pools_bon_total": int(pools_bon_total),
        "pools_skipped_gt": int(pools_skipped_gt),
        "pools_skipped_score": int(pools_skipped_score),
        "pass_at_n": {
            "mean": _pool_mean(pass_pool_means),
            "std": _pool_std(pass_pool_means),
            "n": len(pass_pool_means),
            "pool_means": pass_pool_means,
        },
        "compile_at_n": {
            "mean": _pool_mean(compile_pool_means),
            "std": _pool_std(compile_pool_means),
            "n": len(compile_pool_means),
            "pool_means": compile_pool_means,
        },
        "bon_success_at_n": {
            "mean": _pool_mean(bon_success_pool_means),
            "std": _pool_std(bon_success_pool_means),
            "n": len(bon_success_pool_means),
            "pool_means": bon_success_pool_means,
        },
        "bon_compile_at_n": {
            "mean": _pool_mean(bon_compile_pool_means),
            "std": _pool_std(bon_compile_pool_means),
            "n": len(bon_compile_pool_means),
            "pool_means": bon_compile_pool_means,
        },
        "single_success": {
            "mean": _pool_mean(single_success_pos_means),
            "std": _pool_std(single_success_pos_means),
            "n": len(single_success_pos_means),
        },
        "single_compile": {
            "mean": _pool_mean(single_compile_pos_means),
            "std": _pool_std(single_compile_pos_means),
            "n": len(single_compile_pos_means),
        },
    }

    out_path = Path(args.out) if args.out else (run_dir / f"bon_pool{args.pool_size}__{args.score_method}.json")
    prev = _read_json_any(out_path)
    if isinstance(prev, list):
        payload = prev
    elif isinstance(prev, dict):
        payload = [prev]
    else:
        payload = []
    payload.append(out)
    payload = sorted(
        {json.dumps(d, sort_keys=True): d for d in payload}.values(),
        key=lambda d: (str(d.get("score_file","")), str(d.get("score_method","")), int(d.get("cand_min") or 0), int(d.get("cand_max") or 0), int(d.get("pool_size") or 0)),
    )
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote: {out_path} (entries={len(payload)})")

if __name__ == "__main__":
    main()
