#!/usr/bin/env python3
import argparse
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from collections import deque
from pathlib import Path

import yaml
from tqdm import tqdm

from evaluate.klayout_eval import (
    run_klayout,
    parse_rdb_counts,
    read_labels_csv,
    f1_from_counts,
    _read_csv_header,
)

ROOT = Path(".").resolve()
OUT  = ROOT / "out_eval"

# Simple ANSI colors for pretty terminal output
RESET = "\033[0m"
RED   = "\033[31m"
CYAN  = "\033[36m"
BOLD  = "\033[1m"

def eval_problem(
    spec_path: Path,
    deck_override: Path | None = None,
    out_path: Path | None = None,
    show_progress: bool = False,
) -> tuple[dict, dict[str, list[tuple[str, int, int]]]]:
    spec = yaml.safe_load(spec_path.read_text())
    pid   = spec["id"]
    title = spec.get("title", pid)
    cats  = list(spec["categories"])
    deck  = Path(deck_override) if deck_override else ROOT / spec["deck"]
    data_dir = ROOT / spec["data_dir"]
    labels_csv = data_dir / "labels.csv"

    # Collect files from pass/ and fail/
    files = []
    for sub in ("pass","fail"):
        files += list((data_dir / sub).glob("*.gds"))
    files = sorted(files)
    if not files:
        return {"id": pid, "title": title, "error": "no GDS files found"}, {}
    
    # Precheck: labels.csv header vs spec categories
    spec_cats = set(cats)
    header = _read_csv_header(labels_csv)
    label_cols = set(h for h in header if h!="filename")
    missing_in_labels = sorted(spec_cats - label_cols)
    extra_in_labels   = sorted(label_cols - spec_cats)
    precheck = {}
    if missing_in_labels:
        precheck["labels_missing"] = missing_in_labels
    if extra_in_labels:
        precheck["labels_extra"] = extra_in_labels

    # Ground-truth from labels.csv (still loads even if precheck flags issues)
    gt = read_labels_csv(labels_csv, cats)

    # Run deck on each file and accumulate per-category preds
    compiled = 0
    T = {c:[] for c in cats}; P = {c:[] for c in cats}
    unknown_deck_cats = set()
    per_file_mismatches = {}
    file_iter = tqdm(files, desc=f"[{pid}] {title}", leave=False) if show_progress else files
    for g in file_iter:
        assert str(ROOT) in str(g)
        rel_g = str(g)[len(str(ROOT)) + 1 :]

        rdb = out_path / (g.stem + ".lyrdb")
        ok = run_klayout("klayout", deck, g, rdb)
        compiled += int(ok)
        counts = parse_rdb_counts(rdb) if ok else {}
        # Track any categories the deck emits that are NOT in the spec
        unknown_deck_cats.update(set(counts.keys()) - spec_cats)
        for c in cats:
            t = gt[rel_g][c]
            p = 1 if counts.get(c, 0) > 0 else 0
            T[c].append(t)
            P[c].append(p)
            if t != p:
                per_file_mismatches.setdefault(rel_g, []).append((c, t, p))

    # Metrics
    per_cat = {}
    success = True
    for c in cats:
        m = f1_from_counts(T[c], P[c])
        per_cat[c] = m
        success &= (m["f1"] == 1.0)

    if unknown_deck_cats:
        precheck["deck_unknown"] = sorted(unknown_deck_cats)
        success = False
    if precheck:
        success = False

    return {
        "id": pid,
        "title": title,
        "deck": str(deck),
        "n_cases": len(files),
        "compile_rate": compiled/len(files),
        "per_category": per_cat,
        "precheck": precheck,
        "success": bool(success),
    }, per_file_mismatches

def _eval_worker(task: tuple[int, str, str | None, str, bool]
                 ) -> tuple[int, dict, dict[str, list[tuple[str, int, int]]]]:
    idx, spec_path, deck_path, out_path, show_progress = task
    deck_override = Path(deck_path) if deck_path else None
    result, mismatches = eval_problem(
        Path(spec_path),
        deck_override,
        Path(out_path),
        show_progress=show_progress,
    )
    return idx, result, mismatches

def main():
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="problems", help="Directory containing problems")
    ap.add_argument(
        "--problem",
        action="append",
        nargs="+",
        help="Restrict to specific problem ids (dir names). Accepts multiple ids per flag.",
        default=[],
    )
    ap.add_argument("--rule", help="Override deck for all selected problems (path to .drc).")
    ap.add_argument("--jobs", type=int, default=100, help="Parallel worker processes (problems in flight)")
    args = ap.parse_args()

    base_dir = Path(args.dir)
    if not base_dir.is_absolute():
        base_dir = ROOT / base_dir
    if not base_dir.exists():
        raise SystemExit(f"Problems directory not found: {base_dir}")

    out_dir = OUT / args.dir
    specs = sorted(base_dir.glob("*/spec.yaml"))
    problems = [p for group in args.problem for p in group]
    if problems:
        by_name = {p.parent.name: p for p in specs}
        by_num = {}
        for name, path in by_name.items():
            prefix = name.split("_", 1)[0]
            if prefix and prefix not in by_num:
                by_num[prefix] = path
        selected = []
        for query in problems:
            match = by_name.get(query) or by_num.get(query)
            if not match:
                raise SystemExit(f"Problem not found: {query}")
            selected.append(match)
        specs = selected

    deck_dir = Path(args.rule).name if args.rule else "gold"
    rule_root = Path(args.rule) if args.rule else None
    jobs = max(1, int(args.jobs or 1))

    results: list[dict] = [None] * len(specs)
    mismatches_all: list[tuple[str, dict[str, list[tuple[str, int, int]]]]] = []
    if jobs == 1:
        for idx, sp in enumerate(specs):
            pid = sp.parent.name
            deck_path = (rule_root / pid / f"{pid}.drc") if rule_root else None
            out_path = out_dir / deck_dir / pid
            result, mismatches = eval_problem(sp, deck_path, out_path, show_progress=True)
            results[idx] = result
            if mismatches:
                mismatches_all.append((pid, mismatches))
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as ex:
            task_q = deque()
            for idx, sp in enumerate(specs):
                pid = sp.parent.name
                deck_path = (rule_root / pid / f"{pid}.drc") if rule_root else None
                out_path = out_dir / deck_dir / pid
                task_q.append((idx, str(sp), str(deck_path) if deck_path else None, str(out_path), False))

            max_in_flight = max(1, jobs * 2)
            in_flight: set = set()

            def submit_more() -> None:
                while task_q and len(in_flight) < max_in_flight:
                    in_flight.add(ex.submit(_eval_worker, task_q.popleft()))

            with tqdm(total=len(specs), desc="Problems") as pbar:
                submit_more()
                while in_flight:
                    done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                    for fut in done:
                        in_flight.remove(fut)
                        idx, result, mismatches = fut.result()
                        results[idx] = result
                        if mismatches:
                            mismatches_all.append((result.get("id"), mismatches))
                        pbar.update(1)
                    submit_more()

    success_rate = sum(1 for r in results if r.get("success")) / max(1, len(results))
    payload = {"success_rate": success_rate, "results": results}
    text = json.dumps(payload, indent=2)
    print(text)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / deck_dir / "eval_results.json").write_text(text + "\n")

    # Human-readable summary printed at the very end, with colors.
    if mismatches_all:
        print()
        print(f"{BOLD}Failing GDS files and mismatched categories:{RESET}")
        for pid, per_file_mismatches in mismatches_all:
            print(f"{BOLD}[{pid}]{RESET}")
            for rel_g, mismatches in sorted(per_file_mismatches.items()):
                print(f"  {CYAN}{rel_g}{RESET}")
                for c, t, p in mismatches:
                    # All entries here are mismatches, so print in red.
                    print(f"    {RED}{c}: expected {t}, predicted {p}{RESET}")
    print(f"Finished. elapsed_min={(time.time()-t0)/60:.2f}")

if __name__ == "__main__":
    main()
