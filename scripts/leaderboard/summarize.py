#!/usr/bin/env python3
"""Summarize a leaderboard run and print a ready-to-paste YAML row.

Reads Sample-1 (cand_0000) eval results from one or two bon.py run
directories and reports success / error rates matching the public
leaderboard convention:

  success: generated script matches the ground truth on every layout
  error:   script fails to compile or run on at least one layout

Usage:
  python scripts/leaderboard/summarize.py \
      --model openai/gpt-oss-120b \
      --with-context out_drc/problems/lb_..._ic_klayout_docs_<ts> \
      --without-context out_drc/problems/lb_..._noctx_<ts>

Either directory may be omitted (the corresponding cells print as null).
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path


def summarize_run(run_dir: Path) -> dict:
    n = n_success = n_error = n_missing = 0
    for problem_dir in sorted(run_dir.iterdir()):
        if not problem_dir.is_dir():
            continue
        n += 1
        results_path = problem_dir / "cand_0000" / "eval_results.json"
        try:
            payload = json.loads(results_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            n_missing += 1
            n_error += 1
            continue
        cr = payload.get("compile_rate")
        cr = float(cr) if isinstance(cr, (int, float)) else 0.0
        n_success += int(bool(payload.get("success")))
        n_error += int(cr < 1.0)
    if n == 0:
        raise SystemExit(f"No problem directories found under {run_dir}")
    return {
        "n": n,
        "success": 100.0 * n_success / n,
        "error": 100.0 * n_error / n,
        "missing": n_missing,
    }


def cell(stats: dict | None) -> str:
    if stats is None:
        return "{ success: null, error: null }"
    return f"{{ success: {stats['success']:.1f}, error: {stats['error']:.1f} }}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="Model identifier (e.g. openai/gpt-oss-120b)")
    ap.add_argument("--with-context", default=None, help="bon.py run dir with docs in context")
    ap.add_argument("--without-context", default=None, help="bon.py run dir without context")
    ap.add_argument("--note", default=None, help="Optional note (e.g. 'reasoning effort medium')")
    args = ap.parse_args()

    if not args.with_context and not args.without_context:
        raise SystemExit("Provide --with-context and/or --without-context")

    stats = {}
    for key, run_dir in (("with_context", args.with_context), ("without_context", args.without_context)):
        if run_dir is None:
            stats[key] = None
            continue
        s = summarize_run(Path(run_dir))
        stats[key] = s
        label = key.replace("_", " ")
        print(f"{label:>16}: success {s['success']:5.1f}%  error {s['error']:5.1f}%  "
              f"(n={s['n']}, missing eval results={s['missing']})")

    print("\nLeaderboard row (paste into _data/rule2drc_leaderboard.yml):\n")
    print(f"- model: {args.model}")
    print(f"  link: https://openrouter.ai/{args.model}")
    if args.note:
        print(f"  note: {args.note}")
    print("  source: openrouter")
    print(f'  date: "{date.today():%Y-%m}"')
    print("  public:")
    print(f"    with_context: {cell(stats['with_context'])}")
    print(f"    without_context: {cell(stats['without_context'])}")


if __name__ == "__main__":
    main()
