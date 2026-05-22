#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json, time
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm
from evaluate.klayout_eval import eval_deck_on_gds_dir

ROOT = Path(".").resolve()
SCORE_SCHEMA_VERSION = 1
EVAL_RESULTS = "eval_results.json"
LABELS_FILE = "labels.csv"


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def _cand_dirs(pdir: Path, cand_min: int, cand_max: int | None) -> list[Path]:
    return [
        d for d in sorted(pdir.iterdir())
        if d.is_dir()
        and d.name.startswith("cand_")
        and d.name[5:].isdigit()
        and cand_min <= int(d.name[5:]) and (cand_max is None or int(d.name[5:]) < cand_max)
    ]


def _find_deck(cand_dir: Path, prob: str) -> Path | None:
    p = cand_dir / f"{prob}.drc"
    if p.is_file():
        return p
    ds = sorted(cand_dir.glob("*.drc"))
    return ds[0] if ds else None


def _gt_rewards(cand_dir: Path) -> tuple[float | None, float | None]:
    payload = _read_json(cand_dir / EVAL_RESULTS)
    if not payload:
        return None, None
    cr = payload.get("compile_rate")
    cr_f = float(cr) if isinstance(cr, (int, float)) else 0.0
    return (1.0 if payload.get("success") else 0.0, 1.0 if cr_f >= 1.0 else 0.0)


def _suite_files(gds_dir: Path) -> list[str]:
    if not gds_dir.is_dir():
        return []
    return [p.relative_to(gds_dir).as_posix() for p in sorted(gds_dir.rglob("*.gds")) if p.is_file()]


def _read_labels(labels_csv: Path) -> tuple[list[str], dict[str, dict[str, int]]]:
    if not labels_csv.is_file():
        return [], {}
    rows: dict[str, dict[str, int]] = {}
    with labels_csv.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        cats = [h for h in (r.fieldnames or []) if h and h != "filename"]
        for row in r:
            fn = (row.get("filename") or "").strip()
            if not fn:
                continue
            rows[fn] = {c: int((row.get(c) or "0").strip() or 0) for c in cats}
    return cats, rows


def _expected_pattern(files: list[str], cats: list[str], rows: dict[str, dict[str, int]]) -> str:
    if not files or any(f not in rows for f in files):
        return ""
    return "".join("1" if any(rows[f].get(c, 0) != 0 for c in cats) else "0" for f in files)


def _predicted_pattern(files: list[str], cats: list[str], rows: dict[str, dict[str, int]],
                       mismatches: dict) -> str:
    if not files or any(f not in rows for f in files):
        return ""
    bits = []
    for f in files:
        row = rows[f]
        ov = {str(c): int(p) for c, _t, p in (mismatches.get(f) or [])}
        viol = any(ov.get(c, row.get(c, 0)) != 0 for c in cats)
        bits.append("1" if viol else "0")  # 1=violation
    return "".join(bits)


def _score(payload: dict, files: list[str]) -> float:
    if not files:
        return 0.0
    pre = payload.get("precheck", {})
    cr_f = float(payload.get("compile_rate", 0.0))
    if cr_f < 1.0 or pre.get("deck_unknown"):
        return -1.0
    bad = set(payload.get("mismatches", {}))
    missing_set = {str(x) for x in pre.get("labels_missing_rows", [])}
    return sum(1 for f in files if f not in bad and f not in missing_set) / max(1, len(files))


def _dump(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def _has_cached_score(score_path: Path, regen: bool) -> bool:
    if regen or not score_path.is_file():
        return False
    data = _read_json(score_path) or {}
    scores = data.get("scores", {})
    return (
        data.get("pattern_type") == "violation"
        and isinstance(scores.get("generated_tests"), (int, float))
    )


def eval_one(job: dict) -> dict:
    cand_dir = Path(job["cand_dir"])
    score_path = Path(job["score_path"])
    if _has_cached_score(score_path, bool(job["regen"])):
        return {"cached": True, "problem": job["problem"], "n_drc_evals": 0}

    gds_dir = Path(job["gds_dir"])
    files: list[str] = job["files"]
    labels_csv = gds_dir / LABELS_FILE

    # No tests (or malformed suite) => everyone 0
    score, pattern, exp_pat, gen_eval_rel = 0.0, "", "", ""
    n_drc_evals = 0
    if files and labels_csv.is_file():
        cats, rows = _read_labels(labels_csv)
        exp_pat = job["expected_pattern"]  # already computed in main (cheap)
        deck = Path(job["deck_path"]) if job["deck_path"] else None
        if deck is None or not deck.is_file():
            score = -1.0
        else:
            keep = bool(job["keep_eval"])
            out_eval_dir = Path(job["out_eval_dir"])
            payload = _read_json(out_eval_dir / EVAL_RESULTS) if keep and not job["regen"] else None
            if payload is None:
                payload = eval_deck_on_gds_dir(
                    problem_dir=Path(job["problem_dir"]),
                    deck_path=deck,
                    gds_dir=gds_dir,
                    out_dir=out_eval_dir if keep else None,  # None => temp dir auto-deleted
                    klayout_bin=job["klayout_bin"],
                    show_progress=False,
                )
                n_drc_evals = len(files)

            score = float(_score(payload, files))
            pre = payload.get("precheck", {})
            if float(payload.get("compile_rate", 0.0)) >= 1.0 and not pre.get("deck_unknown") and not pre.get("labels_missing_rows"):
                pattern = _predicted_pattern(files, cats, rows, payload.get("mismatches", {}))

            if keep:
                ep = out_eval_dir / EVAL_RESULTS
                gen_eval_rel = str(ep.relative_to(cand_dir)) if ep.is_file() else ""

    gt_success, gt_compile = _gt_rewards(cand_dir)
    _dump(score_path, {
        "schema_version": SCORE_SCHEMA_VERSION,
        "suite_id": job["suite_id"],
        "subset_tag": job["tag"],
        "cand_min": job["cand_min"],
        "cand_max": job["cand_max"],
        "gds_run": job["gds_run"],
        "keep_eval": bool(job["keep_eval"]),
        "gen_eval_path": gen_eval_rel,
        "scores": {"generated_tests": score},
        "pattern_type": "violation",        # 1=violation, 0=pass
        "pattern": pattern,                 # candidate output (reconstructed)
        "expected_pattern": exp_pat,        # expected from labels.csv
        "gt_success": gt_success,
        "gt_compile": gt_compile,
    })
    return {"cached": False, "score": score, "problem": job["problem"], "n_drc_evals": n_drc_evals}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--gds-run", required=True)
    ap.add_argument("--problems-dir", default="problems_v5")
    ap.add_argument("--suite-id", default=None)
    ap.add_argument("--eval-jobs", type=int, default=8)
    ap.add_argument("--klayout-bin", default="klayout")
    ap.add_argument("--regen", action="store_true")
    ap.add_argument("--cand-min", type=int, default=0, help="Only score cand_<cand-min>..")
    ap.add_argument("--cand-max", type=int, required=True, help="Only score cand_0000..cand_<cand-max-1>")
    ap.add_argument("--keep-eval", action="store_true", help="Persist eval artifacts under cand_dir/gen_eval/")
    ap.add_argument("--score-file", default=None, help="Relative under each cand_dir (overrides default naming)")
    ap.add_argument("--problem-stride", type=int, default=None,
                    help="Sample every N-th problem (e.g., 100 picks indices 0, 100, 200, ...)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    gds_run = Path(args.gds_run).resolve()
    problems_root = Path(args.problems_dir)
    if not problems_root.is_absolute():
        problems_root = (ROOT / problems_root).resolve()

    if args.cand_min < 0:
        raise SystemExit(f"--cand-min must be >= 0, got {args.cand_min}")
    if args.cand_max <= args.cand_min:
        raise SystemExit(f"--cand-max must be > --cand-min, got {args.cand_max} <= {args.cand_min}")

    suite_id = args.suite_id or gds_run.name
    tag = f"c{args.cand_min:04d}-{args.cand_max:04d}"

    default_score_rel = f"scores/generated_tests__{suite_id}.json"
    score_rel = args.score_file or default_score_rel
    jobs: list[dict] = []
    pruns = sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name != "judge_prompts"])
    if not pruns:
        raise SystemExit("No problem dirs found under run_dir")
    if args.problem_stride:
        pruns = pruns[::args.problem_stride]

    for prun in pruns:
        prob = prun.name
        pdir = problems_root / prob
        gds_dir = gds_run / prob / "selfgen_gds"
        files = _suite_files(gds_dir)

        cats, rows = _read_labels(gds_dir / LABELS_FILE)
        exp_pat = _expected_pattern(files, cats, rows)

        for cand_dir in _cand_dirs(prun, args.cand_min, args.cand_max):
            deck = _find_deck(cand_dir, prob)
            out_eval = cand_dir / "gen_eval" / f"{suite_id}__{tag}"
            jobs.append({
                "problem": prob,
                "problem_dir": str(pdir),
                "cand_dir": str(cand_dir),
                "deck_path": str(deck) if deck else "",
                "gds_dir": str(gds_dir),
                "files": files,
                "expected_pattern": exp_pat,
                "suite_id": suite_id,
                "gds_run": str(gds_run),
                "klayout_bin": args.klayout_bin,
                "regen": bool(args.regen),
                "keep_eval": bool(args.keep_eval),
                "out_eval_dir": str(out_eval),
                "score_path": str(cand_dir / score_rel),
                "tag": tag,
                "cand_min": args.cand_min,
                "cand_max": args.cand_max,
            })

    if not jobs:
        raise SystemExit("No candidates found to score.")

    t0 = time.time()
    ctx = mp.get_context("spawn")
    cached = 0
    cost: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=max(1, int(args.eval_jobs)), mp_context=ctx) as ex:
        futs = [ex.submit(eval_one, j) for j in jobs]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Scoring", unit="cand"):
            r = fut.result()
            cached += int(bool(r.get("cached")))
            prob = r.get("problem", "")
            if prob:
                pc = cost.setdefault(prob, {"n_drc_evals": 0})
                pc["n_drc_evals"] += r.get("n_drc_evals", 0)

    elapsed_s = time.time() - t0
    total_drc = sum(pc["n_drc_evals"] for pc in cost.values())
    cost_out = {
        "total": {
            "n_drc_evals": total_drc,
            "wall_clock_min": round(elapsed_s / 60, 2),
        },
        "per_problem": cost,
    }
    (run_dir / f"cost_score_tests__{suite_id}__{tag}.json").write_text(
        json.dumps(cost_out, indent=2) + "\n", encoding="utf-8")
    print(f"[done] suite={suite_id} tag={tag} score_file={score_rel} cached={cached}/{len(jobs)} keep_eval={args.keep_eval}")
    print(f"Cost: n_drc_evals={total_drc} wall_clock_s={round(elapsed_s, 2)}")


if __name__ == "__main__":
    main()
