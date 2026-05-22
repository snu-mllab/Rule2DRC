# evaluate/klayout_eval.py
import csv
import json
import subprocess
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

import yaml


def run_klayout(klayout_bin: str, deck: Path, gds: Path, rdb: Path, timeout_s: int = 120) -> bool:
    rdb.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        klayout_bin, "-b", "-r", str(deck),
        "-rd", f"input={str(gds.resolve())}",
        "-rd", f"output={str(rdb.resolve())}",
    ]
    try:
        cp = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=int(timeout_s)
        )
        return (cp.returncode == 0) and rdb.exists()
    except subprocess.TimeoutExpired as e:
        return False


def parse_rdb_counts(rdb: Path) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    if not rdb.exists():
        return counts

    root = ET.parse(str(rdb)).getroot()

    # Case A: <category name="..."><item>...</item></category>
    for c in root.findall(".//category"):
        name = c.get("name") or ""
        if name:
            counts[name] += len(c.findall("./item"))

    # Case B: <items><item><category>NAME</category>...</item></items>
    for it in root.findall(".//items/item"):
        name = (it.findtext("category") or "").strip()
        if name:
            counts[name] += 1

    return counts


def eval_deck_bits_on_gds_files(
    deck_path: Path,
    gds_paths: list[Path],
    *,
    klayout_bin: str = "klayout",
    out_dir: Path | None = None,
) -> dict[str, int | None]:
    """
    Run deck on each GDS and return {gds_name: bit} where:
      - 0 => PASS (no violations)
      - 1 => VIOLATION (any category has any item)
      - None => ERROR (KLayout failed / no rdb)

    If out_dir is None, uses a temp dir so no .lyrdb files are persisted.
    If out_dir is provided, writes:
      - <out_dir>/lyrdb/*.lyrdb
    """
    if out_dir is None:
        import tempfile
        with tempfile.TemporaryDirectory(prefix="klayout_bits_") as td:
            return eval_deck_bits_on_gds_files(deck_path, gds_paths, klayout_bin=klayout_bin, out_dir=Path(td))

    out_dir.mkdir(parents=True, exist_ok=True)
    lyrdb_dir = out_dir / "lyrdb"

    out: dict[str, int | None] = {}
    for i, g in enumerate(gds_paths):
        key = g.name
        rdb = lyrdb_dir / f"{i:04d}_{g.stem}.lyrdb"
        ok = run_klayout(klayout_bin, deck_path, g, rdb)
        if not ok:
            out[key] = None
            continue
        counts = parse_rdb_counts(rdb)
        out[key] = 1 if any(int(v) > 0 for v in counts.values()) else 0
    return out


def _read_csv_header(path: Path) -> list[str]:
    with open(path, newline="") as f:
        r = csv.reader(f)
        return next(r)


def read_labels_csv(labels_csv: Path, categories: list[str]) -> dict[str, dict[str, int]]:
    gt: dict[str, dict[str, int]] = {}
    with open(labels_csv, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            fname = row["filename"]
            gt[fname] = {c: int(row.get(c, "0")) for c in categories}
    return gt


def f1_from_counts(y_true: list[int], y_pred: list[int]) -> dict[str, float | int]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec, "f1": f1, "accuracy": acc}


def write_predicted_labels_csv(out_csv: Path, rows: list[tuple[str, dict[str, int]]], categories: list[str]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", *categories])
        for rel_fname, preds in rows:
            w.writerow([rel_fname, *[preds.get(c, 0) for c in categories]])


def eval_deck_on_gds_dir(
    problem_dir: Path,
    deck_path: Path,
    gds_dir: Path | None = None,
    out_dir: Path | None = None,
    klayout_bin: str = "klayout",
    show_progress: bool = True,
) -> dict:
    """
    Evaluate a DRC deck against a directory of GDS testcases + labels.csv.
    (unchanged; requires labels.csv)
    """
    if out_dir is None:
        import tempfile
        with tempfile.TemporaryDirectory(prefix="klayout_eval_") as td:
            return eval_deck_on_gds_dir(problem_dir, deck_path, gds_dir, Path(td), klayout_bin, show_progress)

    out_dir.mkdir(parents=True, exist_ok=True)

    spec = yaml.safe_load((problem_dir / "spec.yaml").read_text())
    pid = spec.get("id", problem_dir.name)
    title = spec.get("title", pid)
    cats = list(spec["categories"])
    spec_cats = set(cats)

    root = Path.cwd().resolve()
    use_spec_layout = gds_dir is None
    if use_spec_layout:
        assert spec["data_dir"] is not None
        gds_dir = root / spec["data_dir"]

    labels_csv = gds_dir / "labels.csv"
    if not labels_csv.exists():
        raise FileNotFoundError(labels_csv)

    header = _read_csv_header(labels_csv)
    label_cols = {h for h in header if h != "filename"}
    precheck: dict[str, list[str]] = {}
    if missing := sorted(spec_cats - label_cols):
        precheck["labels_missing"] = missing
    if extra := sorted(label_cols - spec_cats):
        precheck["labels_extra"] = extra

    gt = read_labels_csv(labels_csv, cats)

    files = sorted(p for p in gds_dir.rglob("*.gds") if p.is_file())
    if not files:
        payload = {
            "id": pid, "title": title, "deck": str(deck_path), "gds_dir": str(gds_dir),
            "success": False, "n_cases": 0, "compile_rate": 0.0,
            "precheck": {"no_gds_files": [str(gds_dir)]}, "mismatches": {},
        }
        (out_dir / "eval_results.json").write_text(json.dumps(payload, indent=2) + "\n")
        return payload

    lyrdb_dir = out_dir / "lyrdb"
    pred_rows: list[tuple[str, dict[str, int]]] = []

    compiled = 0
    T: dict[str, list[int]] = {c: [] for c in cats}
    P: dict[str, list[int]] = {c: [] for c in cats}
    unknown_deck_cats: set[str] = set()
    per_file_mismatches: dict[str, list[tuple[str, int, int]]] = {}

    for g in tqdm(files, desc=f"[{pid}] {title}", leave=False, disable=not show_progress):
        if use_spec_layout:
            rel_g = g.resolve().relative_to(root).as_posix()
            rdb = lyrdb_dir / (g.stem + ".lyrdb")
        else:
            rel_g = g.relative_to(gds_dir).as_posix()
            rdb = (lyrdb_dir / rel_g).with_suffix(".lyrdb")

        ok = run_klayout(klayout_bin, deck_path, g, rdb)
        compiled += int(ok)
        counts = parse_rdb_counts(rdb) if ok else {}

        unknown_deck_cats.update(set(counts.keys()) - spec_cats)

        row = gt.get(rel_g)
        if row is None:
            precheck.setdefault("labels_missing_rows", []).append(rel_g)
            row = {c: 0 for c in cats}

        preds_for_file: dict[str, int] = {}
        for c in cats:
            t = int(row.get(c, 0))
            p = 1 if counts.get(c, 0) > 0 else 0
            preds_for_file[c] = p
            T[c].append(t)
            P[c].append(p)
            if t != p:
                per_file_mismatches.setdefault(rel_g, []).append((c, t, p))

        pred_rows.append((rel_g, preds_for_file))

    write_predicted_labels_csv(out_dir / "predicted_labels.csv", pred_rows, cats)

    success = True
    for c in cats:
        success &= (f1_from_counts(T[c], P[c])["f1"] == 1.0)

    if unknown_deck_cats:
        precheck["deck_unknown"] = sorted(unknown_deck_cats)
    if precheck or per_file_mismatches:
        success = False

    payload = {
        "id": pid,
        "title": title,
        "deck": str(deck_path),
        "gds_dir": str(gds_dir),
        "success": bool(success),
        "n_cases": len(files),
        "compile_rate": compiled / len(files),
        "precheck": precheck,
        "mismatches": per_file_mismatches,
    }
    (out_dir / "eval_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload
