#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

import yaml


HF_DATASET = os.environ.get("RULE2DRC_HF_DATASET", "jusjinuk/Rule2DRC")


def materialize_if_missing(problems_dir: str | Path = "problems") -> Path:
    problems_dir = Path(problems_dir).resolve()
    if problems_dir.is_dir() and any(problems_dir.glob("*/spec.yaml")):
        return problems_dir

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            f"Problems directory not found: {problems_dir}\n"
            "Install the 'datasets' package or create the local problems directory."
        ) from exc

    tasks = load_dataset(HF_DATASET, "tasks", split="test")
    testcases = load_dataset(HF_DATASET, "testcases", split="test")

    by_problem: dict[str, list[dict]] = {}
    for row in testcases:
        row = dict(row)
        by_problem.setdefault(str(required(row, "problem_id")), []).append(row)

    for row in tasks:
        row = dict(row)
        pid = str(required(row, "problem_id"))
        if pid not in by_problem:
            raise SystemExit(f"No testcases found for problem_id={pid}")
        _write_problem(problems_dir, row, by_problem[pid])

    return problems_dir


def _write_problem(root: Path, task: dict, testcases: list[dict]) -> None:
    pid = str(required(task, "problem_id"))
    problem_dir = root / pid
    gds_dir = problem_dir / "data" / "gds"
    gold_dir = problem_dir / "gold"
    gds_dir.mkdir(parents=True, exist_ok=True)
    gold_dir.mkdir(parents=True, exist_ok=True)

    spec_yaml = required_str(task, "spec_yaml")
    spec = yaml.safe_load(spec_yaml)
    if not isinstance(spec, dict):
        raise SystemExit(f"Invalid spec_yaml for problem_id={pid}")
    (problem_dir / "spec.yaml").write_bytes(spec_yaml.encode("utf-8"))

    (gold_dir / f"{pid}.drc").write_bytes(required_str(task, "gold_drc").encode("utf-8"))
    (gds_dir / "labels.csv").write_bytes(required_str(task, "labels_csv").encode("utf-8"))

    for i, case in enumerate(testcases):
        rel = _case_path(case, i)
        data = case["gds"] if "gds" in case else case["gds_bytes"] if "gds_bytes" in case else None
        if not isinstance(data, (bytes, bytearray)):
            raise SystemExit(f"Missing binary GDS data for {pid}/{rel}")
        (gds_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        (gds_dir / rel).write_bytes(bytes(data))


def _case_path(case: dict, index: int) -> str:
    if "gds_path" in case:
        path = str(case["gds_path"])
    elif "filename" in case:
        path = str(case["filename"])
    else:
        raise SystemExit(f"Missing gds_path for testcase index {index}")
    path = path.replace("\\", "/").lstrip("./")
    if "data/gds/" in path:
        path = path.split("data/gds/", 1)[1]
    if "/" not in path:
        if "split_hint" not in case:
            raise SystemExit(f"Missing split_hint for testcase {path}")
        split = str(case["split_hint"])
        path = f"{split}/{path}"
    return path


def required(row: dict, key: str):
    if key not in row or row[key] is None:
        raise SystemExit(f"Missing required column: {key}")
    return row[key]


def required_str(row: dict, key: str) -> str:
    value = required(row, key)
    if not isinstance(value, str):
        raise SystemExit(f"Column {key} must be a string")
    return value


if __name__ == "__main__":
    print(materialize_if_missing())
