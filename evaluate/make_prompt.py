#!/usr/bin/env python3
import sys, yaml
from pathlib import Path

import jinja2

ROOT = Path(".").resolve()
TPL = (ROOT / "evaluate" / "prompt_template.j2").read_text()


def find_problem_dir(problems_root, problem_id):
    problems_root = Path(problems_root)
    if (problems_root / problem_id).is_dir():
        return problems_root / problem_id
    matches = sorted(p for p in problems_root.iterdir() if p.is_dir() and p.name.startswith(problem_id))
    if problem_id.isdigit():
        exact = [p for p in matches if p.name.split("_")[0] == problem_id]
        if exact:
            return exact[0]
    if matches:
        return matches[0]
    print(f"No problem directory found for '{problem_id}' in '{problems_root}'")
    sys.exit(1)


def render_prompt(problem_dir):
    spec = yaml.safe_load((problem_dir / "spec.yaml").read_text())
    ctx = {
        "title": spec.get("title", spec["id"]),
        "nl_description": spec.get("nl_description", ""),
        "layers": spec.get("layers", {}),
        "rules": spec.get("rules", [{"text": c, "category": c} for c in spec.get("categories", [])]),
    }
    tpl = jinja2.Template(TPL)
    return tpl.render(**ctx)


def main():
    if len(sys.argv) != 3:
        print("Usage: python make_prompt.py <problems_dir> <problem_id>")
        sys.exit(1)
    problems_dir = Path(sys.argv[1])
    prob = sys.argv[2]
    problem_dir = find_problem_dir(problems_dir, prob)
    print(render_prompt(problem_dir))


if __name__ == "__main__":
    main()
