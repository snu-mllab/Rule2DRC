import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


EVAL = Path(__file__).resolve().parents[1] / "eval.py"


@unittest.skipUnless(os.name == "posix", "requires a POSIX executable KLayout stub")
class EvalCliTests(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        stub = self.root / "bin/klayout"
        stub.parent.mkdir()
        stub.write_text(
            '#!/bin/sh\n'
            'printf "%s\\n" "$3" >> captured_decks.txt\n'
            'test -f "$3" || exit 23\n'
            'for arg do\n'
            '  case "$arg" in\n'
            '    input=*/b.gds) exit 24 ;;\n'
            '    output=*) out="${arg#output=}" ;;\n'
            '  esac\n'
            'done\n'
            'printf "%s\\n" "<report-database><items><item><category>width</category></item></items></report-database>" > "$out"\n',
            encoding="utf-8",
        )
        stub.chmod(0o700)
        self.env = {**os.environ, "PATH": str(stub.parent) + os.pathsep + os.environ.get("PATH", "")}
        (self.root / "override.drc").touch()
        for pid in ("0001", "0002"):
            problem = self.root / "problems" / pid
            (problem / "data/fail").mkdir(parents=True)
            (problem / "data/fail/a.gds").touch()
            (problem / "data/labels.csv").write_text(f"filename,width\nproblems/{pid}/data/fail/a.gds,1\n")
            (problem / "default.drc").touch()
            (problem / "spec.yaml").write_text(
                f"id: '{pid}'\ntitle: Synthetic CLI fixture\ncategories: [width]\n"
                f"deck: problems/{pid}/default.drc\ndata_dir: problems/{pid}/data\n"
            )
            directory_deck = self.root / "decks" / pid / f"{pid}.drc"
            directory_deck.parent.mkdir(parents=True)
            directory_deck.touch()

    def cli(self, *args):
        return subprocess.run(
            [sys.executable, str(EVAL), "--dir", "problems", *args],
            cwd=self.root, env=self.env, capture_output=True, text=True, timeout=30,
        )

    def assert_decks(self, *expected):
        self.assertCountEqual((self.root / "captured_decks.txt").read_text().splitlines(), expected)

    def payload(self, run, deck_dir):
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        result_file = self.root / "out_eval/problems" / deck_dir / "eval_results.json"
        payload = json.loads(result_file.read_text())
        self.assertEqual(set(payload), {"success_rate", "results"})
        for result in payload["results"]:
            self.assertEqual(set(result), {
                "id", "title", "deck", "n_cases", "compile_rate", "per_category", "precheck", "success",
            })
        return payload

    def assert_success(self, payload, *ids):
        self.assertEqual(payload["success_rate"], 1.0)
        self.assertEqual([result["id"] for result in payload["results"]], list(ids))
        for result in payload["results"]:
            self.assertEqual(result["compile_rate"], 1.0)
            self.assertTrue(result["success"])

    def test_file_override_sequential_and_output_placement(self):
        run = self.cli("--problem", "0001", "--jobs", "1", "--rule", "override.drc")
        self.assert_decks("override.drc")
        payload = self.payload(run, "override.drc")
        self.assert_success(payload, "0001")
        self.assertEqual(payload["results"][0]["deck"], "override.drc")
        self.assertTrue((self.root / "out_eval/problems/override.drc/0001/a.lyrdb").is_file())

    def test_shared_file_override_with_spawned_workers(self):
        run = self.cli("--jobs", "2", "--rule", "override.drc")
        self.assert_decks("override.drc", "override.drc")
        self.assert_success(self.payload(run, "override.drc"), "0001", "0002")

    def test_directory_override_preserves_per_problem_decks(self):
        run = self.cli("--jobs", "2", "--rule", "decks")
        self.assert_decks("decks/0001/0001.drc", "decks/0002/0002.drc")
        self.assert_success(self.payload(run, "decks"), "0001", "0002")

    def test_omitted_override_uses_gold_decks(self):
        run = self.cli("--jobs", "1")
        self.assert_decks(*(str(self.root / "problems" / pid / "default.drc") for pid in ("0001", "0002")))
        self.assert_success(self.payload(run, "gold"), "0001", "0002")

    def test_invalid_override_is_rejected_before_evaluation(self):
        run = self.cli("--jobs", "1", "--rule", "missing.drc")
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("--rule must point to an existing file or directory: missing.drc", run.stderr)
        self.assertFalse((self.root / "captured_decks.txt").exists())
        self.assertFalse((self.root / "out_eval").exists())

    def test_failed_testcase_is_unsuccessful_even_with_perfect_f1(self):
        data = self.root / "problems/0001/data"
        (data / "fail/b.gds").touch()
        with (data / "labels.csv").open("a") as labels:
            labels.write("problems/0001/data/fail/b.gds,0\n")
        run = self.cli("--jobs", "1", "--problem", "0001", "--rule", "decks")
        self.assert_decks("decks/0001/0001.drc", "decks/0001/0001.drc")
        payload = self.payload(run, "decks")
        result, = payload["results"]
        self.assertEqual(result["compile_rate"], 0.5)
        self.assertEqual(result["per_category"]["width"]["f1"], 1.0)
        self.assertFalse(result["success"])
        self.assertEqual(payload["success_rate"], 0.0)
