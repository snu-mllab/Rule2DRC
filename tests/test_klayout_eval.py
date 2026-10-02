"""Evaluator regressions; real KLayout cases require KLAYOUT_BIN."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluate import klayout_eval


class DirectoryEvaluationTests(unittest.TestCase):
    def evaluate(self, statuses, labels, predictions, categories=("WIDTH",),
                 use_spec_layout=False):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "data"
            data.mkdir()
            root.joinpath("spec.yaml").write_text(
                "id: synthetic\ndata_dir: data\ncategories: ["
                + ", ".join(categories) + "]\n", encoding="utf-8"
            )
            rows = ["filename," + ",".join(categories)]
            for index, label in enumerate(labels):
                data.joinpath(f"{index}.gds").touch()
                filename = f"data/{index}.gds" if use_spec_layout else f"{index}.gds"
                rows.append(filename + "," + ",".join(map(str, label)))
            data.joinpath("labels.csv").write_text("\n".join(rows) + "\n")

            def run(_binary, _deck, gds, report):
                index = int(gds.stem)
                if statuses[index]:
                    report.parent.mkdir(parents=True, exist_ok=True)
                    items = "".join(
                        f"<item><category>{category}</category></item>"
                        for category, bit in zip(categories, predictions[index]) if bit
                    )
                    report.write_text(
                        "<report-database><items>" + items
                        + "</items></report-database>"
                    )
                return statuses[index]

            with patch.object(klayout_eval, "run_klayout", side_effect=run) as runner:
                previous = Path.cwd()
                try:
                    if use_spec_layout:
                        os.chdir(root)
                    result = klayout_eval.eval_deck_on_gds_dir(
                        root, root / "synthetic.drc",
                        None if use_spec_layout else data,
                        root / "out", show_progress=False,
                    )
                finally:
                    os.chdir(previous)
                self.assertEqual(runner.call_count, len(labels))
            return result

    def test_failed_zero_label_case_cannot_succeed(self):
        result = self.evaluate([True, False], [[1], [0]], [[1], [0]])
        self.assertFalse(result["success"])
        self.assertEqual(result["compile_rate"], 0.5)
        self.assertEqual(result["mismatches"], {})
        self.assertEqual(result["precheck"], {})

    def test_successful_matching_executions_still_succeed(self):
        result = self.evaluate([True, True], [[1], [0]], [[1], [0]])
        self.assertTrue(result["success"])
        self.assertEqual(result["compile_rate"], 1.0)

    def test_failed_positive_label_case_remains_unsuccessful(self):
        result = self.evaluate([False, True], [[1], [0]], [[1], [0]])
        self.assertFalse(result["success"])
        self.assertEqual(result["compile_rate"], 0.5)
        self.assertIn("0.gds", result["mismatches"])

    def test_all_failed_executions_remain_unsuccessful(self):
        result = self.evaluate([False, False], [[1], [0]], [[1], [0]])
        self.assertFalse(result["success"])
        self.assertEqual(result["compile_rate"], 0.0)

    def test_successful_execution_with_wrong_label_remains_unsuccessful(self):
        result = self.evaluate([True, True], [[1], [0]], [[0], [0]])
        self.assertFalse(result["success"])
        self.assertEqual(result["compile_rate"], 1.0)

    def test_failure_with_multiple_categories_cannot_succeed(self):
        result = self.evaluate(
            [True, False], [[1, 1], [0, 0]], [[1, 1], [0, 0]],
            categories=("WIDTH", "SPACE"),
        )
        self.assertFalse(result["success"])
        self.assertEqual(result["mismatches"], {})

    def test_default_spec_data_directory_checks_execution_status(self):
        result = self.evaluate(
            [True, False], [[1], [0]], [[1], [0]], use_spec_layout=True,
        )
        self.assertFalse(result["success"])
        self.assertEqual(result["compile_rate"], 0.5)
        self.assertEqual(result["precheck"], {})

    def test_existing_all_negative_f1_behavior_is_preserved(self):
        result = self.evaluate([True, True], [[0], [0]], [[0], [0]])
        self.assertFalse(result["success"])
        self.assertEqual(result["compile_rate"], 1.0)
        self.assertEqual(result["mismatches"], {})


@unittest.skipUnless(os.environ.get("KLAYOUT_BIN"), "Set KLAYOUT_BIN for real execution")
class KLayoutExecutionTests(unittest.TestCase):
    def setUp(self):
        import klayout.db as db

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        self.binary = str(Path(os.environ["KLAYOUT_BIN"]).resolve())
        home = self.root / "klayout_home"
        home.mkdir()
        environment = patch.dict(os.environ, {
            "KLAYOUT_HOME": str(home),
            "KLAYOUT_PATH": str(Path(self.binary).parent),
        })
        environment.start()
        self.addCleanup(environment.stop)
        self.root.joinpath("spec.yaml").write_text(
            "id: synthetic_width\ncategories: [WIDTH]\n"
        )
        for name, width in (("fail.gds", 100), ("pass.gds", 400)):
            layout = db.Layout()
            layout.dbu = 0.001
            top = layout.create_cell("TOP")
            top.shapes(layout.layer(1, 0)).insert(db.Box(0, 0, width, 1000))
            layout.write(str(self.data / name))
        self.data.joinpath("labels.csv").write_text(
            "filename,WIDTH\nfail.gds,1\npass.gds,0\n"
        )

    def evaluate(self, fail_on_passing_layout=False):
        deck = self.root / "synthetic.drc"
        source = 'source($input)\nreport("Synthetic width", $output)\nmetal = input(1, 0)\n'
        if fail_on_passing_layout:
            source += 'raise "Deliberate test failure" if metal.data.bbox.width >= 400\n'
        source += 'metal.width(0.2).output("WIDTH", "Minimum width 0.2 micrometers")\n'
        deck.write_text(source)
        return klayout_eval.eval_deck_on_gds_dir(
            self.root, deck, self.data, self.root / "out",
            klayout_bin=self.binary, show_progress=False,
        )

    def assert_report_count(self, name, expected):
        import klayout.rdb as rdb

        report = rdb.ReportDatabase()
        report.load(str(self.root / "out" / "lyrdb" / f"{name}.lyrdb"))
        self.assertEqual(report.num_items(), expected)

    def test_normal_width_deck_succeeds(self):
        result = self.evaluate()
        self.assertTrue(result["success"])
        self.assertEqual(result["compile_rate"], 1.0)
        self.assert_report_count("fail", 1)
        self.assert_report_count("pass", 0)

    def test_runtime_failure_on_zero_label_layout_is_unsuccessful(self):
        result = self.evaluate(fail_on_passing_layout=True)
        self.assertFalse(result["success"])
        self.assertEqual(result["compile_rate"], 0.5)
        self.assertEqual(result["mismatches"], {})
        self.assertEqual(result["precheck"], {})
        self.assert_report_count("fail", 1)


if __name__ == "__main__":
    unittest.main()
