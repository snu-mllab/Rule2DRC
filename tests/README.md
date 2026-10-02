# Evaluator regression tests

From the repository root, using Python 3.12 or later:

```sh
python -m pip install PyYAML tqdm
python -m unittest discover -s tests -v
```

Eight unit tests replace `run_klayout` and create synthetic XML reports. Their
GDS files are empty placeholders. These tests exercise directory evaluation,
including a failed execution on an all-zero label row, the default spec data
path, multiple categories, label mismatches, and unchanged successful behavior.

Two integration tests run real KLayout on synthetic 0.1 and 0.4 micrometer-wide
rectangles with a 0.2 micrometer width rule. One uses a normal deck; the other
deliberately raises a runtime error on the passing geometry. Report item counts
are also checked through KLayout's report API. These decks are test fixtures.

Install the KLayout command-line runtime as described in `KLAYOUT.md` and its
Python bindings, then set `KLAYOUT_BIN` to the executable. Without that variable,
the integration tests are skipped.

```sh
python -m pip install klayout==0.30.5
KLAYOUT_BIN=/absolute/path/to/klayout python -m unittest discover -s tests -v
```

PowerShell:

```powershell
$env:KLAYOUT_BIN = 'C:\absolute\path\to\klayout_app.exe'
python -m unittest discover -s tests -v
```
