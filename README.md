# Parameter-Information Active Learning for Non-Stationary GSMs

This project compares parameter-information active learning against a
predictive-variance baseline on a synthetic 2D geospatial field. The original
notebook has been refactored into a small Python package with a CLI, tests, and
a lightweight notebook entry point.

## Repository Layout

```text
.
+-- src/actlearn_gsm/          # Experiment package
|   +-- active_learning.py     # Active-learning loop
|   +-- cli.py                 # Command-line entry point
|   +-- config.py              # Dataclass configuration
|   +-- data.py                # Synthetic geospatial field
|   +-- ensemble.py            # GSM MAP ensemble and acquisitions
|   +-- experiment.py          # Full experiment orchestration
|   +-- kernels.py             # Non-stationary GSM kernel
|   +-- models.py              # GP models and SM baseline
|   +-- plotting.py            # Figure generation
|   +-- uncertainty.py         # Hyperparameter covariance summaries
+-- tests/                     # Smoke tests
+-- upload_final_actlearn.ipynb # Clean notebook driver
+-- pyproject.toml             # Package metadata
+-- requirements.txt           # Runtime dependencies
```

## Installation

Use a virtual environment, then install the project in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
```

If you only want to run the notebook, `pip install -r requirements.txt` is
enough when the notebook is opened from the repository root.

## Running the Experiment

After the editable install, run the default experiment:

```bash
python -m actlearn_gsm.cli
```

Or use the installed console script after `pip install -e .`:

```bash
actlearn-gsm
```

Without installing the package, run from the repository root with:

```bash
PYTHONPATH=src python -m actlearn_gsm.cli
```

The full default run is intentionally compute-heavy. For a fast smoke run:

```bash
PYTHONPATH=src python -m actlearn_gsm.cli \
  --n-lon 8 \
  --n-lat 8 \
  --num-init 10 \
  --num-steps 1 \
  --ensemble-size 2 \
  --num-inducing 4 \
  --num-mixtures 2 \
  --map-iters 2 \
  --sm-iters 2 \
  --no-plots
```

Generated figures are written to `artifacts/`.

## Notebook

Open `upload_final_actlearn.ipynb` from the repository root. It imports the
package from `src/`, defines the default configuration, and runs the same
experiment as the CLI.
