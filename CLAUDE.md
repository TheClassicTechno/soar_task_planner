# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Environment setup
conda env create -f environment.yaml
conda activate soar-task-planner
pip install -r requirements.txt

# Run all tests
pytest

# Run tests for a specific module
pytest system/env_uncertainty/tests/
pytest system/instruction_uncertainty/tests/
pytest baselines/introplan/tests/

# Run a single test file
pytest baselines/introplan/tests/test_runner.py -v

# Run a specific test
pytest baselines/introplan/tests/test_runner.py::test_foo -v

# Run a baseline (requires ANTHROPIC_API_KEY or OPENAI_API_KEY in .env)
python -m baselines.introplan.run_baseline --config baselines/introplan/config.yaml
python -m baselines.knowno.run_baseline --config baselines/knowno/config.yaml
python -m baselines.whentoask.run_baseline --config baselines/whentoask/config.yaml
python -m baselines.sam3.run_baseline --config baselines/sam3/config.yaml --rugd_dir /path/to/rugd --split val

# Type checking
mypy system/ baselines/
```

## Architecture

This is a research codebase for **uncertainty-aware outdoor robot navigation**. The core claim is that robots face two decoupled types of uncertainty that must be handled separately.

### Two uncertainty branches

**Instruction uncertainty** (`system/instruction_uncertainty/`): triggered by an ambiguous or incomplete user command. The `AmbiguityDetector` classifies commands into 6 types (missing action, ambiguous target, missing object, ambiguous action, missing direction, missing distance), each with a severity weight. Produces a non-conformity score κ_I for conformal prediction.

**Environmental uncertainty** (`system/env_uncertainty/`): detected autonomously by the robot's perception pipeline, without any user command. The `EnvironmentalUncertaintyDetector` runs SAM3 (known terrain labeling) then SAM2 (segment everything), and any SAM2 region with <30% overlap with SAM3 coverage is flagged "unknown". The `EnvironmentalUncertaintyRunner` orchestrates: detect → generate trajectories → score against `TraversabilityMap` → decide PROCEED/ASK/STOP → generate clarification question.

Both branches share a 6-step resolution pipeline defined in `docs/methodology.md`.

### Baselines (`baselines/`)

Each baseline is a self-contained directory with `runner.py`, `run_baseline.py`, and `config.yaml`. They all share infrastructure from `baselines/introplan/`:
- `LLMInterface` — wraps Anthropic and OpenAI APIs
- `ConformalPredictor` — calibrated conformal prediction (tau threshold)
- `MetricsCalculator` — SR/HR/FPR/NCR/ESR metrics
- `NavigationScenario` / `load_scenarios_from_json` — data loading

| Baseline | Key idea |
|---|---|
| `always_act` / `always_ask` | Trivial bounds (never ask / always ask) |
| `introplan` | Conformal prediction + K retrieved similar examples from knowledge base |
| `knowno` | Conformal prediction, no retrieval (single LLM call) |
| `whentoask` | Bayesian intent factorization over 5 options including "none of above" |
| `sam3` | SAM3 terrain segmentation on RUGD images, no ask/act decision |

### Data

`baselines/introplan/data/nav_calibration.json` and `nav_test.json` are the primary evaluation datasets. All 70 entries are currently synthetic (no real RUGD image pairing). Each entry has: `entry_id`, `instruction`, `terrain_description`, `uncertainty_type` (1–4), `options` (A/B/C/D), `correct_option`, `reasoning`, `source_image` (None = synthetic).

The 4 uncertainty types are documented in `docs/methodology.md`. Type 3/4 numbering between the code and the April 2025 meeting proposal is unresolved — do not renumber without Jing's sign-off.

### Test strategy

Tests use `pytest-mock` and never require GPU access or real API keys. Models (SAM3, SAM2, LLM) are always injected via constructor arguments so they can be replaced with mocks. Tests live alongside their modules in `tests/` subdirectories.
