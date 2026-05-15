"""
When-to-Ask (UPS) Navigation Baseline Runner

Two-phase execution:
  Phase 1 — Calibration: Score calibration scenarios, compute tau.
  Phase 2 — Evaluation:  Score test scenarios, apply calibrated threshold,
                          report metrics + strategy breakdown.

Usage:
    # Standard run:
    python -m baselines.whentoask.run_baseline \
        --config baselines/whentoask/config.yaml \
        --calib_data baselines/introplan/data/nav_calibration.json \
        --test_data  baselines/introplan/data/nav_test.json

    # Disable intent factorization (single-call mode):
    python -m baselines.whentoask.run_baseline \
        --config baselines/whentoask/config.yaml \
        --no_factorization

    # Load pre-calibrated predictor:
    python -m baselines.whentoask.run_baseline \
        --config baselines/whentoask/config.yaml \
        --load_predictor outputs/whentoask_baseline/predictor.json

Requirements:
    - ANTHROPIC_API_KEY (or OPENAI_API_KEY) set in .env
    - pip install -r requirements.txt
"""

import argparse
import json
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from baselines.introplan.conformal_predictor import ConformalPredictor
from baselines.introplan.llm_interface import LLMInterface
from baselines.introplan.metrics import MetricsCalculator, ScenarioResult
from baselines.introplan.runner import load_scenarios_from_json
from baselines.whentoask.runner import WhenToAskRunner


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run When-to-Ask (UPS) navigation baseline")
    p.add_argument(
        "--config", default="baselines/whentoask/config.yaml",
        help="Path to When-to-Ask config YAML",
    )
    p.add_argument(
        "--calib_data",
        default="baselines/introplan/data/nav_calibration.json",
        help="JSON file with labeled calibration scenarios",
    )
    p.add_argument(
        "--test_data",
        default="baselines/introplan/data/nav_test.json",
        help="JSON file with labeled test scenarios",
    )
    p.add_argument(
        "--output_dir", default="outputs/whentoask_baseline",
        help="Directory for results, decisions, and predictor state",
    )
    p.add_argument(
        "--load_predictor", default=None,
        help="Path to a saved predictor JSON (skips calibration phase)",
    )
    p.add_argument(
        "--no_factorization", action="store_true",
        help="Disable Bayesian intent factorization (single-call scoring mode)",
    )
    return p.parse_args()


def run(args: argparse.Namespace) -> None:
    load_dotenv()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.no_factorization:
        config.setdefault("factorization", {})["enabled"] = False

    api_type = config.get("llm", {}).get("api_type", "anthropic")
    env_var = "OPENAI_API_KEY" if api_type == "openai" else "ANTHROPIC_API_KEY"
    api_key = os.environ.get(env_var)
    if not api_key:
        raise ValueError(f"Set {env_var} in your .env file")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    llm = LLMInterface(
        api_key=api_key,
        model=config.get("llm", {}).get("model", "gpt-4o-mini"),
        max_tokens=config.get("llm", {}).get("max_tokens", 768),
        temperature=config.get("llm", {}).get("temperature", 0.0),
        api_type=api_type,
    )

    # Write updated config so WhenToAskRunner picks up no_factorization override
    tmp_config_path = str(output_dir / "effective_config.yaml")
    with open(tmp_config_path, "w") as f:
        yaml.dump(config, f)

    runner = WhenToAskRunner(config_path=tmp_config_path, llm=llm)

    fact_enabled = config.get("factorization", {}).get("enabled", True)
    mode_label = "intent factorization" if fact_enabled else "direct scoring"

    # ── Phase 1: Load predictor or run calibration ────────────────────────────
    if args.load_predictor:
        print(f"[WhenToAsk] Loading calibrated predictor from: {args.load_predictor}")
        runner._predictor = ConformalPredictor.load(args.load_predictor)
        print(f"  tau = {runner._predictor.tau:.4f}  "
              f"(from {runner._predictor.n_calibration} calibration scenarios)")
    else:
        calib_scenarios = load_scenarios_from_json(args.calib_data)
        print(f"\n[WhenToAsk] Phase 1 — Calibration ({mode_label})")
        print(f"  Calibration scenarios : {len(calib_scenarios)}")
        print(f"  Target coverage       : {(1 - config.get('conformal', {}).get('alpha', 0.15))*100:.0f}%")

        for scenario in tqdm(calib_scenarios, desc="Calibrating", unit="scenario"):
            decision = runner.run_scenario(scenario)
            runner._predictor.record_calibration(
                option_confidences=decision.option_confidences,
                correct_option=scenario.correct_option,
            )
        tau = runner._predictor.calibrate()
        print(f"  Computed tau          : {tau:.4f}")

        if config.get("output", {}).get("save_predictor", True):
            predictor_path = str(output_dir / "predictor.json")
            runner._predictor.save(predictor_path)
            print(f"  Predictor saved to    : {predictor_path}")

    # ── Phase 2: Evaluation ────────────────────────────────────────────────────
    test_scenarios = load_scenarios_from_json(args.test_data)
    print(f"\n[WhenToAsk] Phase 2 — Evaluation ({mode_label})")
    print(f"  Test scenarios        : {len(test_scenarios)}")

    decisions = []
    calc = MetricsCalculator()
    strategy_counts = {"EXECUTE": 0, "CLARIFY": 0, "INCAPABLE": 0}

    for scenario in tqdm(test_scenarios, desc="Evaluating", unit="scenario"):
        decision = runner.run_scenario(scenario)
        strategy_counts[decision.resolution_strategy] += 1

        result = ScenarioResult(
            scenario_id=scenario.scenario_id,
            correct_option=scenario.correct_option,
            prediction_set=decision.prediction_set,
            robot_decision=decision.robot_decision,
        )
        calc.add(result)
        decisions.append({
            "scenario_id": scenario.scenario_id,
            "instruction": scenario.instruction,
            "terrain_description": scenario.terrain_description,
            "correct_option": scenario.correct_option,
            "robot_decision": decision.robot_decision,
            "resolution_strategy": decision.resolution_strategy,
            "prediction_set": decision.prediction_set,
            "option_scores": decision.option_scores,
            "option_confidences": decision.option_confidences,
            "intents": decision.intents,
            "correct": result.correct,
            "asked_human": result.asked_human,
        })

    metrics = calc.summary()
    metrics["tau"] = runner._predictor.tau
    metrics["alpha"] = runner._predictor.alpha
    metrics["n_calibration"] = runner._predictor.n_calibration
    metrics["total_llm_calls"] = llm.total_calls
    metrics["strategy_counts"] = strategy_counts
    metrics["factorization_enabled"] = fact_enabled

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump({"metrics": metrics, "decisions": decisions}, f, indent=2)

    print(f"\n{'='*55}")
    print(f"When-to-Ask (UPS) Baseline Results")
    print(f"{'='*55}")
    print(f"  Test scenarios     : {metrics['n_scenarios']}")
    print(f"  Calibrated tau     : {metrics['tau']:.4f}")
    print(f"  SR  (Success Rate) : {metrics['SR']:.4f}")
    print(f"  HR  (Human Help %) : {metrics['HR']:.4f}")
    print(f"  FPR (Over-asking)  : {metrics['FPR']:.4f}")
    print(f"  NCR (Non-compliant): {metrics['NCR']:.4f}")
    print(f"  ESR (Exact Set)    : {metrics['ESR']:.4f}")
    print(f"  Total LLM calls    : {metrics['total_llm_calls']}")
    print(f"\n  Strategy breakdown:")
    print(f"    EXECUTE   : {strategy_counts['EXECUTE']}")
    print(f"    CLARIFY   : {strategy_counts['CLARIFY']}")
    print(f"    INCAPABLE : {strategy_counts['INCAPABLE']}")
    print(f"\n  Results saved to   : {results_path}")


if __name__ == "__main__":
    run(parse_args())
