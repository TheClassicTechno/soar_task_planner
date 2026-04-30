"""
KnowNo Navigation Baseline Runner

Two-phase execution:
  Phase 1 — Calibration: run LLM on calibration scenarios, compute tau.
  Phase 2 — Evaluation:  run LLM on test scenarios, apply threshold, report metrics.

Usage:
    python -m baselines.knowno.run_baseline \\
        --config baselines/knowno/config.yaml \\
        --calib_data baselines/introplan/data/nav_calibration.json \\
        --test_data  baselines/introplan/data/nav_test.json

    # Load pre-calibrated predictor (skips Phase 1):
    python -m baselines.knowno.run_baseline \\
        --config baselines/knowno/config.yaml \\
        --load_predictor outputs/knowno_baseline/predictor.json

Requirements:
    - ANTHROPIC_API_KEY or OPENAI_API_KEY set in .env
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
from baselines.knowno.runner import KnowNoRunner


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run KnowNo navigation baseline")
    p.add_argument(
        "--config", default="baselines/knowno/config.yaml",
        help="Path to KnowNo config YAML",
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
        "--output_dir", default="outputs/knowno_baseline",
        help="Directory for results and predictor state",
    )
    p.add_argument(
        "--load_predictor", default=None,
        help="Path to a saved predictor JSON (skips calibration phase)",
    )
    return p.parse_args()


def run(args: argparse.Namespace) -> None:
    load_dotenv()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    api_type = config.get("llm", {}).get("api_type", "openai")
    env_var = "OPENAI_API_KEY" if api_type == "openai" else "ANTHROPIC_API_KEY"
    api_key = os.environ.get(env_var)
    if not api_key:
        raise ValueError(f"Set {env_var} in your .env file")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    llm = LLMInterface(
        api_key=api_key,
        model=config.get("llm", {}).get("model", "gpt-4o-mini"),
        max_tokens=config.get("llm", {}).get("max_tokens", 512),
        temperature=config.get("llm", {}).get("temperature", 0.0),
        api_type=api_type,
    )

    runner = KnowNoRunner(config_path=args.config, llm=llm)

    # ── Phase 1: Load predictor or calibrate ──────────────────────────────────
    if args.load_predictor:
        print(f"[KnowNo] Loading calibrated predictor from: {args.load_predictor}")
        runner._predictor = ConformalPredictor.load(args.load_predictor)
        print(f"  tau = {runner._predictor.tau:.4f}  "
              f"(from {runner._predictor.n_calibration} calibration scenarios)")
    else:
        calib_scenarios = load_scenarios_from_json(args.calib_data)
        alpha = config.get("conformal", {}).get("alpha", 0.15)
        print(f"\n[KnowNo] Phase 1 — Calibration")
        print(f"  Calibration scenarios : {len(calib_scenarios)}")
        print(f"  Target coverage       : {(1 - alpha) * 100:.0f}%")

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

    # ── Phase 2: Evaluation ───────────────────────────────────────────────────
    test_scenarios = load_scenarios_from_json(args.test_data)
    print(f"\n[KnowNo] Phase 2 — Evaluation")
    print(f"  Test scenarios        : {len(test_scenarios)}")

    decisions = []
    calc = MetricsCalculator()

    for scenario in tqdm(test_scenarios, desc="Evaluating", unit="scenario"):
        decision = runner.run_scenario(scenario)
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
            "prediction_set": decision.prediction_set,
            "option_confidences": decision.option_confidences,
            "correct": result.correct,
            "asked_human": result.asked_human,
            "llm_raw": decision.llm_raw,
        })

    metrics = calc.summary()
    metrics["tau"] = runner._predictor.tau
    metrics["alpha"] = runner._predictor.alpha
    metrics["n_calibration"] = runner._predictor.n_calibration
    metrics["total_llm_calls"] = llm.total_calls

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump({"metrics": metrics, "decisions": decisions}, f, indent=2)

    print(f"\n{'='*50}")
    print(f"KnowNo Baseline Results")
    print(f"{'='*50}")
    print(f"  Test scenarios     : {metrics['n_scenarios']}")
    print(f"  Calibrated tau     : {metrics['tau']:.4f}")
    print(f"  SR  (Success Rate) : {metrics['SR']:.4f}")
    print(f"  HR  (Human Help %) : {metrics['HR']:.4f}")
    print(f"  FPR (Over-asking)  : {metrics['FPR']:.4f}")
    print(f"  NCR (Non-compliant): {metrics['NCR']:.4f}")
    print(f"  ESR (Exact Set)    : {metrics['ESR']:.4f}")
    print(f"  Total LLM calls    : {metrics['total_llm_calls']}")
    print(f"\n  Results saved to   : {results_path}")


if __name__ == "__main__":
    run(parse_args())
