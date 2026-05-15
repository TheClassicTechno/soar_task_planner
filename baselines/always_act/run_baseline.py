"""
Always Act Navigation Baseline Runner

A simple baseline that always picks the highest-confidence option,
never asks the human (HR = 0). Serves as comparison for IntroPlan.

Usage:
    # Run with test_data:
    python -m baselines.always_act.run_baseline \
        --config baselines/always_act/config.yaml \
        --test_data baselines/introplan/data/nav_test.json

    # Run without test_data (uses calib_data split):
    python -m baselines.always_act.run_baseline \
        --config baselines/always_act/config.yaml

Requirements:
    - ANTHROPIC_API_KEY or OPENAI_API_KEY set in .env
"""

import argparse
import json
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from baselines.introplan.llm_interface import LLMInterface
from baselines.introplan.runner import load_scenarios_from_json
from baselines.introplan.metrics import MetricsCalculator, ScenarioResult
from baselines.always_act.runner import AlwaysActRunner


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Always Act baseline")
    p.add_argument(
        "--config",
        default="baselines/always_act/config.yaml",
        help="Path to config YAML",
    )
    p.add_argument(
        "--calib_data",
        default="baselines/introplan/data/nav_calibration.json",
        help="JSON file with scenarios (split for test if --test_data is None)",
    )
    p.add_argument(
        "--test_data",
        default=None,
        help="JSON file with test scenarios. If None, uses calib_data split.",
    )
    p.add_argument(
        "--calib_fraction",
        type=float,
        default=0.7,
        help="Fraction of calib_data to use for test if test_data is None",
    )
    p.add_argument(
        "--output_dir",
        default="outputs/always_act_baseline",
        help="Directory for results",
    )
    return p.parse_args()


def run(args: argparse.Namespace) -> None:
    load_dotenv()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    api_type = config.get("llm", {}).get("api_type", "anthropic")
    env_var = "OPENAI_API_KEY" if api_type == "openai" else "ANTHROPIC_API_KEY"
    api_key = os.environ.get(env_var)
    if not api_key:
        raise ValueError(f"Set {env_var} in your .env file")

    llm = LLMInterface(
        api_key=api_key,
        model=config.get("llm", {}).get("model", "gpt-4o"),
        max_tokens=config.get("llm", {}).get("max_tokens", 1024),
        temperature=config.get("llm", {}).get("temperature", 0.0),
        api_type=api_type,
    )

    from baselines.introplan.knowledge_base import NavigationKnowledgeBase
    kb_path = config.get("knowledge_base", {}).get(
        "path", "baselines/introplan/data/nav_calibration.json"
    )
    top_k = config.get("knowledge_base", {}).get("top_k", 3)
    kb = NavigationKnowledgeBase.from_json(kb_path)

    runner = AlwaysActRunner(
        config_path=args.config,
        llm=llm,
        knowledge_base=kb,
        top_k=top_k,
    )

    # Load scenarios
    if args.test_data:
        test_scenarios = load_scenarios_from_json(args.test_data)
    else:
        all_scenarios = load_scenarios_from_json(args.calib_data)
        split = int(len(all_scenarios) * args.calib_fraction)
        test_scenarios = all_scenarios[split:]

    print(f"[Always Act] Test scenarios: {len(test_scenarios)}")

    decisions = []
    calc = MetricsCalculator()

    for scenario in tqdm(test_scenarios, desc="Evaluating", unit="scenario"):
        decision = runner.run_scenario(scenario)

        result = ScenarioResult(
            scenario_id=scenario.scenario_id,
            correct_option=scenario.correct_option,
            prediction_set=[decision.robot_decision],
            robot_decision=decision.robot_decision,
        )
        calc.add(result)

        decisions.append({
            "scenario_id": scenario.scenario_id,
            "instruction": scenario.instruction,
            "terrain_description": scenario.terrain_description,
            "correct_option": scenario.correct_option,
            "robot_decision": decision.robot_decision,
            "prediction_set": [decision.robot_decision],
            "option_confidences": decision.option_confidences,
            "correct": result.correct,
            "asked_human": False,
        })

    metrics = calc.summary()
    metrics["total_llm_calls"] = llm.total_calls

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"

    with open(results_path, "w") as f:
        json.dump({"metrics": metrics, "decisions": decisions}, f, indent=2)

    print(f"\n{'='*55}")
    print(f"Always Act Baseline Results")
    print(f"{'='*55}")
    print(f"  Test scenarios     : {metrics['n_scenarios']}")
    print(f"  SR  (Success Rate) : {metrics['SR']:.4f}")
    print(f"  HR  (Human Help %) : {metrics['HR']:.4f}")
    print(f"  FPR (Over-asking)  : {metrics['FPR']:.4f}")
    print(f"  NCR (Non-compliant): {metrics['NCR']:.4f}")
    print(f"  ESR (Exact Set)    : {metrics['ESR']:.4f}")
    print(f"  Total LLM calls    : {metrics['total_llm_calls']}")
    print(f"\n  Results saved to   : {results_path}")


if __name__ == "__main__":
    run(parse_args())