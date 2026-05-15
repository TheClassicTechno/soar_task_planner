"""
LLM-as-Judge evaluation for IntroPlan navigation decisions.

For each scenario in a results.json file the judge evaluates two things:
  1. Label quality  — is the human-written correct_option genuinely the best choice?
  2. Robot quality  — was the robot's decision sound (not just accidentally correct)?

This catches two failure modes that string-match accuracy misses:
  - Robot gets the right letter for the wrong reason ("correct but lucky")
  - Robot gets the wrong letter but its reasoning was defensible ("wrong but reasonable")
  - Human label itself is debatable (label quality < 4/5)

Usage:
    python -m baselines.introplan.llm_judge \
        --results  outputs/introplan_baseline/results.json \
        --test_data baselines/introplan/data/nav_test.json \
        --output   outputs/introplan_baseline/judge_results.json
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

from baselines.introplan.llm_interface import LLMInterface
from baselines.introplan.navigation_prompts import UNCERTAINTY_TYPE_LABELS


# ── Judge prompt ──────────────────────────────────────────────────────────────

JUDGE_PROMPT = """You are an expert judge evaluating a robot navigation decision system.

SCENARIO:
  Instruction from user : "{instruction}"
  Terrain observation   : "{terrain_description}"
  Uncertainty type      : {uncertainty_type_label}

OPTIONS THE ROBOT HAD:
  A: {option_a}
  B: {option_b}
  C: {option_c}
  D: {option_d}

HUMAN LABEL  : The scenario was hand-labeled as correct_option = {correct_option}
ROBOT CHOICE : The robot predicted option {robot_decision} (prediction set: {prediction_set})

Please evaluate two things:

1. LABEL QUALITY
   Is option {correct_option} genuinely the best response for this scenario?
   Consider all 4 options carefully against the uncertainty type and terrain.

2. ROBOT DECISION QUALITY
   Was the robot's choice of {robot_decision} reasonable?
   Even if it matches the label, assess whether the reasoning is sound.
   Even if it does not match the label, assess whether there is merit to the choice.

Respond ONLY in this exact JSON format:
{{
  "label_agrees": true or false,
  "label_score": <integer 1-5>,
  "label_note": "<one sentence explaining why you agree or disagree with the label>",
  "robot_verdict": "<one of: correct, correct_lucky, wrong_reasonable, wrong>",
  "robot_note": "<one sentence assessing the robot decision quality>",
  "judge_best_option": "<A, B, C, or D — what YOU think is the best option>"
}}

Verdict definitions:
  correct          — robot chose the right option AND the reasoning is sound
  correct_lucky    — robot chose the right option BUT the reasoning is weak or accidental
  wrong_reasonable — robot chose the wrong option BUT the reasoning has clear merit
  wrong            — robot chose the wrong option AND the reasoning is poor"""


def _build_judge_prompt(decision: Dict, test_entry: Dict) -> str:
    utype = test_entry.get("uncertainty_type", 1)
    label = UNCERTAINTY_TYPE_LABELS.get(utype, f"Type {utype}")
    opts = test_entry.get("options", {})
    return JUDGE_PROMPT.format(
        instruction=decision["instruction"],
        terrain_description=decision["terrain_description"],
        uncertainty_type_label=label,
        option_a=opts.get("A", ""),
        option_b=opts.get("B", ""),
        option_c=opts.get("C", ""),
        option_d=opts.get("D", ""),
        correct_option=decision["correct_option"],
        robot_decision=decision["robot_decision"],
        prediction_set=", ".join(decision.get("prediction_set", [])),
    )


def _summarize(judgments: List[Dict]) -> Dict:
    n = len(judgments)
    if n == 0:
        return {}

    label_agree_count = sum(1 for j in judgments if j.get("label_agrees"))
    avg_label_score = sum(j.get("label_score", 0) for j in judgments) / n

    verdict_counts = {"correct": 0, "correct_lucky": 0, "wrong_reasonable": 0, "wrong": 0}
    for j in judgments:
        v = j.get("robot_verdict", "wrong")
        if v in verdict_counts:
            verdict_counts[v] += 1

    # Scenarios where judge disagrees with our label (label_score <= 3)
    contested_labels = [
        j["scenario_id"] for j in judgments
        if not j.get("label_agrees") or j.get("label_score", 5) <= 3
    ]

    return {
        "n_scenarios": n,
        "label_agreement_rate": label_agree_count / n,
        "avg_label_score": round(avg_label_score, 3),
        "contested_label_ids": contested_labels,
        "verdict_counts": verdict_counts,
        "verdict_rates": {k: round(v / n, 4) for k, v in verdict_counts.items()},
        # Adjusted SR: count correct + correct_lucky as passing, exclude wrong_reasonable from failures
        "adjusted_SR": round(
            (verdict_counts["correct"] + verdict_counts["correct_lucky"]) / n, 4
        ),
        # Strong SR: only count "correct" (right answer AND sound reasoning)
        "strong_SR": round(verdict_counts["correct"] / n, 4),
    }


def run_judge(
    results_path: str,
    test_data_path: str,
    output_path: str,
    llm: LLMInterface,
) -> None:
    with open(results_path) as f:
        results = json.load(f)

    with open(test_data_path) as f:
        test_data = json.load(f)

    test_lookup: Dict[str, Dict] = {e["entry_id"]: e for e in test_data}
    decisions = results.get("decisions", [])

    print(f"\n[LLM Judge] Evaluating {len(decisions)} decisions...")
    print(f"  Results file : {results_path}")
    print(f"  Test data    : {test_data_path}")

    judgments = []
    for i, decision in enumerate(decisions, 1):
        sid = decision["scenario_id"]
        test_entry = test_lookup.get(sid)
        if test_entry is None:
            print(f"  [{i}/{len(decisions)}] {sid} — SKIPPED (not found in test data)")
            continue

        prompt = _build_judge_prompt(decision, test_entry)
        try:
            result = llm.predict_json(prompt)
        except Exception as e:
            print(f"  [{i}/{len(decisions)}] {sid} — ERROR: {e}")
            continue

        judgment = {
            "scenario_id": sid,
            "instruction": decision["instruction"],
            "correct_option": decision["correct_option"],
            "robot_decision": decision["robot_decision"],
            "robot_was_correct_by_label": decision.get("correct", False),
            "label_agrees": result.get("label_agrees"),
            "label_score": result.get("label_score"),
            "label_note": result.get("label_note", ""),
            "robot_verdict": result.get("robot_verdict", ""),
            "robot_note": result.get("robot_note", ""),
            "judge_best_option": result.get("judge_best_option", ""),
        }
        judgments.append(judgment)

        verdict = judgment["robot_verdict"]
        label_ok = "✓label" if judgment["label_agrees"] else "✗label"
        print(f"  [{i}/{len(decisions)}] {sid}  {label_ok}  verdict={verdict}")

    summary = _summarize(judgments)
    summary["source_results"] = results_path
    summary["source_test_data"] = test_data_path
    summary["total_llm_calls"] = llm.total_calls

    output = {"summary": summary, "judgments": judgments}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*55}")
    print(f"LLM Judge Results — {summary['n_scenarios']} scenarios")
    print(f"{'='*55}")
    print(f"  Label agreement rate : {summary['label_agreement_rate']:.1%}")
    print(f"  Avg label score      : {summary['avg_label_score']:.2f} / 5.0")
    if summary["contested_label_ids"]:
        print(f"  Contested labels     : {summary['contested_label_ids']}")
    print(f"\n  Robot verdict breakdown:")
    for verdict, count in summary["verdict_counts"].items():
        rate = summary["verdict_rates"][verdict]
        print(f"    {verdict:<20} {count:>2}  ({rate:.1%})")
    print(f"\n  Adjusted SR (correct + lucky) : {summary['adjusted_SR']:.1%}")
    print(f"  Strong SR   (correct only)    : {summary['strong_SR']:.1%}")
    print(f"  String-match SR (original)    : {results['metrics']['SR']:.1%}")
    print(f"\n  Total LLM calls : {llm.total_calls}")
    print(f"  Saved to        : {output_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM-as-judge for IntroPlan decisions")
    p.add_argument(
        "--results", default="outputs/introplan_baseline/results.json",
        help="IntroPlan results.json to evaluate",
    )
    p.add_argument(
        "--test_data", default="baselines/introplan/data/nav_test.json",
        help="nav_test.json with full option texts and uncertainty types",
    )
    p.add_argument(
        "--output", default="outputs/introplan_baseline/judge_results.json",
        help="Where to save judge results",
    )
    return p.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        api_key, api_type, model = anthropic_key, "anthropic", "claude-sonnet-4-6"
    elif openai_key:
        api_key, api_type, model = openai_key, "openai", "gpt-4o"
    else:
        raise ValueError("Set ANTHROPIC_API_KEY or OPENAI_API_KEY in your .env file")

    print(f"[LLM Judge] Using {api_type} ({model})")
    llm = LLMInterface(api_key=api_key, api_type=api_type, model=model)
    run_judge(args.results, args.test_data, args.output, llm)


if __name__ == "__main__":
    main()
