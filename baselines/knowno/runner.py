"""
KnowNo Navigation Baseline — Runner

Implements the KnowNo (Ren et al. CoRL 2023) conformal prediction approach
for outdoor robot navigation uncertainty, without retrieval augmentation.

Pipeline per scenario:
  1. Format a direct-scoring prompt: instruction + terrain + options A/B/C/D
  2. Call LLM once → predict best option + confidence
  3. Distribute confidence to per-option scores via extract_option_confidences
  4. Apply conformal prediction → prediction set
  5. |set| == 1 → ACT (that option)   |set| > 1 → ASK

Comparison with IntroPlan:
  IntroPlan asks the LLM to reason over 3 retrieved similar examples before
  predicting. KnowNo gives the LLM only the current scenario.
  Everything else (conformal calibration, metrics, data) is identical.

Reuses: ConformalPredictor, MetricsCalculator, LLMInterface, NavigationScenario,
        load_scenarios_from_json, extract_option_confidences — all from introplan/.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import yaml

from baselines.introplan.conformal_predictor import (
    ConformalPredictor,
    extract_option_confidences,
    normalize_confidences,
)
from baselines.introplan.llm_interface import LLMInterface
from baselines.introplan.metrics import MetricsCalculator, ScenarioResult
from baselines.introplan.navigation_prompts import OPTION_DESCRIPTIONS
from baselines.introplan.runner import NavigationScenario, load_scenarios_from_json


# ── Prompt template ───────────────────────────────────────────────────────────

_DIRECT_SCORING_PROMPT = """\
You are evaluating navigation decisions for an outdoor assistance robot.

Current situation:
  User instruction: "{instruction}"
  Terrain observation: "{terrain}"

Candidate actions:
  A: {option_a}
  B: {option_b}
  C: {option_c}
  D: {option_d}

Choose the BEST action for this situation. Provide your confidence (0.0–1.0) that this is correct.

Rules:
- High confidence (0.8+) means you are sure this is the right action.
- Medium confidence (0.5–0.8) means you think this is likely correct but are not certain.
- Low confidence (<0.5) means the situation is ambiguous.

Respond in JSON:
{{
  "prediction": "A" or "B" or "C" or "D",
  "confidence": 0.0 to 1.0,
  "reasoning": "one-sentence explanation"
}}"""


def format_direct_scoring_prompt(
    instruction: str,
    terrain: str,
    options: Dict[str, str],
) -> str:
    """
    Build the direct-scoring prompt for KnowNo.

    Unlike IntroPlan's introspective prompt, this gives the LLM only the
    current scenario — no retrieved examples, no step-by-step reasoning.
    """
    return _DIRECT_SCORING_PROMPT.format(
        instruction=instruction,
        terrain=terrain,
        option_a=options.get("A", OPTION_DESCRIPTIONS["A"]),
        option_b=options.get("B", OPTION_DESCRIPTIONS["B"]),
        option_c=options.get("C", OPTION_DESCRIPTIONS["C"]),
        option_d=options.get("D", OPTION_DESCRIPTIONS["D"]),
    )


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class KnowNoDecision:
    """Output from one KnowNo inference call."""
    scenario_id: str
    options: Dict[str, str]           # A/B/C/D descriptions
    option_confidences: Dict[str, float]  # normalized per-option scores
    prediction_set: List[str]         # options that passed conformal threshold
    robot_decision: str               # "A"/"B"/"C"/"D" or "ASK"
    llm_raw: Dict                     # raw LLM response dict


# ── Runner ────────────────────────────────────────────────────────────────────

class KnowNoRunner:
    """
    End-to-end KnowNo navigation baseline runner.

    Supports:
      - run_scenario():   Single scenario → KnowNoDecision
      - calibrate():      Calibrate conformal threshold from labeled scenarios
      - run_evaluation(): Full evaluation loop → metrics dict
    """

    def __init__(
        self,
        config_path: str,
        llm: Optional[LLMInterface] = None,
    ):
        """
        Args:
            config_path: Path to baselines/knowno/config.yaml
            llm:         Optional pre-constructed LLMInterface (for testing with mocks)
        """
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self._llm = llm or LLMInterface(
            model=self.config.get("llm", {}).get("model", "gpt-4o-mini"),
            max_tokens=self.config.get("llm", {}).get("max_tokens", 512),
            temperature=self.config.get("llm", {}).get("temperature", 0.0),
            api_type=self.config.get("llm", {}).get("api_type", "openai"),
        )

        alpha = self.config.get("conformal", {}).get("alpha", 0.15)
        self._predictor = ConformalPredictor(alpha=alpha)

    def run_scenario(self, scenario: NavigationScenario) -> KnowNoDecision:
        """
        Run the KnowNo pipeline for one scenario.

        Makes exactly one LLM call — no retrieval, no multi-step reasoning.

        Args:
            scenario: NavigationScenario with instruction, terrain, and options.

        Returns:
            KnowNoDecision with prediction set and final robot decision.
        """
        options = scenario.options or dict(OPTION_DESCRIPTIONS)

        # Single LLM call: direct scoring without retrieval
        prompt = format_direct_scoring_prompt(
            instruction=scenario.instruction,
            terrain=scenario.terrain_description,
            options=options,
        )
        raw = self._llm.predict_json(prompt)

        # Convert single prediction + confidence → per-option distribution
        confidences = extract_option_confidences(raw)
        confidences = normalize_confidences(confidences)

        # Conformal prediction → prediction set
        if self._predictor.tau is not None:
            prediction_set = self._predictor.predict_set(confidences)
        else:
            # Not yet calibrated — fall back to argmax
            best = max(confidences, key=confidences.get)
            prediction_set = [best]

        robot_decision = prediction_set[0] if len(prediction_set) == 1 else "ASK"

        return KnowNoDecision(
            scenario_id=scenario.scenario_id,
            options=options,
            option_confidences=confidences,
            prediction_set=prediction_set,
            robot_decision=robot_decision,
            llm_raw=raw,
        )

    def calibrate(self, calibration_scenarios: List[NavigationScenario]) -> float:
        """
        Calibrate the conformal threshold tau from labeled calibration scenarios.

        Args:
            calibration_scenarios: Scenarios with known correct_option labels.

        Returns:
            The computed tau threshold.
        """
        for scenario in calibration_scenarios:
            decision = self.run_scenario(scenario)
            self._predictor.record_calibration(
                option_confidences=decision.option_confidences,
                correct_option=scenario.correct_option,
            )
        return self._predictor.calibrate()

    def run_evaluation(
        self,
        test_scenarios: List[NavigationScenario],
    ) -> Dict:
        """
        Run the full evaluation loop on a list of test scenarios.

        Args:
            test_scenarios: Scenarios with known correct_option labels.

        Returns:
            Metrics dict from MetricsCalculator.summary().
        """
        calc = MetricsCalculator()
        for scenario in test_scenarios:
            decision = self.run_scenario(scenario)
            result = ScenarioResult(
                scenario_id=scenario.scenario_id,
                correct_option=scenario.correct_option,
                prediction_set=decision.prediction_set,
                robot_decision=decision.robot_decision,
            )
            calc.add(result)
        return calc.summary()
