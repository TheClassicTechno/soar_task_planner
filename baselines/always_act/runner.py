"""
Always Act Runner for Navigation

A simple baseline that always picks the highest-confidence option,
never asks the human.
"""

from dataclasses import dataclass
from typing import Dict, List

from baselines.introplan.llm_interface import LLMInterface
from baselines.introplan.runner import NavigationScenario
from baselines.introplan.navigation_prompts import format_introspective_predict_prompt
from baselines.introplan.conformal_predictor import extract_option_confidences, normalize_confidences


@dataclass
class AlwaysActDecision:
    """Output from Always Act inference."""
    scenario_id: str
    options: Dict[str, str]
    option_confidences: Dict[str, float]
    robot_decision: str


class AlwaysActRunner:
    """Always Act runner — same as IntroPlan but without tau threshold."""

    def __init__(
        self,
        config_path: str,
        llm,
        knowledge_base,
        top_k: int = 3,
    ):
        self._config_path = config_path
        self._llm = llm
        self._kb = knowledge_base
        self._top_k = top_k

    def run_scenario(self, scenario: NavigationScenario) -> AlwaysActDecision:
        """Run Always Act for one scenario."""
        options = scenario.options or {}

        similar = self._kb.retrieve_as_dicts(
            instruction=scenario.instruction,
            terrain_description=scenario.terrain_description,
            top_k=self._top_k,
        )

        prompt = format_introspective_predict_prompt(
            instruction=scenario.instruction,
            terrain_description=scenario.terrain_description,
            options=options,
            retrieved_examples=similar,
        )
        llm_response = self._llm.predict_json(prompt)

        raw_confidences = extract_option_confidences(llm_response)
        confidences = normalize_confidences(raw_confidences)

        # Always Act: pick highest confidence
        best_option = max(confidences, key=confidences.get)

        return AlwaysActDecision(
            scenario_id=scenario.scenario_id,
            options=options,
            option_confidences=confidences,
            robot_decision=best_option,
        )