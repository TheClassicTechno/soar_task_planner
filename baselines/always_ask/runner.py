"""
Always Ask Runner for Navigation

A simple baseline that always asks the human for every decision.
"""

from dataclasses import dataclass
from typing import Dict

from baselines.introplan.runner import NavigationScenario


@dataclass
class AlwaysAskDecision:
    """Output from Always Ask inference."""
    scenario_id: str
    options: Dict[str, str]
    robot_decision: str = "ASK"


class AlwaysAskRunner:
    """Always Ask runner — always returns ASK, never decides on its own."""

    def __init__(
        self,
        config_path: str,
    ):
        self._config_path = config_path

    def run_scenario(self, scenario: NavigationScenario) -> AlwaysAskDecision:
        """Run Always Ask for one scenario — always ask the human."""
        options = scenario.options or {}

        return AlwaysAskDecision(
            scenario_id=scenario.scenario_id,
            options=options,
            robot_decision="ASK",
        )