"""
Evaluation metrics for the IntroPlan navigation baseline.

Metrics mirror those used in IntroPlan (Liang et al., NeurIPS 2024) and
KnowNo (Ren et al., CoRL 2023), adapted to our navigation uncertainty setting.

Definitions (per scenario):
  - Correct prediction: robot's chosen option == ground-truth correct option
  - Human Help (HR):   robot decided to ASK (prediction set size > 1)
  - False Positive:    robot ASKed when the correct option was NOT B (autonomous action was right)
  - Exact Set:         prediction set == {correct option} exactly

Primary metrics:
  SR   (Success Rate)               — % scenarios where robot correctly identifies the right action
  HR   (Human Help Rate)            — % scenarios where robot decides to ask human
  FPR  (False Positive Rate)        — % scenarios robot asks when it could have acted correctly
  NCR  (Non-Compliant Contamination Rate) — % prediction sets containing a wrong option
  ESR  (Exact Set Rate)             — % prediction sets that contain exactly the correct option
  SR-HR AUC                         — area under SR vs HR trade-off curve (sweep tau)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


class ScenarioResult:
    """Result for a single evaluation scenario."""

    def __init__(
        self,
        scenario_id: str,
        correct_option: str,
        prediction_set: List[str],
        robot_decision: str,
    ):
        """
        Args:
            scenario_id:    Unique ID of the scenario.
            correct_option: Ground-truth correct option label (A/B/C/D).
            prediction_set: Options in the conformal prediction set.
            robot_decision: The option the robot acts on, or "ASK" if |set| > 1.
        """
        self.scenario_id = scenario_id
        self.correct_option = correct_option
        self.prediction_set = prediction_set
        self.robot_decision = robot_decision

    @property
    def asked_human(self) -> bool:
        """Robot decided to ask the human (prediction set was ambiguous)."""
        return self.robot_decision == "ASK"

    @property
    def correct(self) -> bool:
        """Robot's chosen action matches the ground-truth correct option."""
        if self.asked_human:
            # Asking always resolves correctly (human provides the answer)
            return True
        return self.robot_decision == self.correct_option

    @property
    def exact_set(self) -> bool:
        """Prediction set contains exactly the correct option (no more, no less)."""
        return (
            len(self.prediction_set) == 1
            and self.prediction_set[0] == self.correct_option
        )

    @property
    def noncompliant(self) -> bool:
        """Prediction set contains at least one option that is NOT the correct one."""
        return any(opt != self.correct_option for opt in self.prediction_set)

    @property
    def false_positive(self) -> bool:
        """Robot asked when it should have acted autonomously without consulting the user.

        Option B is the only action that involves asking the user.  Options A
        (proceed), C (reroute), and D (slow down) are all autonomous acts.
        If the robot triggers a human-help request when the correct answer was
        any of those, it over-asked — that is a false positive.
        """
        return self.asked_human and self.correct_option != "B"


class MetricsCalculator:
    """
    Accumulates scenario results and computes aggregate metrics.

    Usage:
        calc = MetricsCalculator()
        for result in results:
            calc.add(result)
        summary = calc.summary()
    """

    def __init__(self):
        self._results: List[ScenarioResult] = []

    def add(self, result: ScenarioResult) -> None:
        self._results.append(result)

    def __len__(self) -> int:
        return len(self._results)

    def success_rate(self) -> float:
        """SR: fraction of scenarios where robot's decision is correct."""
        if not self._results:
            return 0.0
        return sum(r.correct for r in self._results) / len(self._results)

    def human_help_rate(self) -> float:
        """HR: fraction of scenarios where robot asked the human."""
        if not self._results:
            return 0.0
        return sum(r.asked_human for r in self._results) / len(self._results)

    def false_positive_rate(self) -> float:
        """FPR: fraction of questions asked when the robot should have acted autonomously (correct option was not B)."""
        questions = [r for r in self._results if r.asked_human]
        if not questions:
            return 0.0
        return sum(r.false_positive for r in questions) / len(questions)

    def non_compliant_contamination_rate(self) -> float:
        """NCR: fraction of prediction sets that contain a wrong option."""
        if not self._results:
            return 0.0
        return sum(r.noncompliant for r in self._results) / len(self._results)

    def exact_set_rate(self) -> float:
        """ESR: fraction of prediction sets containing exactly the correct option."""
        if not self._results:
            return 0.0
        return sum(r.exact_set for r in self._results) / len(self._results)

    def summary(self) -> Dict:
        """Return all metrics as a dict."""
        return {
            "n_scenarios": len(self._results),
            "SR":  round(self.success_rate(), 4),
            "HR":  round(self.human_help_rate(), 4),
            "FPR": round(self.false_positive_rate(), 4),
            "NCR": round(self.non_compliant_contamination_rate(), 4),
            "ESR": round(self.exact_set_rate(), 4),
        }


def compute_sr_hr_auc(
    scenario_results_by_tau: Dict[float, List[ScenarioResult]],
) -> float:
    """
    Compute the area under the SR vs HR trade-off curve.

    To generate the curve, sweep tau (the conformal threshold) across multiple
    values. For each tau, compute SR and HR. Plot SR vs HR and compute AUC
    using the trapezoidal rule (same as KnowNo paper §4).

    Args:
        scenario_results_by_tau: {tau_value: [ScenarioResult, ...]}
            One entry per tau value, each containing the full set of results
            obtained when that tau was used for the prediction set threshold.

    Returns:
        AUC value in [0, 1]. Higher = better trade-off between SR and HR.
    """
    points: List[Tuple[float, float]] = []  # (HR, SR) pairs

    for tau in sorted(scenario_results_by_tau.keys()):
        calc = MetricsCalculator()
        for r in scenario_results_by_tau[tau]:
            calc.add(r)
        points.append((calc.human_help_rate(), calc.success_rate()))

    if len(points) < 2:
        return 0.0

    # Sort by HR (x-axis) for trapezoidal integration
    points.sort(key=lambda p: p[0])
    hr_vals = np.array([p[0] for p in points])
    sr_vals = np.array([p[1] for p in points])
    # np.trapezoid is the NumPy 2.0+ name; np.trapz was removed in 2.0
    try:
        return float(np.trapezoid(sr_vals, hr_vals))
    except AttributeError:
        return float(np.trapz(sr_vals, hr_vals))  # type: ignore[attr-defined]
