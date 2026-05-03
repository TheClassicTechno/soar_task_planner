"""
When-to-Ask (UPS) Navigation Baseline — Runner

Implements the three-strategy uncertainty resolution pipeline adapted to
text-based outdoor navigation.

Pipeline per scenario:
  1. Format 5 options: A/B/C/D (navigation choices) + E (none of the above)
  2. If factorization enabled (UPS §IV-B):
       a. Ask LLM for n_intents possible user intents P(θ|L)
       b. For each intent θ, ask LLM to score all options P(y|options, θ)
       c. Marginalize: score[y] = Σ P(θ|L) * P(y|options, θ)
     Else:
       Direct single-call scoring (fallback)
  3. Normalize scores (A–D only) for conformal prediction
  4. Apply conformal prediction → prediction set (over A/B/C/D/E)
  5. Map prediction set to resolution strategy:
       EXECUTE   — |set| == 1, E not in set → robot acts directly
       CLARIFY   — |set| > 1, E not in set → robot asks user
       INCAPABLE — E in set → robot cannot handle scenario, escalate

robot_decision values:
  "A"/"B"/"C"/"D"        → EXECUTE (robot acts directly)
  "ASK"                  → CLARIFY (robot asks semantic question)
  "ESCALATE"             → INCAPABLE (robot cannot safely navigate)

Reuses: ConformalPredictor, MetricsCalculator, LLMInterface, NavigationScenario,
        load_scenarios_from_json — all from baselines/introplan.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import yaml

from baselines.introplan.conformal_predictor import (
    ConformalPredictor,
    normalize_confidences,
)
from baselines.introplan.llm_interface import LLMInterface
from baselines.introplan.metrics import MetricsCalculator, ScenarioResult
from baselines.introplan.navigation_prompts import OPTION_DESCRIPTIONS
from baselines.introplan.runner import NavigationScenario, load_scenarios_from_json
from baselines.whentoask.prompt import (
    NONE_OPTION_LABEL,
    NONE_OPTION_TEXT,
    format_direct_scoring_prompt,
    format_intent_prompt,
    format_scoring_prompt,
    marginalize_scores,
    parse_intent_response,
    parse_option_scores,
)

# Resolution strategy constants
STRATEGY_EXECUTE = "EXECUTE"
STRATEGY_CLARIFY = "CLARIFY"
STRATEGY_INCAPABLE = "INCAPABLE"


@dataclass
class WhenToAskDecision:
    """Output from one When-to-Ask inference call."""
    scenario_id: str
    options: Dict[str, str]                # A/B/C/D (nav options, no E)
    option_scores: Dict[str, float]        # Marginalized raw scores (A/B/C/D/E)
    option_confidences: Dict[str, float]   # Normalized scores for conformal prediction
    prediction_set: List[str]              # Options that passed the conformal threshold
    resolution_strategy: str              # EXECUTE / CLARIFY / INCAPABLE
    robot_decision: str                   # "A"/"B"/"C"/"D", "ASK", or "ESCALATE"
    intents: List[Dict]                   # Intent hypotheses from Stage 1 (empty if no factorization)
    llm_calls: int                        # Number of LLM calls used for this scenario


class WhenToAskRunner:
    """
    End-to-end When-to-Ask (UPS) navigation baseline runner.

    Supports:
      - run_scenario():   Single scenario → WhenToAskDecision
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
            config_path: Path to baselines/whentoask/config.yaml
            llm:         Optional pre-constructed LLMInterface (for testing with mocks)
        """
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self._llm = llm or LLMInterface(
            model=self.config.get("llm", {}).get("model", "claude-sonnet-4-6"),
            max_tokens=self.config.get("llm", {}).get("max_tokens", 768),
            temperature=self.config.get("llm", {}).get("temperature", 0.0),
        )

        alpha = self.config.get("conformal", {}).get("alpha", 0.15)
        self._predictor = ConformalPredictor(alpha=alpha)

        fact_config = self.config.get("factorization", {})
        self._factorization_enabled = fact_config.get("enabled", True)
        self._n_intents = fact_config.get("n_intents", 3)

    def run_scenario(self, scenario: NavigationScenario) -> WhenToAskDecision:
        """
        Run the When-to-Ask pipeline for one scenario.

        Args:
            scenario: Navigation scenario with instruction, terrain, and options.

        Returns:
            WhenToAskDecision with prediction set and resolution strategy.
        """
        options = scenario.options or dict(OPTION_DESCRIPTIONS)
        llm_calls = 0

        if self._factorization_enabled:
            # ── Stage 1: Intent hypotheses ─────────────────────────────────────
            intent_prompt = format_intent_prompt(
                instruction=scenario.instruction,
                terrain_description=scenario.terrain_description,
                n_intents=self._n_intents,
            )
            intent_response = self._llm.predict_json(intent_prompt)
            llm_calls += 1

            intents = parse_intent_response(intent_response, self._n_intents)
            intent_dicts = [{"description": d, "probability": p} for d, p in intents]

            # ── Stage 2: Score options per intent, then marginalize ────────────
            intent_scores_with_probs = []
            for intent_desc, intent_prob in intents:
                scoring_prompt = format_scoring_prompt(
                    instruction=scenario.instruction,
                    terrain_description=scenario.terrain_description,
                    intent_description=intent_desc,
                    options=options,
                )
                scoring_response = self._llm.predict_json(scoring_prompt)
                llm_calls += 1
                scores = parse_option_scores(scoring_response)
                intent_scores_with_probs.append((scores, intent_prob))

            raw_scores = marginalize_scores(intent_scores_with_probs)
        else:
            # ── Direct scoring (no factorization) ─────────────────────────────
            direct_prompt = format_direct_scoring_prompt(
                instruction=scenario.instruction,
                terrain_description=scenario.terrain_description,
                options=options,
            )
            direct_response = self._llm.predict_json(direct_prompt)
            llm_calls += 1
            raw_scores = parse_option_scores(direct_response)
            intent_dicts = []

        # ── Normalize A-D scores for conformal prediction ─────────────────────
        # E is tracked separately (incapability signal) — not normalized with A-D
        scores_abcd = {k: v for k, v in raw_scores.items() if k != NONE_OPTION_LABEL}
        confidences_abcd = normalize_confidences(scores_abcd)

        # Reconstruct full confidence dict (E is treated as a flag, not normalized)
        confidences = {**confidences_abcd, NONE_OPTION_LABEL: raw_scores.get(NONE_OPTION_LABEL, 0.0)}

        # ── Conformal prediction over all 5 options ────────────────────────────
        if self._predictor.tau is not None:
            prediction_set = self._predictor.predict_set(confidences)
        else:
            # No calibration yet — pick highest-confidence real option (not E)
            best = max(confidences_abcd, key=confidences_abcd.get)
            prediction_set = [best]

        # ── Map prediction set to resolution strategy ─────────────────────────
        strategy, robot_decision = _resolve_strategy(prediction_set)

        return WhenToAskDecision(
            scenario_id=scenario.scenario_id,
            options=options,
            option_scores=raw_scores,
            option_confidences=confidences,
            prediction_set=prediction_set,
            resolution_strategy=strategy,
            robot_decision=robot_decision,
            intents=intent_dicts,
            llm_calls=llm_calls,
        )

    def calibrate(self, calibration_scenarios: List[NavigationScenario]) -> float:
        """
        Calibrate the conformal threshold tau.

        Uses the UPS non-conformity score: κ = 1 - confidence_correct_option.
        For single-step scenarios, the sequence-level min reduces to a single step.

        Args:
            calibration_scenarios: Scenarios with known correct_option labels.

        Returns:
            The computed tau threshold.
        """
        for scenario in calibration_scenarios:
            decision = self.run_scenario(scenario)
            # Record calibration using the normalized confidence of the correct option
            # (not E — calibration is over the real nav options A/B/C/D)
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
            Metrics dict from MetricsCalculator.summary() plus strategy counts.
        """
        calc = MetricsCalculator()
        strategy_counts = {STRATEGY_EXECUTE: 0, STRATEGY_CLARIFY: 0, STRATEGY_INCAPABLE: 0}

        for scenario in test_scenarios:
            decision = self.run_scenario(scenario)
            strategy_counts[decision.resolution_strategy] += 1

            result = ScenarioResult(
                scenario_id=scenario.scenario_id,
                correct_option=scenario.correct_option,
                prediction_set=decision.prediction_set,
                robot_decision=decision.robot_decision,
            )
            calc.add(result)

        metrics = calc.summary()
        metrics["strategy_counts"] = strategy_counts
        return metrics


def _resolve_strategy(prediction_set: List[str]) -> tuple:
    """
    Map a prediction set to a (resolution_strategy, robot_decision) pair.

    Rules (from UPS §IV-C):
      EXECUTE:   singleton set, not E  → act directly with that option
      CLARIFY:   multi-option set, no E → ask semantic clarification
      INCAPABLE: E anywhere in set     → robot cannot handle this safely

    Returns:
        (strategy_str, robot_decision_str)
    """
    has_none = NONE_OPTION_LABEL in prediction_set
    real_options = [o for o in prediction_set if o != NONE_OPTION_LABEL]

    if has_none:
        # Incapability detected — E is in the set regardless of other options
        return STRATEGY_INCAPABLE, "ESCALATE"

    if len(real_options) == 1:
        # High confidence on exactly one real option → execute directly
        return STRATEGY_EXECUTE, real_options[0]

    # Multiple plausible options (or empty set) → semantic uncertainty → ask
    return STRATEGY_CLARIFY, "ASK"
