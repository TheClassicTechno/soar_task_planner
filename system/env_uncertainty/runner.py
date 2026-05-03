"""
End-to-end Environmental Uncertainty Runner.

Orchestrates the full pipeline for one scene:
  1. Detect unknown regions (EnvironmentalUncertaintyDetector)
  2. Build candidate trajectories (TrajectoryGenerator)
  3. Score trajectories against traversability map
  4. Select best trajectory or decide to ASK
  5. Generate a clarification question if needed
  6. Apply user feedback and re-score trajectories (optional second pass)

Decision logic:
  - PROCEED: at least one trajectory avoids all unknown regions
  - ASK:     forward trajectory passes through unknown, but alternatives exist
             OR unknown coverage > ask_unknown_threshold
  - STOP:    all trajectories pass through unknown AND no safe path found
             (typically when unknown_coverage is very large)

Results are returned as EnvUncertaintyDecision objects, which are compatible
with the nav_env_test.json evaluation schema.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

from system.env_uncertainty.detector import DetectionResult, EnvironmentalUncertaintyDetector
from system.env_uncertainty.map_updater import MapUpdater
from system.env_uncertainty.question_generator import QuestionGenerator
from system.env_uncertainty.trajectory import Trajectory, TrajectoryGenerator
from system.env_uncertainty.traversability import TraversabilityMap


@dataclass
class EnvUncertaintyDecision:
    """
    Output from one EnvironmentalUncertaintyRunner.run_scene() call.

    scene_id:             identifier for this scene (from test case or image name)
    has_unknown:          whether any unknown region was detected
    unknown_coverage:     fraction of image pixels in unknown regions
    sam3_coverage:        fraction of image pixels labeled by SAM3
    best_trajectory:      the selected trajectory, or None if ASK/STOP
    robot_action:         "PROCEED", "ASK", or "STOP"
    question:             the generated question if robot_action == "ASK", else None
    n_known_regions:      number of SAM3-identified regions
    n_unknown_regions:    number of SAM2 residual unknown regions
    """

    scene_id: str
    has_unknown: bool
    unknown_coverage: float
    sam3_coverage: float
    best_trajectory: Optional[Trajectory]
    robot_action: str
    question: Optional[str]
    n_known_regions: int
    n_unknown_regions: int


class EnvironmentalUncertaintyRunner:
    """
    End-to-end runner for environmental uncertainty detection and resolution.

    Supports:
      - run_scene():      process one image → EnvUncertaintyDecision
      - run_evaluation(): evaluate on nav_env_test.json-format test cases
    """

    def __init__(
        self,
        config_path: str,
        detector: Optional[EnvironmentalUncertaintyDetector] = None,
        llm: Optional[Any] = None,
    ):
        """
        Args:
            config_path: Path to system/env_uncertainty/config.yaml.
            detector:    Pre-built detector (for testing with mocks).
            llm:         Optional LLMInterface for LLM-mode question generation.
        """
        with open(config_path) as f:
            self._config = yaml.safe_load(f)

        decision_cfg = self._config.get("decision", {})
        self._ask_threshold = decision_cfg.get("ask_unknown_threshold", 0.10)
        self._stop_threshold = decision_cfg.get("stop_unknown_threshold", 0.80)

        traj_cfg = self._config.get("trajectory", {})
        self._n_waypoints = traj_cfg.get("n_waypoints", 20)

        q_cfg = self._config.get("question", {})
        q_mode = q_cfg.get("mode", "template")
        if llm is not None:
            q_mode = "llm"

        self._detector = detector or EnvironmentalUncertaintyDetector()
        self._question_gen = QuestionGenerator(mode=q_mode, llm=llm)
        self._map_updater = MapUpdater()

    def run_scene(
        self,
        image,
        scene_id: str = "scene",
    ) -> EnvUncertaintyDecision:
        """
        Run the full pipeline on one image.

        Args:
            image:    (H, W, 3) uint8 numpy array (RGB).
            scene_id: Identifier string for logging/results.

        Returns:
            EnvUncertaintyDecision with action and optional question.
        """
        import numpy as np
        image = np.asarray(image)
        h, w = image.shape[:2]

        # Step 1: Detect unknown regions
        result: DetectionResult = self._detector.detect(image)

        # Step 2: Generate and score candidate trajectories
        gen = TrajectoryGenerator(h, w, n_waypoints=self._n_waypoints)
        raw_trajectories = gen.generate_trajectories()
        scored_trajectories = [
            gen.score_trajectory(t, result.traversability_map)
            for t in raw_trajectories
        ]

        # Step 3: Select best trajectory
        best = gen.select_best_trajectory(scored_trajectories)

        # Step 4: Determine robot action
        action, question = self._decide_action(result, scored_trajectories, best)

        return EnvUncertaintyDecision(
            scene_id=scene_id,
            has_unknown=result.has_unknown,
            unknown_coverage=result.unknown_coverage,
            sam3_coverage=result.sam3_coverage,
            best_trajectory=best,
            robot_action=action,
            question=question,
            n_known_regions=len(result.known_regions),
            n_unknown_regions=len(result.unknown_regions),
        )

    def run_evaluation(self, test_cases: List[Dict]) -> Dict:
        """
        Evaluate on a list of test cases from nav_env_test.json.

        Each test case must contain:
            scene_description, correct_action, should_ask, unknown_region_pixel_fraction

        For this evaluation we use synthetic images (blank arrays) since real
        RUGD images may not be available. The detector is expected to be mocked
        or pre-configured with the correct outputs for each test case.

        Args:
            test_cases: List of dicts matching nav_env_test.json schema.

        Returns:
            Metrics dict with AAR, SAR, URDR, and n_scenarios.
        """
        import numpy as np

        n = len(test_cases)
        n_correct_ask = 0
        n_should_ask = 0
        n_correct_proceed = 0
        n_should_proceed = 0

        for case in test_cases:
            # Use a minimal placeholder image — real evaluation uses actual images
            dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
            scene_id = case.get("entry_id", "?")
            decision = self.run_scene(dummy_image, scene_id=scene_id)

            should_ask = bool(case.get("should_ask", False))
            predicted_ask = decision.robot_action == "ASK"

            if should_ask:
                n_should_ask += 1
                if predicted_ask:
                    n_correct_ask += 1
            else:
                n_should_proceed += 1
                if not predicted_ask:
                    n_correct_proceed += 1

        aar = n_correct_ask / n_should_ask if n_should_ask > 0 else 0.0
        sar = (
            (n_should_proceed - n_correct_proceed) / n_should_proceed
            if n_should_proceed > 0
            else 0.0
        )

        return {
            "n_scenarios": n,
            "n_should_ask": n_should_ask,
            "n_should_proceed": n_should_proceed,
            "AAR": round(aar, 4),   # Appropriate Ask Rate
            "SAR": round(sar, 4),   # Spurious Ask Rate
            "n_correct_ask": n_correct_ask,
            "n_correct_proceed": n_correct_proceed,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _decide_action(
        self,
        result: DetectionResult,
        trajectories: List[Trajectory],
        best_trajectory: Optional[Trajectory],
    ):
        """
        Apply decision thresholds to pick PROCEED, ASK, or STOP.

        Returns (action_str, question_str_or_None).
        """
        # Very large unknown area → STOP (robot cannot proceed safely)
        if result.unknown_coverage >= self._stop_threshold:
            question = self._question_gen.generate(result, trajectories)
            return "STOP", question

        # No unknown regions, or unknown area is small and off-path → PROCEED
        if not result.has_unknown or (
            result.unknown_coverage < self._ask_threshold
            and best_trajectory is not None
            and not best_trajectory.passes_through_unknown
        ):
            return "PROCEED", None

        # Unknown region exists and robot's path is affected → ASK
        question = self._question_gen.generate(result, trajectories)
        return "ASK", question
