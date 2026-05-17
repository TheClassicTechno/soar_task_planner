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
  - PROCEED: at least one trajectory avoids all unknown regions AND all on-path
             scene-graph nodes (if a SceneGraph is provided) have low Dirichlet
             semantic entropy.
  - ASK:     unknown coverage > ask_unknown_threshold, OR best trajectory passes
             through unknown, OR any on-path node's semantic_entropy() exceeds
             entropy_ask_threshold (the robot is uncertain about terrain class).
  - STOP:    unknown_coverage >= stop_unknown_threshold (no safe path exists).

Results are returned as EnvUncertaintyDecision objects, which are compatible
with the nav_env_test.json evaluation schema.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from system.env_uncertainty.detector import DetectionResult, EnvironmentalUncertaintyDetector
from system.env_uncertainty.gp_traversability import GPTraversabilityMap
from system.env_uncertainty.map_updater import MapUpdater
from system.env_uncertainty.question_generator import QuestionGenerator
from system.env_uncertainty.scene_graph import SceneGraph, TerrainNode
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
    unknown_world_coords: world coordinates (x, y) of unknown region centroids
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
    unknown_world_coords: Optional[List[Tuple[float, float]]] = None


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
        self._entropy_threshold = decision_cfg.get("entropy_ask_threshold", 1.5)
        self._lcb_stop_threshold = decision_cfg.get("lcb_stop_threshold", 0.20)

        traj_cfg = self._config.get("trajectory", {})
        self._n_waypoints = traj_cfg.get("n_waypoints", 20)

        q_cfg = self._config.get("question", {})
        q_mode = q_cfg.get("mode", "template")
        if llm is not None:
            q_mode = "llm"

        self._detector = detector or EnvironmentalUncertaintyDetector()
        self._question_gen = QuestionGenerator(mode=q_mode, llm=llm)
        self._map_updater = MapUpdater()
        self._gp_map = GPTraversabilityMap()

    def run_scene(
        self,
        image,
        scene_id: str = "scene",
        scene_graph: Optional[SceneGraph] = None,
        robot_pose: Optional[Any] = None,
        camera_params: Optional[Any] = None,
        T_cam_to_base: Optional[np.ndarray] = None,
        depth_map: Optional[np.ndarray] = None,
    ) -> EnvUncertaintyDecision:
        """
        Run the full pipeline on one image.

        Args:
            image:        (H, W, 3) uint8 numpy array (RGB).
            scene_id:     Identifier string for logging/results.
            scene_graph:  Optional SceneGraph built from Steps 3–5 of the
                          pipeline.  When provided, nodes adjacent to the best
                          trajectory are checked for high Dirichlet semantic
                          entropy and can trigger ASK independently of unknown
                          coverage.
            robot_pose:   Optional (x, y, theta) robot pose in world frame.
                          If provided, unknown region centroids will be
                          transformed to world coordinates (Step 10).
            camera_params: Optional CameraParams for coordinate transform.
            T_cam_to_base: Optional 4x4 transformation matrix.
            depth_map:    Optional (H, W) depth image in meters.
                          If provided, used for per-pixel depth in coordinate
                          transform. If None, uses default_depth from config.

        Returns:
            EnvUncertaintyDecision with action and optional question.
        """
        import numpy as np
        image = np.asarray(image)
        h, w = image.shape[:2]

        # Step 1: Detect unknown regions
        result: DetectionResult = self._detector.detect(image)

        # Step 4: Seed GP with traversability observations from known regions
        self._seed_gp_from_detection(result, h, w)

        # Step 2: Generate and score candidate trajectories (passes_through_unknown flag)
        gen = TrajectoryGenerator(h, w, n_waypoints=self._n_waypoints)
        raw_trajectories = gen.generate_trajectories()
        scored_trajectories = [
            gen.score_trajectory(t, result.traversability_map)
            for t in raw_trajectories
        ]

        # Step 5: Select best trajectory using GP LCB (pessimistic safety ranking)
        best, best_lcb = self._select_best_trajectory_lcb(scored_trajectories, h, w)

        # Resolve on-path scene-graph nodes for entropy check (Steps 3–5 output)
        on_path_nodes: List[TerrainNode] = []
        if scene_graph is not None and best is not None:
            on_path_nodes = self._on_path_nodes(scene_graph, best, h, w)

        # Step 4: Determine robot action
        action, question = self._decide_action(
            result, scored_trajectories, best, on_path_nodes, best_lcb=best_lcb
        )

        # Step 10: Transform unknown region centroids to world coordinates
        unknown_world_coords = None
        if robot_pose is not None and camera_params is not None and result.unknown_regions:
            unknown_world_coords = self._compute_world_coords(
                result, robot_pose, camera_params, T_cam_to_base, depth_map
            )

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
            unknown_world_coords=unknown_world_coords,
        )

    def _compute_world_coords(
        self,
        result: DetectionResult,
        robot_pose: Any,
        camera_params: Any,
        T_cam_to_base: Optional[np.ndarray] = None,
        depth_map: Optional[np.ndarray] = None,
    ) -> List[Tuple[float, float]]:
        """
        Transform unknown region centroids to world coordinates (Step 10).

        Args:
            result: DetectionResult with unknown regions
            robot_pose: (x, y, theta) or RobotPose in world frame
            camera_params: CameraParams with K matrix
            T_cam_to_base: Optional 4x4 transform matrix
            depth_map: Optional (H, W) depth image in meters.
                       If provided, depth for each pixel is read from this map.
                       If None, uses default_depth from config.

        Returns:
            List of (world_x, world_y) tuples for each unknown region
        """
        from system.env_uncertainty.coordinate_transform import (
            RobotPose,
            create_default_transform,
            create_default_camera_params,
            region_centroid_to_world,
        )

        coord_cfg = self._config.get("coordinate", {})
        default_depth = coord_cfg.get("default_depth", 2.0)
        camera_height = coord_cfg.get("camera_height", 0.3)

        if T_cam_to_base is None:
            T_cam_to_base = create_default_transform(camera_height=camera_height)

        if isinstance(robot_pose, tuple):
            pose = RobotPose(x=robot_pose[0], y=robot_pose[1], theta=robot_pose[2])
        else:
            pose = robot_pose

        if camera_params is None:
            coord_cfg = self._config.get("coordinate", {})
            hfov = coord_cfg.get("camera_hfov", 90.0)
            camera_params = create_default_camera_params(640, 480, hfov)
            K = camera_params.K
        elif hasattr(camera_params, 'K'):
            K = camera_params.K
        elif isinstance(camera_params, np.ndarray):
            K = camera_params
        else:
            raise ValueError(f"camera_params must be CameraParams, np.ndarray, or None, got {type(camera_params)}")

        world_coords = []
        for region in result.unknown_regions:
            depth = default_depth
            if depth_map is not None:
                region_depths = depth_map[region.mask]
                valid_depths = region_depths[(region_depths > 0) & np.isfinite(region_depths)]
                if len(valid_depths) > 0:
                    depth = float(np.median(valid_depths))

            coord = region_centroid_to_world(
                mask=region.mask,
                depth=depth,
                K=K,
                T_cam_to_base=T_cam_to_base,
                robot_pose=pose,
            )
            if coord is not None:
                world_coords.append(coord)

        return world_coords

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

    def _seed_gp_from_detection(
        self, result: DetectionResult, height: int, width: int
    ) -> None:
        """
        Add one GP observation per known region, at its mask centroid.

        This runs Step 4 of the pipeline: the GP posterior is updated with
        SAM3's traversability labels before trajectory LCB scoring (Step 5).
        Unknown regions are intentionally excluded — the GP only learns from
        regions the robot already has a label for.
        """
        import numpy as np
        for region in result.known_regions:
            mask = np.asarray(region.mask)
            if not np.any(mask):
                continue
            ys, xs = np.where(mask)
            cy, cx = int(np.mean(ys)), int(np.mean(xs))
            self._gp_map.add_observation(cy, cx, region.traversability, height, width)

    def _select_best_trajectory_lcb(
        self,
        trajectories: List[Trajectory],
        height: int,
        width: int,
    ) -> tuple:
        """
        Select the safest trajectory by GP LCB score.

        Only considers trajectories where passes_through_unknown=False.
        Returns (best_trajectory, best_lcb_score).  Both are None when every
        trajectory passes through unknown (triggers ASK in _decide_action).

        Scores each safe trajectory once and keeps the winning score so
        _decide_action can apply the LCB-threshold STOP without a second GP call.
        """
        safe = [t for t in trajectories if not t.passes_through_unknown]
        if not safe:
            return None, None
        scored = [
            (t, self._gp_map.score_trajectory_lcb(t.waypoints, height, width))
            for t in safe
        ]
        best, best_lcb = max(scored, key=lambda ts: ts[1])
        return best, best_lcb

    def _decide_action(
        self,
        result: DetectionResult,
        trajectories: List[Trajectory],
        best_trajectory: Optional[Trajectory],
        on_path_nodes: Optional[List[TerrainNode]] = None,
        best_lcb: Optional[float] = None,
    ):
        """
        Apply decision thresholds to pick PROCEED, ASK, or STOP.

        Args:
            result:          DetectionResult from the detector.
            trajectories:    All scored candidate trajectories.
            best_trajectory: Highest-scoring trajectory (may be None).
            on_path_nodes:   TerrainNodes adjacent to best_trajectory from the
                             scene graph.  When any node's semantic_entropy()
                             exceeds self._entropy_threshold the robot must ASK
                             even if unknown_coverage looks acceptable.
            best_lcb:        Pre-computed GP LCB score for best_trajectory.
                             When this falls below lcb_stop_threshold and the GP
                             has observations, the robot STOPs on known-terrain
                             danger (e.g., ice, deep mud) even at low coverage.

        Returns:
            (action_str, question_str_or_None)
        """
        if on_path_nodes is None:
            on_path_nodes = []

        # Extract top-k terrain candidates from the most uncertain on-path node.
        # Passed into question gen so the robot names what it thinks it's seeing
        # instead of saying "unrecognized area" generically.
        top_k = self._top_k_from_nodes(on_path_nodes)

        # Very large unknown area → STOP (robot cannot proceed safely)
        if result.unknown_coverage >= self._stop_threshold:
            question = self._question_gen.generate(result, trajectories, top_k_classes=top_k)
            return "STOP", question

        # High Dirichlet semantic entropy on planned path → ASK.
        # The robot doesn't know what terrain class it's heading into, so it
        # must clarify before committing — regardless of unknown_coverage.
        if any(
            node.semantic_entropy() > self._entropy_threshold
            for node in on_path_nodes
        ):
            question = self._question_gen.generate(result, trajectories, top_k_classes=top_k)
            return "ASK", question

        # GP LCB-based STOP: best safe path has dangerously low traversability.
        # Guard: n_observations > 0 prevents false STOPs from the uninformative
        # prior (prior LCB = 0.5 - 1.5*0.4 = -0.1 < threshold without any data).
        if (
            best_lcb is not None
            and self._gp_map.n_observations > 0
            and best_lcb < self._lcb_stop_threshold
        ):
            question = self._question_gen.generate(result, trajectories, top_k_classes=top_k)
            return "STOP", question

        # No unknown regions, or unknown area is small and off-path → PROCEED
        if not result.has_unknown or (
            result.unknown_coverage < self._ask_threshold
            and best_trajectory is not None
            and not best_trajectory.passes_through_unknown
        ):
            return "PROCEED", None

        # Unknown region exists and robot's path is affected → ASK
        question = self._question_gen.generate(result, trajectories, top_k_classes=top_k)
        return "ASK", question

    def _top_k_from_nodes(
        self,
        on_path_nodes: List[TerrainNode],
        k: int = 3,
    ):
        """
        Return top-k terrain class candidates from the most uncertain on-path node.

        Finds the node with the highest semantic entropy (most uncertain about
        terrain class) and returns its Dirichlet top-k. Returns None when there
        are no on-path nodes, so question gen falls back to generic templates.
        """
        if not on_path_nodes:
            return None
        most_uncertain = max(on_path_nodes, key=lambda n: n.semantic_entropy())
        return most_uncertain.top_k_classes(k=k)

    def _on_path_nodes(
        self,
        scene_graph: SceneGraph,
        trajectory: Trajectory,
        height: int,
        width: int,
    ) -> List[TerrainNode]:
        """
        Return all scene-graph nodes whose grid cell any waypoint touches.

        Deduplicates by (label, cell) so each node appears at most once.

        Args:
            scene_graph: Current SceneGraph instance.
            trajectory:  Scored trajectory with (y, x) waypoints.
            height:      Image height in pixels.
            width:       Image width in pixels.

        Returns:
            Flat list of unique TerrainNodes along the trajectory.
        """
        seen: set = set()
        nodes: List[TerrainNode] = []
        for wy, wx in trajectory.waypoints:
            cy, cx = scene_graph.pixel_to_cell(wy, wx, height, width)
            for node in scene_graph.nodes_in_cell(cy, cx):
                key = (node.label, node.position_cell_id)
                if key not in seen:
                    seen.add(key)
                    nodes.append(node)
        return nodes
