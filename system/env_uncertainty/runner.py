"""
End-to-end Environmental Uncertainty Runner.

Orchestrates the S1-S5 pipeline for one scene (environmental branch only).
The instruction branch runs in parallel in joint_decision.py; both branches
merge at κ_joint = max(κ_I, κ_E).

Per june1meeting.txt (line 10-11): the pipeline is exactly FOUR OR FIVE stages.
The mentor defined S1-S5 as the authoritative numbering:

S1 — Perception & Segmentation
     SAM3 labels known terrain (13 classes, e.g. grass/concrete/gravel).
     SAM2 detects residual unknown regions (<30% overlap with SAM3 output).
     Output: DetectionResult — segments, labels, masks, confidence, unknown_coverage.

S2 — Scene Understanding & Trajectory
     (a) GP seeding: initialize GP with one centroid observation per S1 known region.
         Traversability prior is looked up from the static label→score table.
         PersistentTerrainKnowledge adjusts these priors using past user feedback.
     (b) Trajectory generation: 3 goal-directed Bezier curves (left / straight / right).
         Per may26 meeting (lines 302-309): trajectory generation and LCB scoring are
         the same conceptual operation — generate all paths then rank by traversability
         (same as MPPI). Combined into one step.
     (c) LCB scoring: score each path using GP Lower Confidence Bound (pessimistic).
         Select the safest trajectory.
     (d) Scene graph update: build/update TerrainNodes from live S1 detections.
         Dirichlet α initialized from SAM3 confidence. High-confidence → low entropy.
     Output: best_trajectory, best_lcb, scene_graph with on-path TerrainNodes.

S3 — Uncertainty Resolution (decision + communication + parse + update score)
     (a) Decision: PROCEED / ASK / STOP based on coverage, LCB, Dirichlet entropy.
         · STOP  if coverage ≥ 0.80 OR GP LCB < 0.20 (known dangerous terrain)
         · ASK   if coverage > 0.10, path crosses unknown, OR Dirichlet entropy > 1.5
         · PROCEED otherwise
     (b) Question generation: grounded in top-k Dirichlet candidates from target_node.
     (c) Response parsing (run_with_feedback): keyword matching → terrain_label,
         is_traversable, label_confidence (default 0.70 when no keyword found).
     (d) Score update: GP apply_user_feedback() at uncertain trajectory waypoints
         and all unknown region pixels. Updates traversability beliefs for this frame.
     Output: robot_action, question (if ASK/STOP), target_node, parsed_response.

S4 — Node Update (Bayesian update of scene graph, per june1meeting line 10)
     Dirichlet conjugate update: target_node.update_from_user(label, confidence).
     α[label] += confidence → lowers semantic_entropy() → robot stops re-asking
     about the same terrain class.
     PersistentTerrainKnowledge updated so confirmed labels carry forward to new frames.
     Output: updated scene_graph nodes, updated PersistentTerrainKnowledge.

S5 — Replan Loop (per june1meeting line 10: "do another round if more uncertainties")
     Re-run S1-S3 on the same image with updated GP posterior.
     If path is now safe → PROCEED. If still uncertain → another round of S3-S4.
     Implemented as a second run_scene() call inside run_with_feedback().
     Output: new EnvUncertaintyDecision reflecting updated terrain beliefs.

Decision logic:
  - PROCEED: at least one trajectory avoids all unknown regions AND all on-path
             scene-graph nodes (if a SceneGraph is provided) have low Dirichlet
             semantic entropy.
  - ASK:     unknown coverage > ask_unknown_threshold, OR best trajectory passes
             through unknown, OR any on-path node's semantic_entropy() exceeds
             entropy_ask_threshold (the robot is uncertain about terrain class).
  - STOP:    unknown_coverage >= stop_unknown_threshold (no safe path exists).

Goal-directed trajectories (D15, May 19 mentor requirement):
  Pass goal_pixel=(y, x) to run_scene() or run_with_feedback() to activate
  GoalDirectedTrajectoryGenerator. Without a goal, falls back to the original
  fixed-geometry TrajectoryGenerator.

Replan after feedback (D18, May 19 mentor requirement):
  run_with_feedback() runs the pipeline, incorporates user text feedback into
  the GP, then re-runs run_scene() on the same image with updated beliefs.
  Returns (initial_decision, replanned_decision) so callers can see both.

Results are returned as EnvUncertaintyDecision objects, which are compatible
with the nav_env_test.json evaluation schema.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from system.env_uncertainty.detector import DetectionResult, EnvironmentalUncertaintyDetector
from system.env_uncertainty.gp_traversability import GPTraversabilityMap, WorldGPTraversabilityMap
from system.env_uncertainty.map_updater import MapUpdater, parse_user_response_rich
from system.env_uncertainty.question_generator import QuestionGenerator
from system.env_uncertainty.scene_graph import SceneGraph, TerrainNode, WorldSceneGraph
from system.env_uncertainty.terrain_knowledge import PersistentTerrainKnowledge
from system.env_uncertainty.trajectory import (
    Trajectory,
    TrajectoryGenerator,
    GoalDirectedTrajectoryGenerator,
)
from system.env_uncertainty.traversability import TraversabilityMap
from system.env_uncertainty.user_profile import DEFAULT_PROFILE, UserProfile
from system.env_uncertainty.world_coords import (
    CameraMount,
    RobotPose,
    mask_centroids_to_world,
    pixel_to_world,
)


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
    target_node:          most uncertain on-path scene graph node (for S8 Dirichlet
                          update). Set to the node with highest semantic_entropy()
                          among on_path_nodes when robot_action == "ASK". None when
                          proceeding (no update needed) or when no scene graph used.
    decision_reason:      human-readable string explaining why this action was chosen.
                          Useful for debugging and paper evaluation.
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
    target_node: Optional["TerrainNode"] = None
    decision_reason: str = ""


def _decision_reason(action: str, result: "DetectionResult", best_lcb: Optional[float]) -> str:
    """One-sentence explanation of why this action was chosen — for logs and paper evaluation."""
    cov = result.unknown_coverage
    if action == "STOP":
        if cov >= 0.80:
            return f"STOP: unknown_coverage={cov:.2f} exceeds stop_threshold=0.80"
        if best_lcb is not None and best_lcb < 0.20:
            return f"STOP: best path GP LCB={best_lcb:.3f} below lcb_stop_threshold=0.20 (known dangerous terrain)"
        return f"STOP: all candidate paths pass through unknown regions (coverage={cov:.2f})"
    if action == "ASK":
        if cov >= 0.10:
            return f"ASK: unknown_coverage={cov:.2f} above ask_threshold=0.10"
        return "ASK: on-path scene graph node has high semantic entropy (terrain class unclear)"
    return f"PROCEED: safe path found, unknown_coverage={cov:.2f} below ask_threshold=0.10"


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
        use_real_models: bool = False,
        device: Optional[str] = None,
    ):
        """
        Args:
            config_path:     Path to system/env_uncertainty/config.yaml.
            detector:        Pre-built detector (for testing with mocks).
            llm:             Optional LLMInterface for LLM-mode question generation.
            use_real_models: If True, dynamically loads and uses real SAM2 and SAM3 models.
            device:          Optional torch device override (e.g., "cpu", "mps", "cuda").
        """
        with open(config_path) as f:
            self._config = yaml.safe_load(f)

        decision_cfg = self._config.get("decision", {})
        self._ask_threshold = decision_cfg.get("ask_unknown_threshold", 0.10)
        self._stop_threshold = decision_cfg.get("stop_unknown_threshold", 0.80)
        self._entropy_threshold = decision_cfg.get("entropy_ask_threshold", 1.5)
        self._lcb_stop_threshold = decision_cfg.get("lcb_stop_threshold", 0.20)
        self._path_unknown_tolerance = decision_cfg.get("path_unknown_tolerance", 0.0)

        traj_cfg = self._config.get("trajectory", {})
        self._n_waypoints = traj_cfg.get("n_waypoints", 20)

        q_cfg = self._config.get("question", {})
        q_mode = q_cfg.get("mode", "template")
        if llm is not None:
            q_mode = "llm"

        if use_real_models and detector is None:
            from pathlib import Path
            project_root = Path(config_path).resolve().parent.parent.parent
            sam3_cfg_path = str(project_root / "baselines" / "sam3" / "config.yaml")
            sam2_cfg_path = str(project_root / "baselines" / "sam2" / "config.yaml")

            # Dynamically import baseline classes to avoid CUDA/weights dependency during CPU tests
            from baselines.sam3.sam3_standalone import SAM3Baseline
            from baselines.sam2.sam2_standalone import SAM2Baseline

            print(f"[EnvironmentalUncertaintyRunner] Loading real SAM3 baseline from: {sam3_cfg_path} ({device or 'auto'})")
            sam3_model = SAM3Baseline(config_path=sam3_cfg_path, device=device)

            print(f"[EnvironmentalUncertaintyRunner] Loading real SAM2 baseline from: {sam2_cfg_path} ({device or 'auto'})")
            sam2_model = SAM2Baseline(config_path=sam2_cfg_path, device=device)

            det_cfg = self._config.get("detector", {})
            overlap_th = det_cfg.get("overlap_threshold", 0.30)
            min_unk_pixel_frac = det_cfg.get("min_unknown_pixel_fraction", 0.02)

            self._detector = EnvironmentalUncertaintyDetector(
                sam3_model=sam3_model,
                sam2_model=sam2_model,
                overlap_threshold=overlap_th,
                min_unknown_pixel_fraction=min_unk_pixel_frac,
                sam3_queries=sam3_model.queries,
            )
        else:
            self._detector = detector or EnvironmentalUncertaintyDetector()

        self._question_gen = QuestionGenerator(mode=q_mode, llm=llm)
        self._map_updater = MapUpdater()
        self._gp_map = GPTraversabilityMap()
        # Cross-frame semantic knowledge: survives between run_scene() calls so
        # that confirmed labels ("grass is safe") are applied to new frames.
        self._terrain_knowledge = PersistentTerrainKnowledge()

        # World-coordinate GP and scene graph (multi-frame mode).
        # Populated only when run_scene_with_pose() is used.
        # Camera intrinsics default to RealSense D435i at image resolution.
        self._world_gp = WorldGPTraversabilityMap()
        self._world_scene_graph = WorldSceneGraph()
        self._camera_mount = CameraMount()   # overridable via set_camera_mount()
        # Focal length / principal point — updated per-frame from image shape.
        self._fx: float = 615.0
        self._fy: float = 615.0
        self._cx: float = 320.0
        self._cy: float = 240.0

    @property
    def terrain_knowledge(self) -> PersistentTerrainKnowledge:
        """Cross-frame semantic terrain knowledge (persists between frames)."""
        return self._terrain_knowledge

    @property
    def world_gp(self) -> WorldGPTraversabilityMap:
        """World-coordinate GP (persists across frames in multi-pose mode)."""
        return self._world_gp

    @property
    def world_scene_graph(self) -> WorldSceneGraph:
        """World-coordinate scene graph (true multi-image terrain map)."""
        return self._world_scene_graph

    def set_camera_mount(self, mount: CameraMount) -> None:
        """
        Override the camera mounting parameters used for pixel→world projection.

        Call this once before the first run_scene_with_pose() call when the robot's
        camera height and pitch differ from the CameraMount defaults (h=0.5m, p=15°).

        Args:
            mount: CameraMount with actual robot camera height_m and pitch_rad.
        """
        self._camera_mount = mount

    def set_camera_intrinsics(
        self, fx: float, fy: float, cx: float, cy: float
    ) -> None:
        """
        Override camera intrinsics used for pixel→world projection.

        Defaults are for a RealSense D435i at 640×480. Call this if your camera
        has different intrinsics (e.g. wide-angle lens, higher resolution).

        Args:
            fx, fy: Focal lengths in pixels.
            cx, cy: Principal point in pixels.
        """
        self._fx = fx
        self._fy = fy
        self._cx = cx
        self._cy = cy

    def reset_frame_state(self) -> None:
        """
        Clear the GP posterior between frames (multi-image sequential mode).

        Resets ONLY the per-frame GP (pixel-relative observations become stale
        when the robot moves).  Does NOT reset PersistentTerrainKnowledge —
        cross-frame label beliefs ("grass is safe") intentionally carry forward.

        Call this between run_scene() calls when the robot has moved significantly
        between frames and image-relative GP coordinates no longer correspond to
        the same real-world terrain. Without a call here, observations from prior
        frames accumulate in the GP with image-relative coordinates — meaning the
        same pixel position in two frames is treated as the same world location,
        which is incorrect when the robot has moved.

        When NOT to call this: when running multiple views of the same scene
        (camera jitter, slight rotation) where image coordinates are stable.
        When to call this: sequential frames from a moving robot.
        """
        self._gp_map.reset()

    def reset_all_knowledge(self) -> None:
        """
        Clear BOTH the per-frame GP and cross-frame terrain knowledge.

        Use this when starting a completely new navigation task where prior
        label-level beliefs should not carry over (e.g. different environment).
        """
        self._gp_map.reset()
        self._terrain_knowledge.reset()

    def run_scene(
        self,
        image,
        scene_id: str = "scene",
        scene_graph: Optional[SceneGraph] = None,
        robot_pose: Optional[Any] = None,
        camera_params: Optional[Any] = None,
        T_cam_to_base: Optional[np.ndarray] = None,
        depth_map: Optional[np.ndarray] = None,
        goal_pixel: Optional[Tuple[int, int]] = None,
        user_profile: Optional[UserProfile] = None,
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
            goal_pixel:   Optional (y, x) navigation goal in image coordinates.
                          When given, uses GoalDirectedTrajectoryGenerator so all
                          candidate paths lead toward the goal (May 19 mentor req).
                          When None, falls back to fixed-geometry TrajectoryGenerator.
            user_profile: Controls question verbosity and format (F27). Options:
                          verbosity="terse"|"standard"|"verbose",
                          preferred_format="question"|"option_list"|"statement".
                          Uses DEFAULT_PROFILE (standard verbosity) when None.

        Returns:
            EnvUncertaintyDecision with action and optional question.
        """
        import numpy as np
        image = np.asarray(image)
        h, w = image.shape[:2]

        # S1: Perception & Segmentation — SAM3/SAM2 detection
        result: DetectionResult = self._detector.detect(image)

        # S2a: Scene Understanding — seed GP from S1 known regions.
        # Runs BEFORE trajectory generation so LCB uses the current terrain priors.
        # User-feedback GP updates happen later in S4 (run_with_feedback), not here.
        self._seed_gp_from_detection(result, h, w)

        # S2b: Scene Understanding — trajectory generation + LCB scoring (MPPI-style).
        # Per may26 meeting (lines 302-309): generate all 3 Bezier paths then rank by
        # GP LCB simultaneously — same conceptual operation as MPPI.
        raw_trajectories = self._generate_trajectories(h, w, goal_pixel)
        scored_trajectories = self._score_trajectories(raw_trajectories, result, h, w, goal_pixel)

        # Select best trajectory by GP LCB (pessimistic safety ranking)
        best, best_lcb = self._select_best_trajectory_lcb(scored_trajectories, h, w)

        # S2c: Scene Understanding — build/update scene graph from live detections.
        # TerrainNode Dirichlet α initialized from SAM3 confidence.
        # High-confidence regions (≥0.73) → low entropy → no entropy ASK from S3.
        # If caller provides scene_graph (multi-frame tracking), update it in-place.
        _sg = scene_graph if scene_graph is not None else SceneGraph()
        _sg.update_from_gp(
            gp_map=self._gp_map,
            regions=result.known_regions,
            height=h,
            width=w,
            trajectories=scored_trajectories,
        )

        # Resolve on-path scene-graph nodes for Dirichlet entropy check (S3) and
        # Dirichlet update (S4).  Always non-empty now that the scene graph is built.
        on_path_nodes: List[TerrainNode] = []
        if best is not None:
            on_path_nodes = self._on_path_nodes(_sg, best, h, w)

        # S3: Uncertainty Resolution — decide action, generate question if needed.
        # Also identifies target_node (highest-entropy on-path node) so S4
        # (run_with_feedback) knows exactly which node's Dirichlet to update.
        action, question, target_node = self._decide_action(
            result, scored_trajectories, best, on_path_nodes,
            best_lcb=best_lcb, user_profile=user_profile,
        )

        # Optional: transform unknown region centroids to world coordinates
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
            target_node=target_node,
            decision_reason=_decision_reason(action, result, best_lcb),
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

    def run_with_feedback(
        self,
        image,
        user_response: str,
        scene_id: str = "scene",
        scene_graph: Optional[SceneGraph] = None,
        goal_pixel: Optional[Tuple[int, int]] = None,
        user_profile: Optional[UserProfile] = None,
    ) -> Tuple[EnvUncertaintyDecision, EnvUncertaintyDecision]:
        """
        Run the full pipeline, incorporate user feedback, then replan (D18).

        This implements the complete ask→feedback→replan loop:
          1. run_scene()           → initial decision (usually ASK)
          2. parse_user_response() → extract terrain label, confidence, safety
          3. GP apply_user_feedback() → update traversability at uncertain waypoints
          4. run_scene() again     → replanned decision with updated GP beliefs

        Args:
            image:         (H, W, 3) uint8 numpy array (RGB).
            user_response: Free-text user answer (e.g. "it's wet grass, be careful").
            scene_id:      Identifier for logging.
            scene_graph:   Optional SceneGraph for Dirichlet entropy checks.
            goal_pixel:    Optional (y, x) navigation goal; activates goal-directed paths.

        Returns:
            (initial_decision, replanned_decision) — both EnvUncertaintyDecision.
            Compare robot_action in both to see whether user feedback resolved the issue.
        """
        import numpy as np
        image_arr = np.asarray(image)
        h, w = image_arr.shape[:2]

        # S1-S3: Initial pipeline run (detect → scene understanding → decide/ask)
        initial = self.run_scene(image_arr, scene_id, scene_graph, goal_pixel, user_profile)

        # S3 (parse response): keyword matching → terrain_label, is_traversable,
        # label_confidence. Default confidence=0.70 when no keyword found (may26 line 374).
        parsed = parse_user_response_rich(user_response)

        # S4 (GP update): Bayesian update at uncertain waypoints on the best trajectory.
        # Updates only the pixels the robot was actually uncertain about — area-specific,
        # not semantic-category-wide (May 19 mentor: local updates are the right approach).
        #
        # When initial.best_trajectory is None (all paths passed through unknown terrain),
        # fall back to the direct trajectory so the GP still receives the user's safety
        # judgment. On replan (Step 10), _select_best_trajectory_lcb will accept this
        # path if its updated LCB now meets the safety threshold.
        update_trajectory = initial.best_trajectory
        if update_trajectory is None:
            raw_trajs = self._generate_trajectories(h, w, goal_pixel)
            # Use the direct path (first trajectory) — it's the shortest route to goal.
            update_trajectory = raw_trajs[0] if raw_trajs else None

        if update_trajectory is not None:
            # Primary update: waypoints on the selected trajectory (path-specific).
            for wy, wx in update_trajectory.waypoints:
                self._gp_map.apply_user_feedback(
                    pixel_y=wy,
                    pixel_x=wx,
                    is_traversable=parsed.is_traversable,
                    height=h,
                    width=w,
                )

            # Region-mask update: sample N evenly-spaced pixels from EACH unknown
            # region and apply the user's answer to them.  The robot asked about a
            # specific region; the user's response applies to the whole visible region,
            # not just the 20 waypoints the trajectory happened to cross.
            # Re-running detection is deterministic and fast (~10 ms), so we do it
            # here rather than threading the detection result through run_scene().
            N_REGION_SAMPLES = 9  # 3×3 interior coverage per unknown region
            fresh_detection: DetectionResult = self._detector.detect(np.asarray(image_arr))
            for region in fresh_detection.unknown_regions:
                mask = np.asarray(region.mask)
                ys, xs = np.where(mask)
                if len(ys) == 0:
                    continue
                indices = np.linspace(0, len(ys) - 1, N_REGION_SAMPLES, dtype=int)
                for idx in indices:
                    self._gp_map.apply_user_feedback(
                        pixel_y=int(ys[idx]),
                        pixel_x=int(xs[idx]),
                        is_traversable=parsed.is_traversable,
                        height=h,
                        width=w,
                    )

        # S4 (Dirichlet update): update the specific uncertain node's class distribution.
        # target_node is the highest-entropy on-path TerrainNode from S3.
        # Conjugate prior update: α[label] += label_confidence.
        # Lowers semantic_entropy() so the robot won't keep re-asking the same node.
        # Only fires when robot was in ASK state and a terrain label was parsed.
        if (
            parsed.terrain_label
            and initial.target_node is not None
            and initial.robot_action == "ASK"
        ):
            initial.target_node.update_from_user(parsed.terrain_label, parsed.label_confidence)

        # Cross-frame knowledge update: record this user response at the label level
        # so future frames benefit from the same confirmation without re-asking.
        # PersistentTerrainKnowledge is NOT reset between frames (unlike the GP),
        # so a confirmed "grass is safe" here propagates to the next frame's GP seed.
        if parsed.terrain_label:
            self._terrain_knowledge.update_from_feedback(
                label=parsed.terrain_label,
                is_traversable=parsed.is_traversable,
                confidence=parsed.label_confidence,
            )

        # S5: Replan — re-run S1-S3 with updated GP posterior.
        # Per june1meeting line 10: "if there is more uncertainties, do another round."
        # GP now incorporates user's safety judgment, so LCB scores shift accordingly.
        replanned = self.run_scene(image_arr, scene_id + "_replanned", scene_graph, goal_pixel, user_profile)

        return initial, replanned

    def run_scene_with_pose(
        self,
        image,
        pose: RobotPose,
        scene_id: str = "scene",
        goal_pixel: Optional[Tuple[int, int]] = None,
        user_profile: Optional[UserProfile] = None,
        depth_map: Optional[Any] = None,
    ) -> EnvUncertaintyDecision:
        """
        Run the pipeline with robot pose — enables correct multi-frame GP accumulation.

        Unlike run_scene(), observations from this call are stored in metric world
        coordinates using `pose`. The WorldGPTraversabilityMap and WorldSceneGraph
        persist across calls so the robot builds a growing terrain map as it moves.

        This implements the May 19 extension requirement: visual odometry or GPS
        converts image pixels to world (x, y) metres so observations from frame 1
        at world position (2.0, 0.5) correctly persist when the robot is at (5.0, 0.0)
        in frame 10. The per-frame GP (run_scene) is still reset between frames; the
        world GP is never reset except via reset_world_knowledge().

        Args:
            image:      (H, W, 3) uint8 numpy array (RGB).
            pose:       Current robot pose in world frame from odometry or GPS.
                        Use MockForwardOdometry, OpticalFlowOdometry, or GPSOdometry
                        from system.env_uncertainty.world_coords.
            scene_id:   Identifier string for logging.
            goal_pixel: Optional (y, x) goal in image coordinates (activates
                        GoalDirectedTrajectoryGenerator).
            user_profile: Question verbosity/format profile.
            depth_map:  Optional (H, W) float depth image in metres. When None,
                        monocular ground-plane depth estimation is used.

        Returns:
            EnvUncertaintyDecision — same structure as run_scene().
            The world_gp and world_scene_graph properties are also updated.

        Example usage with MockForwardOdometry::

            from system.env_uncertainty.world_coords import MockForwardOdometry
            odometry = MockForwardOdometry(speed_mps=0.5, fps=5.0)
            runner = EnvironmentalUncertaintyRunner(config_path)
            for img in frame_sequence:
                pose = odometry.next_pose()
                decision = runner.run_scene_with_pose(img, pose, goal_pixel=(50, 320))

        Example usage with OpticalFlowOdometry::

            from system.env_uncertainty.world_coords import OpticalFlowOdometry
            odometry = OpticalFlowOdometry(fx=615, fy=615, cx=320, cy=240)
            runner.set_camera_mount(CameraMount(height_m=0.6, pitch_rad=0.3))
            for img in frame_sequence:
                pose = odometry.update(img, dt=0.2)
                decision = runner.run_scene_with_pose(img, pose)
        """
        import numpy as np
        image = np.asarray(image)
        h, w = image.shape[:2]

        # Update intrinsics to match image resolution (scale from 640×480 baseline)
        scale_x = w / 640.0
        scale_y = h / 480.0
        fx = self._fx * scale_x
        fy = self._fy * scale_y
        cx = self._cx * scale_x
        cy = self._cy * scale_y

        # S1: Perception & Segmentation (SAM3 + SAM2)
        result: DetectionResult = self._detector.detect(image)

        # S2a: Scene Understanding — seed per-frame GP from S1 detections
        self._seed_gp_from_detection(result, h, w)

        # S2b: Scene Understanding — project known regions to world coordinates
        #          and add to the world GP. This is the key multi-frame operation:
        #          observations accumulate at real-world (x, y) positions regardless
        #          of which frame they came from.
        depth_np = np.asarray(depth_map) if depth_map is not None else None
        for region in result.known_regions:
            trav = self._terrain_knowledge.adjusted_traversability(
                region.label, default_score=region.traversability
            )
            world_pts = mask_centroids_to_world(
                mask=region.mask,
                pose=pose,
                fx=fx, fy=fy,
                cx_principal=cx,
                cy_principal=cy,
                mount=self._camera_mount,
                n_samples=9,
                depth_map=depth_np,
            )
            for (x_w, y_w) in world_pts:
                self._world_gp.add_world_observation(x_w, y_w, trav)
                self._world_scene_graph.upsert_world_region(
                    label=region.label, x_w=x_w, y_w=y_w,
                    gp_mean=trav, gp_variance=0.1,
                )

        # S2c: Scene Understanding — trajectory generation + LCB scoring (combined MPPI-style)
        raw_trajectories = self._generate_trajectories(h, w, goal_pixel)
        scored_trajectories = self._score_trajectories(raw_trajectories, result, h, w, goal_pixel)
        best, best_lcb = self._select_best_trajectory_lcb(scored_trajectories, h, w)

        # Collect on-path world-scene-graph nodes for Dirichlet entropy check.
        # Convert trajectory waypoints to world coordinates and query WorldSceneGraph.
        on_path_nodes: List[TerrainNode] = []
        if best is not None:
            for (wy, wx) in best.waypoints:
                pt = pixel_to_world(wy, wx, pose, fx, fy, cx, cy, self._camera_mount,
                                    depth_m=None)
                if pt is not None:
                    x_w, y_w = pt
                    nodes = self._world_scene_graph.nodes_near_world(x_w, y_w, radius_m=self._camera_mount.height_m * 2)
                    for node in nodes:
                        key = (node.label, node.position_cell_id)
                        if key not in {(n.label, n.position_cell_id) for n in on_path_nodes}:
                            on_path_nodes.append(node)

        # S3: Uncertainty Resolution — decide action, returns target_node for S4 update
        action, question, target_node = self._decide_action(
            result, scored_trajectories, best, on_path_nodes,
            best_lcb=best_lcb, user_profile=user_profile,
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
            target_node=target_node,
            decision_reason=_decision_reason(action, result, best_lcb),
        )

    def reset_world_knowledge(self) -> None:
        """
        Clear the world-coordinate GP and WorldSceneGraph.

        Use when starting in a completely new environment where accumulated world
        observations should not carry over. Does not affect PersistentTerrainKnowledge
        (label-level beliefs like "grass is safe" are environment-independent).
        """
        self._world_gp.reset()
        self._world_scene_graph = WorldSceneGraph()

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
        # help_rate: fraction of all scenarios where the robot asked for help.
        # High help_rate → over-cautious; low help_rate may miss real uncertainty.
        # Baseline compare: always_ask has help_rate=1.0, always_act has help_rate=0.0.
        n_asked = n_correct_ask + (n_should_proceed - n_correct_proceed)
        help_rate = n_asked / n if n > 0 else 0.0

        return {
            "n_scenarios": n,
            "n_should_ask": n_should_ask,
            "n_should_proceed": n_should_proceed,
            "AAR": round(aar, 4),        # Appropriate Ask Rate: asked when should ask
            "SAR": round(sar, 4),        # Spurious Ask Rate: asked when should proceed
            "help_rate": round(help_rate, 4),  # overall fraction of scenes where robot asked
            "n_correct_ask": n_correct_ask,
            "n_correct_proceed": n_correct_proceed,
            "n_asked": n_asked,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _generate_trajectories(
        self,
        height: int,
        width: int,
        goal_pixel: Optional[Tuple[int, int]],
    ) -> List[Trajectory]:
        """
        Return raw (unscored) candidate trajectories.

        When goal_pixel is provided: uses GoalDirectedTrajectoryGenerator so all
        three paths lead toward the navigation goal (May 19 mentor requirement).
        When goal_pixel is None: falls back to the original fixed-geometry generator.
        """
        if goal_pixel is not None:
            gen = GoalDirectedTrajectoryGenerator(height, width, n_waypoints=self._n_waypoints)
            # Robot is assumed to start at bottom-center of the image (camera-frame
            # convention: robot is at the bottom, looking forward/up in the image).
            start = (height - 1, width // 2)
            return gen.generate_toward_goal(start, goal_pixel)
        else:
            gen = TrajectoryGenerator(height, width, n_waypoints=self._n_waypoints)
            return gen.generate_trajectories()

    def _score_trajectories(
        self,
        raw_trajectories: List[Trajectory],
        result: DetectionResult,
        height: int,
        width: int,
        goal_pixel: Optional[Tuple[int, int]],
    ) -> List[Trajectory]:
        """Score raw trajectories against the traversability map."""
        if goal_pixel is not None:
            scorer = GoalDirectedTrajectoryGenerator(height, width, n_waypoints=self._n_waypoints)
            return [scorer.score_trajectory(t, result.traversability_map) for t in raw_trajectories]
        else:
            scorer = TrajectoryGenerator(height, width, n_waypoints=self._n_waypoints)
            return [scorer.score_trajectory(t, result.traversability_map) for t in raw_trajectories]

    def _seed_gp_from_detection(
        self, result: DetectionResult, height: int, width: int
    ) -> None:
        """
        Add one GP observation per known region, at its mask centroid.

        This is S2a of the pipeline: the GP posterior is seeded with SAM3's
        traversability labels before trajectory LCB scoring (S2b).
        Unknown regions are intentionally excluded — the GP only learns from
        regions the robot already has a label for.

        Cross-frame knowledge: if PersistentTerrainKnowledge has previously
        recorded user feedback for a label (e.g. "grass is safe" from frame 1),
        the adjusted traversability is used instead of the static default.
        This lets confirmed labels propagate to new frames automatically.
        """
        import numpy as np
        for region in result.known_regions:
            mask = np.asarray(region.mask)
            if not np.any(mask):
                continue
            ys, xs = np.where(mask)
            cy, cx = int(np.mean(ys)), int(np.mean(xs))
            # Use cross-frame knowledge when available; fall back to static score.
            trav = self._terrain_knowledge.adjusted_traversability(
                region.label, default_score=region.traversability
            )
            self._gp_map.add_observation(cy, cx, trav, height, width)

    def _select_best_trajectory_lcb(
        self,
        trajectories: List[Trajectory],
        height: int,
        width: int,
    ) -> tuple:
        """
        Select the safest trajectory by GP LCB score.

        Primary selection: trajectories where passes_through_unknown=False.
        Fallback (user-feedback replan): when ALL trajectories pass through
        unknown territory AND the GP has observations (i.e. user feedback was
        applied), accept any trajectory whose GP LCB exceeds ask_threshold.
        This lets a user-blessed "unknown" path become the selected best path on
        the second run_scene() call inside run_with_feedback() (Step 10).

        Returns (best_trajectory, best_lcb_score). Both None when no trajectory
        meets the safety criteria, which triggers ASK in _decide_action.
        """
        safe = [t for t in trajectories if not t.passes_through_unknown]
        if safe:
            scored = [
                (t, self._gp_map.score_trajectory_lcb(t.waypoints, height, width))
                for t in safe
            ]
            best, best_lcb = max(scored, key=lambda ts: ts[1])
            return best, best_lcb

        # All paths pass through unknown — check whether the GP posterior (updated
        # from user feedback) now rates any path as safe enough to proceed.
        if self._gp_map.n_observations > 0:
            scored = [
                (t, self._gp_map.score_trajectory_lcb(t.waypoints, height, width))
                for t in trajectories
            ]
            best, best_lcb = max(scored, key=lambda ts: ts[1])
            if best_lcb >= self._ask_threshold:
                return best, best_lcb

        return None, None

    def _decide_action(
        self,
        result: DetectionResult,
        trajectories: List[Trajectory],
        best_trajectory: Optional[Trajectory],
        on_path_nodes: Optional[List[TerrainNode]] = None,
        best_lcb: Optional[float] = None,
        user_profile: Optional[UserProfile] = None,
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
            (action_str, question_str_or_None, target_node_or_None)
            target_node is the most uncertain on-path node (for S8 Dirichlet update).
        """
        if on_path_nodes is None:
            on_path_nodes = []

        # Extract top-k terrain candidates from the most uncertain on-path node.
        # Passed into question gen so the robot names what it thinks it's seeing
        # instead of saying "unrecognized area" generically.
        top_k = self._top_k_from_nodes(on_path_nodes)

        # The most uncertain on-path node — returned for S8 Dirichlet update.
        # S5 identifies this node so S8 knows exactly which node's class
        # distribution to update with user feedback (one node at a time).
        target_node = (
            max(on_path_nodes, key=lambda n: n.semantic_entropy())
            if on_path_nodes else None
        )

        profile = user_profile or DEFAULT_PROFILE

        # Coverage small enough to be GT labeling noise: skip LCB STOP and the
        # path-through-unknown guard below.  Sparse single-centroid GP seeding
        # gives unreliable LCB estimates at this scale, so we trust known-region
        # traversability instead.
        path_noise = result.unknown_coverage < self._path_unknown_tolerance

        # Very large unknown area → STOP (robot cannot proceed safely)
        if result.unknown_coverage >= self._stop_threshold:
            question = self._question_gen.generate(
                result, trajectories, user_profile=profile, top_k_classes=top_k
            )
            return "STOP", question, target_node

        # High Dirichlet semantic entropy on planned path → ASK.
        # The robot doesn't know what terrain class it's heading into, so it
        # must clarify before committing — regardless of unknown_coverage.
        if any(
            node.semantic_entropy() > self._entropy_threshold
            for node in on_path_nodes
        ):
            question = self._question_gen.generate(
                result, trajectories, user_profile=profile, top_k_classes=top_k
            )
            return "ASK", question, target_node

        # GP LCB-based STOP: best safe path has dangerously low traversability.
        # path_noise can bypass this when the GP is merely uncertain (LCB ≥ 0)
        # but NOT when LCB is negative — negative LCB signals genuinely dangerous
        # terrain (traversability near 0), not just sparse-observation uncertainty.
        # Guard: n_observations > 0 prevents false STOPs from the uninformative
        # prior (prior LCB = 0.5 - 1.5*0.4 = -0.1 < threshold without any data).
        # Bypass only when best_lcb indicates terrain is not genuinely dangerous.
        # Dense-seeded dangerous terrain (e.g., adj_trav=0.03) gives LCB ≈ 0.02,
        # while sparse-seeded safe terrain (gravel=0.70, grass=0.60) gives LCB ≥ 0.10.
        # Floor=0.05 separates these cleanly. best_lcb=None (no safe trajectory found)
        # is also safe to bypass — path_noise will then trigger PROCEED directly.
        _LCB_NOISE_BYPASS_FLOOR = 0.05
        lcb_noise_bypass = path_noise and (best_lcb is None or best_lcb >= _LCB_NOISE_BYPASS_FLOOR)
        if (
            not lcb_noise_bypass
            and best_lcb is not None
            and self._gp_map.n_observations > 0
            and best_lcb < self._lcb_stop_threshold
        ):
            question = self._question_gen.generate(
                result, trajectories, user_profile=profile, top_k_classes=top_k
            )
            return "STOP", question, target_node

        # No unknown regions, or unknown area is small and off-path → PROCEED.
        # path_noise bypasses the path-through-unknown requirement for tiny coverage.
        if not result.has_unknown or (
            result.unknown_coverage < self._ask_threshold
            and (path_noise or (best_trajectory is not None and not best_trajectory.passes_through_unknown))
        ):
            return "PROCEED", None, None

        # Unknown region exists and robot's path is affected → ASK
        question = self._question_gen.generate(
            result, trajectories, user_profile=profile, top_k_classes=top_k
        )
        return "ASK", question, target_node

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
