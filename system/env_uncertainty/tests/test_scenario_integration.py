"""
Real-world scenario integration tests — Steps 1–9 end-to-end.

Based on concrete scenarios defined in mentor meetings (april27, april29, aprl25):

  Scenario 0 — Unknown terrain on path (the simplest baseline case)
    Robot navigates forward. SAM2 detects an unknown region covering 30% of the
    path. The scene graph has a high-entropy node on the trajectory. Robot should
    ASK with a grounded question naming its top-3 terrain guesses.

  Scenario 1 — Wet grass (known terrain, but uncertain traversability)
    Robot sees a known grass region but the user previously told it the grass
    was wet. GP posterior should reflect reduced traversability. Robot should
    still PROCEED if no unknown regions and best trajectory is clear.

  Scenario 2 — Clear sidewalk path (fully known, high traversability)
    Robot sees only sidewalk ahead. No unknown regions. All trajectories are
    clear. Robot should PROCEED without asking.

  Scenario 3 — User responds and beliefs update (Steps 8 + 9)
    After asking (Scenario 0), user replies "I think it's mud, seems okay."
    Pipeline should: parse → terrain_label=mud, is_traversable=True,
    confidence=0.60. Then GP and scene graph both update correctly.

  Scenario 4 — STOP: almost entirely unknown scene
    Robot sees >80% unknown terrain. No safe path exists. Robot must STOP, not ASK.

  Scenario 5 — Entropy drops after user confirms (robot stops re-asking)
    Once the user confirms a terrain class, Dirichlet entropy falls below
    threshold. Same scene on next pass → PROCEED instead of ASK.
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from system.env_uncertainty.detector import DetectionResult, RegionInfo
from system.env_uncertainty.gp_traversability import GPTraversabilityMap
from system.env_uncertainty.map_updater import parse_user_response_rich
from system.env_uncertainty.question_generator import QuestionGenerator, generate_question_template
from system.env_uncertainty.runner import EnvUncertaintyDecision, EnvironmentalUncertaintyRunner
from system.env_uncertainty.scene_graph import SceneGraph, TERRAIN_CLASSES
from system.env_uncertainty.traversability import TraversabilityMap

CONFIG_PATH = str(Path(__file__).parents[1] / "config.yaml")
H, W = 100, 100
IMAGE = np.zeros((H, W, 3), dtype=np.uint8)


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _mask_top_fraction(frac: float) -> np.ndarray:
    """Boolean mask covering the top `frac` of the image (the robot's forward path)."""
    mask = np.zeros((H, W), dtype=bool)
    rows = int(H * frac)
    mask[:rows, :] = True
    return mask


def _region(label: str, mask: np.ndarray, traversability: float) -> RegionInfo:
    return RegionInfo(
        label=label,
        mask=mask,
        confidence=0.85,
        pixel_fraction=float(mask.sum()) / (H * W),
        source="sam3" if label != "unknown" else "sam2",
        traversability=traversability,
    )


def _make_detector(known_regions, unknown_regions, unknown_coverage: float, all_zeros: bool = False):
    """Build a mock detector with explicit known/unknown region lists."""
    mock = MagicMock()
    tmap = TraversabilityMap.create(H, W)

    if not all_zeros:
        for r in known_regions:
            tmap = tmap.update_region(r.mask, r.label)
    for r in unknown_regions:
        tmap = tmap.update_region(r.mask, "unknown")

    mock.detect.return_value = DetectionResult(
        known_regions=known_regions,
        unknown_regions=unknown_regions,
        image_shape=(H, W),
        sam3_coverage=sum(r.pixel_fraction for r in known_regions),
        unknown_coverage=unknown_coverage,
        has_unknown=len(unknown_regions) > 0,
        traversability_map=tmap,
    )
    return mock


def _runner(detector) -> EnvironmentalUncertaintyRunner:
    return EnvironmentalUncertaintyRunner(CONFIG_PATH, detector=detector)


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO 0 — Unknown terrain on path
# Mentor reference: "Robot observes an unknown patch of terrain ahead and cannot
#                   determine whether it is traversable." (april29meeting.txt)
# ══════════════════════════════════════════════════════════════════════════════

class TestScenario0UnknownTerrainOnPath:
    """30% unknown terrain covering the forward path → robot must ASK."""

    def _setup(self):
        unknown_mask = _mask_top_fraction(0.30)
        unknown = [_region("unknown", unknown_mask, traversability=0.0)]
        detector = _make_detector([], unknown, unknown_coverage=0.30, all_zeros=True)
        return _runner(detector)

    def test_robot_action_is_ask(self):
        runner = self._setup()
        d = runner.run_scene(IMAGE)
        assert d.robot_action == "ASK"

    def test_ask_has_a_question(self):
        runner = self._setup()
        d = runner.run_scene(IMAGE)
        assert isinstance(d.question, str) and len(d.question) > 5

    def test_high_entropy_node_triggers_ask_with_grounded_question(self):
        # When best trajectory exists and a high-entropy scene graph node sits on it,
        # the question must name terrain candidates, not say "unrecognized area".
        # Bottom 85% is known grass (traversable) so safe trajectories exist.
        # Top 15% is unknown (but covered by the scene graph entropy trigger).
        unknown_mask = _mask_top_fraction(0.15)
        grass_mask = ~unknown_mask
        unknown_r = _region("unknown", unknown_mask, traversability=0.0)
        grass_r = _region("grass", grass_mask, traversability=0.90)
        detector = _make_detector([grass_r], [unknown_r], unknown_coverage=0.15, all_zeros=False)
        runner = _runner(detector)
        # High-entropy node at centre of image — on the forward trajectory
        sg = SceneGraph()
        sg.upsert_region("unknown", pixel_y=50, pixel_x=50, height=H, width=W)
        d = runner.run_scene(IMAGE, scene_graph=sg)
        assert d.robot_action == "ASK"
        # Grounded question: at least one terrain class name should appear
        assert any(cls in d.question.lower() for cls in TERRAIN_CLASSES)

    def test_question_is_grounded_when_dirichlet_provides_top_k(self):
        # Directly test question gen with top_k from a high-entropy node
        sg = SceneGraph()
        node = sg.upsert_region("unknown", pixel_y=10, pixel_x=50, height=H, width=W)
        top_k = node.top_k_classes(k=3)
        from system.env_uncertainty.traversability import TraversabilityMap
        tmap = TraversabilityMap.create(H, W)
        result = DetectionResult(
            known_regions=[], unknown_regions=[],
            image_shape=(H, W), sam3_coverage=0.0,
            unknown_coverage=0.30, has_unknown=True, traversability_map=tmap,
        )
        q = generate_question_template(result, top_k_classes=top_k)
        # All 3 candidates must appear in the question
        for label, _ in top_k:
            assert label in q.lower()

    def test_unknown_coverage_reported_correctly(self):
        runner = self._setup()
        d = runner.run_scene(IMAGE)
        assert d.unknown_coverage == pytest.approx(0.30)

    def test_has_unknown_is_true(self):
        runner = self._setup()
        d = runner.run_scene(IMAGE)
        assert d.has_unknown is True


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO 1 — Wet grass (known terrain, GP feedback adjusts traversability)
# Mentor reference: "Robot sees wet grass, user says keep going." (aprl25meeting.txt)
# ══════════════════════════════════════════════════════════════════════════════

class TestScenario1WetGrass:
    """Known grass region but GP has been told it's wet (unsafe). No unknowns."""

    def test_gp_feedback_unsafe_lowers_traversability(self):
        gp = GPTraversabilityMap()
        before = gp.predict(10, 50, H, W).mu  # prior ≈ 0.5
        # Simulate user saying "the grass is wet, not safe"
        gp.apply_user_feedback(10, 50, is_traversable=False, height=H, width=W)
        after = gp.predict(10, 50, H, W).mu
        assert after < before, "Unsafe feedback must reduce traversability estimate"

    def test_lcb_at_wet_location_is_low(self):
        # LCB = mu - beta*sigma. After unsafe feedback, LCB must be well below 0.5
        gp = GPTraversabilityMap()
        gp.apply_user_feedback(10, 50, is_traversable=False, height=H, width=W)
        lcb = gp.predict(10, 50, H, W).lcb
        assert lcb < 0.4

    def test_proceed_when_grass_is_clear_no_unknown(self):
        # If grass region is known, no unknowns, traversability good → PROCEED
        grass_mask = _mask_top_fraction(0.80)
        known = [_region("grass", grass_mask, traversability=0.90)]
        detector = _make_detector(known, [], unknown_coverage=0.0, all_zeros=False)
        runner = _runner(detector)
        d = runner.run_scene(IMAGE)
        assert d.robot_action == "PROCEED"

    def test_no_question_when_proceeding(self):
        grass_mask = _mask_top_fraction(0.80)
        known = [_region("grass", grass_mask, traversability=0.90)]
        detector = _make_detector(known, [], unknown_coverage=0.0, all_zeros=False)
        runner = _runner(detector)
        d = runner.run_scene(IMAGE)
        assert d.question is None


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO 2 — Clear sidewalk path
# Mentor reference: "Robot sees cracked pavement ahead." (aprl25meeting.txt)
# Here we test the happy path: clear sidewalk → PROCEED
# ══════════════════════════════════════════════════════════════════════════════

class TestScenario2ClearSidewalk:
    """Fully known sidewalk, no unknown regions, high traversability → PROCEED."""

    def _setup(self):
        # Use 8 horizontal strips so the GP is seeded at multiple y positions,
        # giving it enough spatial coverage to stay above the LCB STOP threshold
        # (0.20) along the full trajectory (rows 20–99).  A single centroid at
        # (50,50) leaves σ≈1.0 at row 99 → LCB = 0.95 - 1.5*1.0 = -0.55 → STOP.
        step = H // 8
        strips = []
        for i in range(8):
            mask = np.zeros((H, W), dtype=bool)
            mask[i * step : min((i + 1) * step, H), :] = True
            strips.append(_region("sidewalk", mask, traversability=0.95))
        detector = _make_detector(strips, [], unknown_coverage=0.0, all_zeros=False)
        return _runner(detector)

    def test_robot_proceeds_on_clear_sidewalk(self):
        runner = self._setup()
        d = runner.run_scene(IMAGE)
        assert d.robot_action == "PROCEED"

    def test_no_question_on_clear_path(self):
        runner = self._setup()
        d = runner.run_scene(IMAGE)
        assert d.question is None

    def test_no_unknown_regions(self):
        runner = self._setup()
        d = runner.run_scene(IMAGE)
        assert d.has_unknown is False
        assert d.n_unknown_regions == 0

    def test_gp_seeded_from_sidewalk_observation(self):
        # GP should have at least one observation from the known sidewalk regions
        runner = self._setup()
        runner.run_scene(IMAGE)
        assert runner._gp_map.n_observations >= 1

    def test_sidewalk_gp_mean_near_high_traversability(self):
        # After seeing sidewalk (trav=0.95), GP posterior at that centroid should be high
        runner = self._setup()
        runner.run_scene(IMAGE)
        mu = runner._gp_map.predict(H // 2, W // 2, H, W).mu
        assert mu > 0.5


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO 3 — User response parsing + belief update (Steps 8 + 9)
# Mentor reference: Full loop: robot asks → user responds → robot updates.
# ══════════════════════════════════════════════════════════════════════════════

class TestScenario3UserResponseAndUpdate:
    """After asking, user says 'I think it's mud, seems okay.' Both GP and
    scene graph must update in the correct direction."""

    USER_RESPONSE = "I think it's mud, looks safe"

    def test_parse_confidence_is_0_60(self):
        # "think" → 0.60 (document wrongly said 0.50)
        r = parse_user_response_rich(self.USER_RESPONSE)
        assert r.label_confidence == pytest.approx(0.60)

    def test_parse_terrain_label_is_mud(self):
        r = parse_user_response_rich(self.USER_RESPONSE)
        assert r.terrain_label == "mud"

    def test_parse_is_traversable_true(self):
        # "okay" is a positive affordance signal
        r = parse_user_response_rich(self.USER_RESPONSE)
        assert r.is_traversable is True

    def test_gp_updates_after_user_says_safe(self):
        r = parse_user_response_rich(self.USER_RESPONSE)
        gp = GPTraversabilityMap()
        before = gp.n_observations
        gp.apply_user_feedback(10, 50, is_traversable=r.is_traversable, height=H, width=W)
        assert gp.n_observations == before + 1

    def test_gp_mean_increases_after_safe_feedback(self):
        gp = GPTraversabilityMap()
        before = gp.predict(10, 50, H, W).mu
        gp.apply_user_feedback(10, 50, is_traversable=True, height=H, width=W)
        after = gp.predict(10, 50, H, W).mu
        assert after > before

    def test_dirichlet_mud_alpha_increases(self):
        r = parse_user_response_rich(self.USER_RESPONSE)
        sg = SceneGraph()
        node = sg.upsert_region("unknown", pixel_y=10, pixel_x=50, height=H, width=W)
        mud_idx = TERRAIN_CLASSES.index("mud")
        before = node.dirichlet_alpha[mud_idx]
        node.update_from_user(r.terrain_label, confidence=r.label_confidence)
        assert node.dirichlet_alpha[mud_idx] > before

    def test_entropy_decreases_after_update(self):
        sg = SceneGraph()
        node = sg.upsert_region("unknown", pixel_y=10, pixel_x=50, height=H, width=W)
        entropy_before = node.semantic_entropy()
        node.update_from_user("mud", confidence=5.0)
        assert node.semantic_entropy() < entropy_before

    def test_full_loop_gp_and_scene_graph_both_update(self):
        """Full loop: parse response → update GP + scene graph simultaneously."""
        r = parse_user_response_rich(self.USER_RESPONSE)
        gp = GPTraversabilityMap()
        sg = SceneGraph()
        node = sg.upsert_region("unknown", pixel_y=10, pixel_x=50, height=H, width=W)

        mud_idx = TERRAIN_CLASSES.index("mud")
        alpha_before = node.dirichlet_alpha[mud_idx]
        obs_before = gp.n_observations

        # Step 9: both updates fire from the same parsed response
        gp.apply_user_feedback(10, 50, is_traversable=r.is_traversable, height=H, width=W)
        node.update_from_user(r.terrain_label, confidence=r.label_confidence)

        assert gp.n_observations == obs_before + 1
        assert node.dirichlet_alpha[mud_idx] > alpha_before


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO 4 — STOP: almost entirely unknown scene
# Mentor reference: "Robot sees an unknown patch covering most of the view."
# ══════════════════════════════════════════════════════════════════════════════

class TestScenario4StopOnMassiveUnknown:
    """90% unknown coverage exceeds stop threshold (0.80) → robot must STOP."""

    def _setup(self):
        unknown_mask = _mask_top_fraction(0.90)
        unknown = [_region("unknown", unknown_mask, traversability=0.0)]
        detector = _make_detector([], unknown, unknown_coverage=0.90, all_zeros=True)
        return _runner(detector)

    def test_robot_action_is_stop(self):
        runner = self._setup()
        d = runner.run_scene(IMAGE)
        assert d.robot_action == "STOP"

    def test_stop_has_question(self):
        # STOP must include a question asking the user what to do
        runner = self._setup()
        d = runner.run_scene(IMAGE)
        assert isinstance(d.question, str) and len(d.question) > 5

    def test_stop_not_ask_at_90_percent(self):
        # 90% is above stop threshold; must not downgrade to ASK
        runner = self._setup()
        d = runner.run_scene(IMAGE)
        assert d.robot_action != "ASK"

    def test_stop_coverage_reported(self):
        runner = self._setup()
        d = runner.run_scene(IMAGE)
        assert d.unknown_coverage == pytest.approx(0.90)


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO 5 — Robot stops re-asking after user confirms terrain
# Mentor reference: Robot should not repeatedly ask about terrain it already knows.
# ══════════════════════════════════════════════════════════════════════════════

class TestScenario5NoRepeatAsk:
    """After user confirms a terrain class, Dirichlet entropy drops.
    On next pass through same area, entropy is below threshold → no ASK."""

    def test_entropy_below_threshold_after_strong_confirmation(self):
        sg = SceneGraph()
        node = sg.upsert_region("grass", pixel_y=10, pixel_x=50, height=H, width=W)
        entropy_threshold = 1.5  # from config

        # Before confirmation: uniform prior → high entropy
        assert node.semantic_entropy() > entropy_threshold

        # User confirms "grass" with high confidence — needs ~50 pseudocounts
        # to push entropy below 1.5 with K=21 classes
        for _ in range(5):
            node.update_from_user("grass", confidence=10.0)

        # After confirmation: entropy must drop below threshold
        assert node.semantic_entropy() < entropy_threshold

    def test_confirmed_node_does_not_trigger_ask(self):
        # With a confirmed low-entropy node on path, runner should PROCEED (no unknown)
        grass_mask = _mask_top_fraction(0.80)
        known = [_region("grass", grass_mask, traversability=0.90)]
        detector = _make_detector(known, [], unknown_coverage=0.0, all_zeros=False)
        runner = _runner(detector)

        # Build scene graph with confirmed grass node
        sg = SceneGraph()
        node = sg.upsert_region("grass", pixel_y=10, pixel_x=50, height=H, width=W)
        for _ in range(5):
            node.update_from_user("grass", confidence=10.0)

        assert node.semantic_entropy() < 1.5  # confirm entropy is low before running

        d = runner.run_scene(IMAGE, scene_graph=sg)
        assert d.robot_action == "PROCEED"

    def test_should_skip_asking_after_gp_confirmed(self):
        # scene_graph.should_skip_asking() returns True once user_confirmed + low GP variance
        sg = SceneGraph()
        node = sg.upsert_region("grass", pixel_y=10, pixel_x=50, height=H, width=W)

        # GP variance starts at 0.0 (no GP data seeded → default), user_confirmed=False
        skip, _ = sg.should_skip_asking("grass", cy=1, cx=5)
        assert skip is False  # not confirmed yet

        # Mark confirmed
        sg.mark_confirmed("grass", cy=1, cx=5, is_traversable=True)
        skip, confirmed_node = sg.should_skip_asking("grass", cy=1, cx=5)
        # gp_variance is 0.0 (< SKIP_VARIANCE_THRESHOLD=0.04) and user_confirmed=True
        assert skip is True
        assert confirmed_node.user_confirmed is True
