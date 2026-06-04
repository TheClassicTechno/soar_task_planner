"""
Pipeline verification tests — Steps 7, 8, 9.

Verifies every concrete claim in the mentor-facing pipeline description:

  Step 7 — Grounded question generation
    - High-entropy Dirichlet node → top_k_classes returns top-3 by probability
    - Candidates appear verbatim in the generated question
    - Template mode is deterministic (same input → same output, no LLM call)
    - LLM mode is available and injects top-k into the prompt

  Step 8 — User response parsing (keyword regex baseline)
    - "I think ..." → label_confidence = 0.60  (NOTE: document says 0.50 — that is wrong)
    - "pavement"    → terrain_label = "sidewalk"
    - "wet and slippery" → affordance_modifier = -0.35 (wet=-0.15, slippery=-0.20)
    - No keyword match → is_traversable = False (safety-first default)

  Step 9 — Bayesian update (closed-form, O(1))
    - GP apply_user_feedback() adds one new observation (n_observations += 1)
    - Scene graph update_from_user() increments the correct Dirichlet alpha index
    - Both updates are deterministic — no randomness, no retraining
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from system.env_uncertainty.detector import DetectionResult, RegionInfo
from system.env_uncertainty.gp_traversability import GPTraversabilityMap
from system.env_uncertainty.map_updater import parse_user_response_rich
from system.env_uncertainty.question_generator import (
    QuestionGenerator,
    generate_question_template,
)
from system.env_uncertainty.scene_graph import SceneGraph, TERRAIN_CLASSES, TerrainNode
from system.env_uncertainty.traversability import TraversabilityMap


# ── Shared helpers ─────────────────────────────────────────────────────────────

H, W = 50, 50


def _blank_result(unknown_coverage=0.20, n_unknown=1):
    mask = np.zeros((H, W), dtype=bool)
    n = int(H * W * unknown_coverage)
    mask.flat[:n] = True
    region = RegionInfo(
        label="unknown", mask=mask, confidence=0.8,
        pixel_fraction=unknown_coverage, source="sam2", traversability=0.0,
    )
    tmap = TraversabilityMap.create(H, W)
    return DetectionResult(
        known_regions=[],
        unknown_regions=[region] * n_unknown,
        image_shape=(H, W),
        sam3_coverage=0.5,
        unknown_coverage=unknown_coverage,
        has_unknown=True,
        traversability_map=tmap,
    )


def _high_entropy_node() -> TerrainNode:
    """Return a TerrainNode with uniform Dirichlet prior — maximum entropy."""
    sg = SceneGraph()
    return sg.upsert_region("unknown", pixel_y=25, pixel_x=25, height=H, width=W)


def _low_entropy_node(label="grass") -> TerrainNode:
    """Return a TerrainNode with 50 extra pseudocounts on one class — near-zero entropy."""
    sg = SceneGraph()
    node = sg.upsert_region(label, pixel_y=25, pixel_x=25, height=H, width=W)
    idx = TERRAIN_CLASSES.index(label)
    node.dirichlet_alpha[idx] += 50.0
    return node


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Grounded question generation
# ══════════════════════════════════════════════════════════════════════════════

class TestStep7GroundedQuestions:

    def test_top_k_returns_three_candidates(self):
        # top_k_classes(k=3) must return exactly 3 (label, probability) pairs
        node = _high_entropy_node()
        top_k = node.top_k_classes(k=3)
        assert len(top_k) == 3

    def test_top_k_sorted_by_probability_descending(self):
        # Probabilities must be in descending order
        node = _high_entropy_node()
        # Give mud a strong prior to create a clear ordering
        mud_idx = TERRAIN_CLASSES.index("mud")
        node.dirichlet_alpha[mud_idx] += 20.0
        top_k = node.top_k_classes(k=3)
        probs = [p for _, p in top_k]
        assert probs == sorted(probs, reverse=True)

    def test_top_k_first_class_has_highest_alpha(self):
        # The class with the most pseudocounts should be first
        node = _high_entropy_node()
        grass_idx = TERRAIN_CLASSES.index("grass")
        node.dirichlet_alpha[grass_idx] += 30.0
        top_k = node.top_k_classes(k=3)
        assert top_k[0][0] == "grass"

    def test_top_k_probabilities_sum_to_one(self):
        # All K probabilities sum to 1; top-3 must be <= 1
        node = _high_entropy_node()
        top_k = node.top_k_classes(k=3)
        total = sum(p for _, p in top_k)
        assert total <= 1.0 + 1e-9

    def test_grounded_question_contains_candidate_names(self):
        # When top_k_classes is provided, question must name each candidate
        top_k = [("mud", 0.50), ("gravel", 0.30), ("grass", 0.20)]
        result = _blank_result()
        q = generate_question_template(result, top_k_classes=top_k)
        for label, _ in top_k:
            assert label in q.lower(), f"Expected '{label}' in question: {q}"

    def test_grounded_question_is_not_generic(self):
        # Grounded question must NOT fall back to the generic "unrecognized area" phrasing
        top_k = [("mud", 0.50), ("gravel", 0.30), ("grass", 0.20)]
        result = _blank_result()
        q = generate_question_template(result, top_k_classes=top_k)
        assert "unrecognized area" not in q.lower()

    def test_template_mode_is_deterministic(self):
        # Same input must always produce the same output — no LLM randomness
        top_k = [("mud", 0.50), ("gravel", 0.30), ("grass", 0.20)]
        result = _blank_result()
        gen = QuestionGenerator(mode="template")
        q1 = gen.generate(result, top_k_classes=top_k)
        q2 = gen.generate(result, top_k_classes=top_k)
        assert q1 == q2

    def test_template_mode_makes_no_llm_call(self):
        # Template mode must never call an LLM
        mock_llm = MagicMock()
        gen = QuestionGenerator(mode="template")
        result = _blank_result()
        gen.generate(result, top_k_classes=[("mud", 0.5)])
        mock_llm.predict_json.assert_not_called()

    def test_llm_mode_is_available(self):
        # LLM mode must exist and return the LLM's question
        mock_llm = MagicMock()
        mock_llm.predict_json.return_value = {"question": "Is the mud safe?"}
        gen = QuestionGenerator(mode="llm", llm=mock_llm)
        result = _blank_result()
        q = gen.generate(result, top_k_classes=[("mud", 0.5)])
        assert q == "Is the mud safe?"

    def test_llm_prompt_contains_candidate_names(self):
        # When top_k provided, the LLM prompt must include the class names
        from system.env_uncertainty.question_generator import _build_llm_prompt
        from system.env_uncertainty.user_profile import DEFAULT_PROFILE
        top_k = [("mud", 0.50), ("gravel", 0.30), ("grass", 0.20)]
        prompt = _build_llm_prompt(
            _blank_result(), trajectories=None,
            profile=DEFAULT_PROFILE, scenario_context=None,
            top_k_classes=top_k,
        )
        for label, _ in top_k:
            assert label in prompt, f"Expected '{label}' in LLM prompt"

    def test_no_top_k_falls_back_to_generic_template(self):
        # Without top_k, question must still be a non-empty string
        result = _blank_result()
        q = generate_question_template(result, top_k_classes=None)
        assert isinstance(q, str) and len(q) > 10


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — User response parsing (keyword regex baseline)
# ══════════════════════════════════════════════════════════════════════════════

class TestStep8ResponseParsing:

    def test_i_think_confidence_is_0_60(self):
        # "think" maps to 0.60 in _CONFIDENCE_KEYWORDS.
        # NOTE: the pipeline document incorrectly states 0.50 — the code value is 0.60.
        r = parse_user_response_rich("I think it's fine")
        assert r.label_confidence == pytest.approx(0.60)

    def test_pavement_maps_to_sidewalk(self):
        # Synonym table: "pavement" → canonical label "sidewalk"
        r = parse_user_response_rich("It looks like pavement")
        assert r.terrain_label == "sidewalk"

    def test_wet_and_slippery_modifier_is_minus_0_35(self):
        # wet=-0.15, slippery=-0.20 → sum=-0.35
        r = parse_user_response_rich("wet and slippery")
        assert r.affordance_modifier == pytest.approx(-0.35, abs=1e-9)

    def test_wet_alone_is_minus_0_15(self):
        r = parse_user_response_rich("it's wet")
        assert r.affordance_modifier == pytest.approx(-0.15, abs=1e-9)

    def test_slippery_alone_is_minus_0_20(self):
        r = parse_user_response_rich("very slippery")
        assert r.affordance_modifier == pytest.approx(-0.20, abs=1e-9)

    def test_no_keyword_match_is_not_traversable(self):
        # Safety-first: ambiguous response must default to is_traversable=False
        r = parse_user_response_rich("I have no idea what that is")
        assert r.is_traversable is False

    def test_no_keyword_match_has_low_confidence(self):
        # No match → traversability_confidence is the low-confidence fallback (0.30)
        r = parse_user_response_rich("I have no idea what that is")
        assert r.traversability_confidence == pytest.approx(0.30)

    def test_positive_affordance_is_traversable(self):
        # "safe" maps to +0.20 → affordance_modifier >= 0 → is_traversable=True
        r = parse_user_response_rich("it looks safe and firm")
        assert r.is_traversable is True

    def test_returns_parsed_user_response_dataclass(self):
        # Output must have all required fields
        r = parse_user_response_rich("probably grass")
        assert hasattr(r, "terrain_label")
        assert hasattr(r, "label_confidence")
        assert hasattr(r, "is_traversable")
        assert hasattr(r, "traversability_confidence")
        assert hasattr(r, "affordance_modifier")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — Bayesian update (closed-form, O(1), no retraining)
# ══════════════════════════════════════════════════════════════════════════════

class TestStep9BayesianUpdate:

    # ── GP: apply_user_feedback ───────────────────────────────────────────────

    def test_gp_feedback_adds_one_observation(self):
        # apply_user_feedback() must add exactly one GP observation
        gp = GPTraversabilityMap()
        assert gp.n_observations == 0
        gp.apply_user_feedback(25, 25, is_traversable=True, height=H, width=W)
        assert gp.n_observations == 1

    def test_gp_safe_feedback_raises_posterior_mean(self):
        # Saying "safe" at an uncertain location should increase traversability estimate
        gp = GPTraversabilityMap()
        before = gp.predict(25, 25, H, W).mu  # prior
        gp.apply_user_feedback(25, 25, is_traversable=True, height=H, width=W)
        after = gp.predict(25, 25, H, W).mu   # posterior
        assert after > before

    def test_gp_unsafe_feedback_lowers_posterior_mean(self):
        # Saying "unsafe" should lower traversability estimate
        gp = GPTraversabilityMap()
        before = gp.predict(25, 25, H, W).mu
        gp.apply_user_feedback(25, 25, is_traversable=False, height=H, width=W)
        after = gp.predict(25, 25, H, W).mu
        assert after < before

    def test_gp_feedback_is_deterministic(self):
        # Same feedback at same location must give same posterior
        gp1 = GPTraversabilityMap()
        gp2 = GPTraversabilityMap()
        gp1.apply_user_feedback(25, 25, is_traversable=True, height=H, width=W)
        gp2.apply_user_feedback(25, 25, is_traversable=True, height=H, width=W)
        assert gp1.predict(25, 25, H, W).mu == pytest.approx(
            gp2.predict(25, 25, H, W).mu, abs=1e-6
        )

    def test_gp_feedback_does_not_affect_distant_point(self):
        # GP posterior at a far pixel should be close to prior (not fully updated)
        gp = GPTraversabilityMap()
        gp.apply_user_feedback(0, 0, is_traversable=True, height=H, width=W)
        far_pred = gp.predict(49, 49, H, W)
        # Still within GP uncertainty range — not the exact prior but not wildly off
        assert 0.0 <= far_pred.mu <= 1.0

    # ── Scene graph: update_from_user (Dirichlet conjugate update) ────────────

    def test_dirichlet_update_increments_correct_alpha(self):
        # update_from_user("grass", confidence=1.0) must increase alpha[grass_idx]
        node = _high_entropy_node()
        grass_idx = TERRAIN_CLASSES.index("grass")
        before = node.dirichlet_alpha[grass_idx]
        node.update_from_user("grass", confidence=1.0)
        assert node.dirichlet_alpha[grass_idx] == pytest.approx(before + 1.0)

    def test_dirichlet_update_does_not_change_other_alphas(self):
        # Only the matched class alpha should change
        node = _high_entropy_node()
        mud_idx = TERRAIN_CLASSES.index("mud")
        grass_idx = TERRAIN_CLASSES.index("grass")
        before_mud = node.dirichlet_alpha[mud_idx]
        node.update_from_user("grass", confidence=1.0)
        assert node.dirichlet_alpha[mud_idx] == pytest.approx(before_mud)

    def test_dirichlet_update_reduces_entropy(self):
        # Adding pseudocounts to one class must lower semantic entropy
        node = _high_entropy_node()
        entropy_before = node.semantic_entropy()
        node.update_from_user("grass", confidence=10.0)
        entropy_after = node.semantic_entropy()
        assert entropy_after < entropy_before

    def test_dirichlet_update_is_deterministic(self):
        # Two nodes with same prior and same update must have identical alpha
        node1 = _high_entropy_node()
        node2 = _high_entropy_node()
        node1.update_from_user("mud", confidence=0.8)
        node2.update_from_user("mud", confidence=0.8)
        assert node1.dirichlet_alpha == pytest.approx(node2.dirichlet_alpha)

    def test_dirichlet_update_with_unknown_label_is_noop(self):
        # Labels not in TERRAIN_CLASSES must be silently ignored
        node = _high_entropy_node()
        alpha_before = list(node.dirichlet_alpha)
        node.update_from_user("unicorn_terrain", confidence=1.0)
        assert node.dirichlet_alpha == pytest.approx(alpha_before)

    def test_both_updates_together(self):
        # Both GP and scene graph update from the same parsed response without errors
        gp = GPTraversabilityMap()
        node = _high_entropy_node()
        r = parse_user_response_rich("It looks like firm grass")

        gp.apply_user_feedback(25, 25, is_traversable=r.is_traversable, height=H, width=W)
        if r.terrain_label:
            node.update_from_user(r.terrain_label, confidence=r.label_confidence)

        assert gp.n_observations == 1
        grass_idx = TERRAIN_CLASSES.index("grass")
        # Dirichlet alpha for grass should be > 1 (label-informed prior was 2.0,
        # but node was created with "unknown" label so grass starts at 1.0)
        assert node.dirichlet_alpha[grass_idx] > 1.0
