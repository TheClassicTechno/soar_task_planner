"""
Tests for GPTraversabilityMap.

Covers:
  - Cold start (no observations): returns prior mu/sigma/source
  - Cold start lcb == MU_PRIOR - beta * SIGMA_PRIOR
  - After 1 safe observation: predict() mu increases near observation
  - After 1 unsafe observation: predict() mu decreases near observation
  - score_trajectory_lcb with empty waypoints
  - score_trajectory_lcb is min over waypoints
  - score_trajectory_lcb > STOP_THRESHOLD for all-safe trajectory
  - apply_user_feedback: posterior uses _bayesian_update math
  - apply_user_feedback: safe response raises mu; unsafe lowers it
  - n_observations increments correctly
  - normalize: (0,0) → (0,0); (H-1,W-1) → (1,1)
  - predict source: "prior" before fit, "posterior" after
"""

import pytest
import numpy as np

from system.env_uncertainty.gp_traversability import GPPrediction, GPTraversabilityMap
from system.env_uncertainty.traversability import STOP_THRESHOLD, _bayesian_update


H, W = 100, 100  # image dimensions used throughout


# ── Cold start ────────────────────────────────────────────────────────────────

def test_cold_start_returns_prior_mu():
    gp = GPTraversabilityMap()
    pred = gp.predict(50, 50, H, W)
    assert pred.mu == GPTraversabilityMap.MU_PRIOR


def test_cold_start_returns_prior_sigma():
    gp = GPTraversabilityMap()
    pred = gp.predict(50, 50, H, W)
    assert pred.sigma == GPTraversabilityMap.SIGMA_PRIOR


def test_cold_start_source_is_prior():
    gp = GPTraversabilityMap()
    pred = gp.predict(10, 10, H, W)
    assert pred.source == "prior"


def test_cold_start_lcb():
    gp = GPTraversabilityMap()
    beta = GPTraversabilityMap.DEFAULT_BETA
    pred = gp.predict(50, 50, H, W, beta=beta)
    expected_lcb = GPTraversabilityMap.MU_PRIOR - beta * GPTraversabilityMap.SIGMA_PRIOR
    assert abs(pred.lcb - expected_lcb) < 1e-9


def test_cold_start_beta_stored():
    gp = GPTraversabilityMap()
    pred = gp.predict(50, 50, H, W, beta=2.0)
    assert pred.beta == 2.0


# ── After observations ────────────────────────────────────────────────────────

def test_posterior_source_after_observation():
    gp = GPTraversabilityMap()
    gp.add_observation(50, 50, 0.90, H, W)
    pred = gp.predict(50, 50, H, W)
    assert pred.source == "posterior"


def test_safe_observation_raises_nearby_mu():
    gp = GPTraversabilityMap()
    mu_before = gp.predict(50, 50, H, W).mu
    gp.add_observation(50, 50, 0.95, H, W)
    mu_after = gp.predict(50, 50, H, W).mu
    assert mu_after > mu_before


def test_unsafe_observation_lowers_nearby_mu():
    gp = GPTraversabilityMap()
    gp.add_observation(50, 50, 0.05, H, W)
    pred = gp.predict(50, 50, H, W)
    assert pred.mu < GPTraversabilityMap.MU_PRIOR


def test_observation_at_exact_location_close_to_value():
    gp = GPTraversabilityMap()
    gp.add_observation(50, 50, 0.85, H, W)
    pred = gp.predict(50, 50, H, W)
    assert abs(pred.mu - 0.85) < 0.15


def test_multiple_observations_increase_certainty():
    """Multiple observations at a point should reduce sigma."""
    gp = GPTraversabilityMap()
    gp.add_observation(50, 50, 0.80, H, W)
    sigma_1 = gp.predict(50, 50, H, W).sigma
    gp.add_observation(50, 51, 0.80, H, W)
    gp.add_observation(51, 50, 0.80, H, W)
    sigma_3 = gp.predict(50, 50, H, W).sigma
    assert sigma_3 <= sigma_1


def test_n_observations_increments():
    gp = GPTraversabilityMap()
    assert gp.n_observations == 0
    gp.add_observation(10, 10, 0.80, H, W)
    assert gp.n_observations == 1
    gp.add_observation(20, 20, 0.60, H, W)
    assert gp.n_observations == 2


# ── score_trajectory_lcb ─────────────────────────────────────────────────────

def test_score_trajectory_lcb_empty_returns_prior_lcb():
    gp = GPTraversabilityMap()
    beta = GPTraversabilityMap.DEFAULT_BETA
    score = gp.score_trajectory_lcb([], H, W)
    expected = GPTraversabilityMap.MU_PRIOR - beta * GPTraversabilityMap.SIGMA_PRIOR
    assert abs(score - expected) < 1e-9


def test_score_trajectory_lcb_is_minimum():
    gp = GPTraversabilityMap()
    gp.add_observation(10, 10, 0.90, H, W)
    gp.add_observation(90, 90, 0.05, H, W)
    lcb_safe = gp.predict(10, 10, H, W).lcb
    lcb_unsafe = gp.predict(90, 90, H, W).lcb
    traj_score = gp.score_trajectory_lcb([(10, 10), (90, 90)], H, W)
    assert abs(traj_score - min(lcb_safe, lcb_unsafe)) < 1e-9


def test_score_trajectory_lcb_safe_path_above_stop_threshold():
    gp = GPTraversabilityMap()
    for r in range(20, 80, 10):
        gp.add_observation(r, 50, 0.90, H, W)
    waypoints = [(r, 50) for r in range(20, 80, 10)]
    score = gp.score_trajectory_lcb(waypoints, H, W)
    assert score > STOP_THRESHOLD


def test_score_trajectory_lcb_uses_custom_beta():
    gp = GPTraversabilityMap()
    gp.add_observation(50, 50, 0.80, H, W)
    score_low_beta = gp.score_trajectory_lcb([(50, 50)], H, W, beta=0.0)
    score_high_beta = gp.score_trajectory_lcb([(50, 50)], H, W, beta=5.0)
    assert score_low_beta > score_high_beta


# ── apply_user_feedback ───────────────────────────────────────────────────────

def test_feedback_safe_raises_mu():
    gp = GPTraversabilityMap()
    mu_before = gp.predict(50, 50, H, W).mu   # prior
    gp.apply_user_feedback(50, 50, True, H, W)
    mu_after = gp.predict(50, 50, H, W).mu
    assert mu_after > mu_before


def test_feedback_unsafe_lowers_mu():
    gp = GPTraversabilityMap()
    gp.apply_user_feedback(50, 50, False, H, W)
    mu_after = gp.predict(50, 50, H, W).mu
    assert mu_after < GPTraversabilityMap.MU_PRIOR


def test_feedback_uses_bayesian_update_math():
    """Verify apply_user_feedback matches _bayesian_update() from traversability.py."""
    gp = GPTraversabilityMap()
    prior = gp.predict(50, 50, H, W).mu
    expected_posterior = _bayesian_update(prior, True, p_tp=0.95, p_fp=0.10)
    gp.apply_user_feedback(50, 50, True, H, W)
    mu_after = gp.predict(50, 50, H, W).mu
    assert abs(mu_after - expected_posterior) < 0.15


def test_feedback_adds_observation():
    gp = GPTraversabilityMap()
    assert gp.n_observations == 0
    gp.apply_user_feedback(50, 50, True, H, W)
    assert gp.n_observations == 1


def test_feedback_sequential_refinement():
    """Two consecutive safe responses should push mu higher than one."""
    gp1 = GPTraversabilityMap()
    gp1.apply_user_feedback(50, 50, True, H, W)
    mu_1 = gp1.predict(50, 50, H, W).mu

    gp2 = GPTraversabilityMap()
    gp2.apply_user_feedback(50, 50, True, H, W)
    gp2.apply_user_feedback(50, 50, True, H, W)
    mu_2 = gp2.predict(50, 50, H, W).mu

    assert mu_2 >= mu_1


# ── Normalize helper ──────────────────────────────────────────────────────────

def test_normalize_origin():
    y_n, x_n = GPTraversabilityMap._normalize(0, 0, H, W)
    assert y_n == 0.0 and x_n == 0.0


def test_normalize_max_corner():
    y_n, x_n = GPTraversabilityMap._normalize(H - 1, W - 1, H, W)
    assert abs(y_n - 1.0) < 1e-9 and abs(x_n - 1.0) < 1e-9


def test_normalize_center():
    y_n, x_n = GPTraversabilityMap._normalize(49, 49, H, W)
    assert abs(y_n - 49 / 99) < 1e-9
    assert abs(x_n - 49 / 99) < 1e-9
