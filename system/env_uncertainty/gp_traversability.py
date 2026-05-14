"""
Gaussian Process traversability map for outdoor terrain.

Maintains a GP posterior over traversability in normalized pixel space [0,1]².
Supports three use cases:

  1. add_observation()       — record a known terrain score at a pixel location
  2. predict()               — posterior mean + std + LCB at any pixel
  3. score_trajectory_lcb()  — LCB-safety score for a full waypoint sequence
  4. apply_user_feedback()   — Bayesian update from a user's safe/unsafe response

Non-AI novelty
--------------
Replaces FM-based terrain scoring (rejected by mentor) with a deterministic GP
whose kernel is fixed (no optimizer restarts), making it fast and reproducible.
Posterior mean provides the traversability estimate; σ quantifies how uncertain
the estimate is at each location.

Lower Confidence Bound (LCB) trajectory scoring
------------------------------------------------
  LCB(τ) = min_t [μ_post(p_t) − β · σ_post(p_t)]

A low LCB signals either low expected traversability OR high uncertainty — both
are reasons to ask the user before proceeding.

User feedback integration
-------------------------
Uses _bayesian_update() from traversability.py (identical likelihood model):
  τ₁ = p_tp · τ₀ / (p_tp · τ₀ + p_fp · (1 − τ₀))   if user says "safe"
  τ₁ = (1−p_tp) · τ₀ / (…)                            if user says "unsafe"
The Bayesian posterior τ₁ is then added as a new GP observation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from system.env_uncertainty.traversability import STOP_THRESHOLD, _bayesian_update


@dataclass
class GPPrediction:
    """
    Output from GPTraversabilityMap.predict().

    mu:     Posterior mean traversability ∈ [0, 1].
    sigma:  Posterior std dev ∈ [0, ∞).
    lcb:    mu − beta * sigma  (lower confidence bound).
    beta:   Exploration parameter used to compute lcb.
    source: "prior" (no observations yet) or "posterior".
    """

    mu: float
    sigma: float
    lcb: float
    beta: float
    source: str


class GPTraversabilityMap:
    """
    Gaussian Process traversability model over an image plane.

    Observations are stored as normalized (y, x) coordinates in [0, 1]²
    so that the GP length scale is image-size agnostic.

    Kernel is fixed — no optimizer restarts — for deterministic, fast inference.

    Class constants
    ---------------
    MU_PRIOR:     Prior mean traversability (uniform prior before any data).
    SIGMA_PRIOR:  Prior std dev returned when no observations exist.
    DEFAULT_BETA: LCB exploration parameter β (higher → more conservative).

    Args:
        length_scale: RBF kernel length scale in normalized coords (default 0.15).
        noise_level:  WhiteKernel noise for observation noise (default 0.01).
    """

    MU_PRIOR: float = 0.5
    SIGMA_PRIOR: float = 0.4
    DEFAULT_BETA: float = 1.5

    def __init__(
        self,
        length_scale: float = 0.15,
        noise_level: float = 0.01,
    ) -> None:
        kernel = RBF(length_scale=length_scale) + WhiteKernel(noise_level=noise_level)
        self._gpr = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=0,
            normalize_y=True,
        )
        self._X: List[List[float]] = []   # [[y_n, x_n], ...]
        self._y: List[float] = []         # traversability values

    # ── Public API ────────────────────────────────────────────────────────────

    def add_observation(
        self,
        pixel_y: int,
        pixel_x: int,
        traversability: float,
        height: int,
        width: int,
    ) -> None:
        """
        Record a terrain observation at (pixel_y, pixel_x).

        Args:
            pixel_y:       Row index (0-indexed).
            pixel_x:       Column index (0-indexed).
            traversability: Score ∈ [0, 1].
            height:        Image height in pixels.
            width:         Image width in pixels.
        """
        y_n, x_n = self._normalize(pixel_y, pixel_x, height, width)
        self._X.append([y_n, x_n])
        self._y.append(float(np.clip(traversability, 0.0, 1.0)))
        self._refit()

    def predict(
        self,
        pixel_y: int,
        pixel_x: int,
        height: int,
        width: int,
        beta: float = DEFAULT_BETA,
    ) -> GPPrediction:
        """
        Return posterior (or prior) traversability estimate at (pixel_y, pixel_x).

        Args:
            pixel_y: Row index.
            pixel_x: Column index.
            height:  Image height.
            width:   Image width.
            beta:    LCB exploration parameter.

        Returns:
            GPPrediction with mu, sigma, lcb, beta, source.
        """
        if not self._X:
            lcb = self.MU_PRIOR - beta * self.SIGMA_PRIOR
            return GPPrediction(
                mu=self.MU_PRIOR,
                sigma=self.SIGMA_PRIOR,
                lcb=lcb,
                beta=beta,
                source="prior",
            )

        y_n, x_n = self._normalize(pixel_y, pixel_x, height, width)
        mu_arr, sigma_arr = self._gpr.predict([[y_n, x_n]], return_std=True)
        mu = float(np.clip(mu_arr[0], 0.0, 1.0))
        sigma = float(max(0.0, sigma_arr[0]))
        lcb = mu - beta * sigma
        return GPPrediction(mu=mu, sigma=sigma, lcb=lcb, beta=beta, source="posterior")

    def score_trajectory_lcb(
        self,
        waypoints: List[Tuple[int, int]],
        height: int,
        width: int,
        beta: float = DEFAULT_BETA,
    ) -> float:
        """
        Return the minimum LCB over all waypoints (safety-first trajectory score).

        A low value means at least one point on the path is either low-traversability
        or high-uncertainty — both warrant asking the user before proceeding.

        Args:
            waypoints: List of (y, x) pixel coordinate tuples.
            height:    Image height.
            width:     Image width.
            beta:      LCB exploration parameter.

        Returns:
            min_t[μ(p_t) − β·σ(p_t)].  Returns MU_PRIOR − β·SIGMA_PRIOR if empty.
        """
        if not waypoints:
            return self.MU_PRIOR - beta * self.SIGMA_PRIOR

        scores = [
            self.predict(y, x, height, width, beta).lcb
            for y, x in waypoints
        ]
        return float(min(scores))

    def apply_user_feedback(
        self,
        pixel_y: int,
        pixel_x: int,
        is_traversable: bool,
        height: int,
        width: int,
        p_tp: float = 0.95,
        p_fp: float = 0.10,
    ) -> None:
        """
        Update the GP with a user traversability response via Bayesian posterior.

        Reads the current GP posterior mean at (pixel_y, pixel_x) as the prior
        τ₀, applies _bayesian_update() to get τ₁, then records τ₁ as a new
        observation.  Calling this multiple times performs sequential refinement.

        Args:
            pixel_y:        Row index.
            pixel_x:        Column index.
            is_traversable: True if user says the region is safe.
            height:         Image height.
            width:          Image width.
            p_tp:           P(response="safe" | traversable=True).
            p_fp:           P(response="safe" | traversable=False).
        """
        prior = self.predict(pixel_y, pixel_x, height, width).mu
        posterior = _bayesian_update(prior, is_traversable, p_tp, p_fp)
        self.add_observation(pixel_y, pixel_x, posterior, height, width)

    @property
    def n_observations(self) -> int:
        """Number of terrain observations recorded."""
        return len(self._X)

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _normalize(pixel_y: int, pixel_x: int, height: int, width: int) -> Tuple[float, float]:
        """Map pixel coordinates to [0, 1]²."""
        y_n = float(pixel_y) / max(height - 1, 1)
        x_n = float(pixel_x) / max(width - 1, 1)
        return y_n, x_n

    def _refit(self) -> None:
        """Refit the GP on all stored observations."""
        X = np.array(self._X)
        y = np.array(self._y)
        self._gpr.fit(X, y)
