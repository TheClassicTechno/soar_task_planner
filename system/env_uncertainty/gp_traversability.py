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

    def reset(self) -> None:
        """Clear all observations, returning to the uninformative prior."""
        self._X = []
        self._y = []

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


# ── WorldGPTraversabilityMap ──────────────────────────────────────────────────


class WorldGPTraversabilityMap:
    """
    GP traversability map in metric world coordinates (x_world, y_world metres).

    Unlike GPTraversabilityMap, observations are stored in world coordinates so
    they accumulate correctly across frames as the robot moves. The RBF kernel
    length scale is in metres (default 1.0 m) rather than normalized image units.

    This enables true multi-frame GP: observations from frame 1 at world position
    (2.0, 0.5) are still valid in frame 10 when the robot is at (5.0, 0.0) and
    looking back at the same terrain patch.

    Usage:
        # With MockForwardOdometry (dataset evaluation)
        odometry = MockForwardOdometry(speed_mps=0.5, fps=5.0)
        world_gp = WorldGPTraversabilityMap()

        for frame in frames:
            pose = odometry.next_pose()
            for region in detection.known_regions:
                for (x_w, y_w) in mask_centroids_to_world(region.mask, pose, ...):
                    world_gp.add_world_observation(x_w, y_w, region.traversability)

    Args:
        length_scale_m: RBF kernel length scale in metres (default 1.0 m).
                        Controls how quickly traversability estimates decay
                        with distance. 1.0 m is appropriate for outdoor ground robots.
        noise_level:    WhiteKernel noise for observation noise (default 0.01).
        max_observations: Cap on stored observations (oldest are dropped when
                          exceeded) to bound GP fit time. Default 500.
    """

    MU_PRIOR: float = 0.5
    SIGMA_PRIOR: float = 0.4
    DEFAULT_BETA: float = 1.5

    def __init__(
        self,
        length_scale_m: float = 1.0,
        noise_level: float = 0.01,
        max_observations: int = 500,
    ) -> None:
        kernel = RBF(length_scale=length_scale_m) + WhiteKernel(noise_level=noise_level)
        self._gpr = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=0,
            normalize_y=True,
        )
        self._X: List[List[float]] = []   # [[x_w, y_w], ...]  in metres
        self._y: List[float] = []
        self._max_obs = max_observations

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all world-coordinate observations."""
        self._X = []
        self._y = []

    def add_world_observation(
        self,
        x_w: float,
        y_w: float,
        traversability: float,
    ) -> None:
        """
        Record a terrain observation at world position (x_w, y_w).

        Maintains a rolling window of max_observations to keep GP inference fast.

        Args:
            x_w:            World x coordinate in metres.
            y_w:            World y coordinate in metres.
            traversability: Score ∈ [0, 1].
        """
        if len(self._X) >= self._max_obs:
            self._X.pop(0)
            self._y.pop(0)
        self._X.append([x_w, y_w])
        self._y.append(float(np.clip(traversability, 0.0, 1.0)))
        self._refit()

    def predict_world(
        self,
        x_w: float,
        y_w: float,
        beta: float = DEFAULT_BETA,
    ) -> GPPrediction:
        """
        Posterior traversability estimate at world position (x_w, y_w).

        Args:
            x_w:  World x in metres.
            y_w:  World y in metres.
            beta: LCB exploration parameter.

        Returns:
            GPPrediction with mu, sigma, lcb, beta, source.
        """
        if not self._X:
            lcb = self.MU_PRIOR - beta * self.SIGMA_PRIOR
            return GPPrediction(
                mu=self.MU_PRIOR, sigma=self.SIGMA_PRIOR,
                lcb=lcb, beta=beta, source="prior",
            )

        mu_arr, sigma_arr = self._gpr.predict([[x_w, y_w]], return_std=True)
        mu = float(np.clip(mu_arr[0], 0.0, 1.0))
        sigma = float(max(0.0, sigma_arr[0]))
        lcb = mu - beta * sigma
        return GPPrediction(mu=mu, sigma=sigma, lcb=lcb, beta=beta, source="posterior")

    def score_trajectory_world_lcb(
        self,
        world_waypoints: List[Tuple[float, float]],
        beta: float = DEFAULT_BETA,
    ) -> float:
        """
        Min LCB over world-coordinate trajectory waypoints.

        Args:
            world_waypoints: List of (x_w, y_w) in metres.
            beta:            LCB exploration parameter.

        Returns:
            Minimum LCB across all waypoints.
        """
        if not world_waypoints:
            return self.MU_PRIOR - beta * self.SIGMA_PRIOR
        scores = [self.predict_world(x, y, beta).lcb for x, y in world_waypoints]
        return float(min(scores))

    def apply_world_feedback(
        self,
        x_w: float,
        y_w: float,
        is_traversable: bool,
        p_tp: float = 0.95,
        p_fp: float = 0.10,
    ) -> None:
        """
        Bayesian GP update at world position from a user traversability response.

        Reads the current GP posterior mean at (x_w, y_w) as the prior, applies
        _bayesian_update() to get the posterior, then records it as a new observation.

        Args:
            x_w, y_w:       World position in metres.
            is_traversable: True if user says the region is safe.
            p_tp:           P(response="safe" | traversable=True).
            p_fp:           P(response="safe" | traversable=False).
        """
        prior = self.predict_world(x_w, y_w).mu
        posterior = _bayesian_update(prior, is_traversable, p_tp, p_fp)
        self.add_world_observation(x_w, y_w, posterior)

    @property
    def n_observations(self) -> int:
        """Number of world-coordinate terrain observations recorded."""
        return len(self._X)

    @property
    def observation_bounds(self) -> Optional[Tuple[float, float, float, float]]:
        """
        Bounding box of all observations: (min_x, max_x, min_y, max_y) in metres.
        Returns None when no observations exist.
        """
        if not self._X:
            return None
        arr = np.array(self._X)
        return float(arr[:, 0].min()), float(arr[:, 0].max()), \
               float(arr[:, 1].min()), float(arr[:, 1].max())

    # ── Private ───────────────────────────────────────────────────────────────

    def _refit(self) -> None:
        X = np.array(self._X)
        y = np.array(self._y)
        self._gpr.fit(X, y)
