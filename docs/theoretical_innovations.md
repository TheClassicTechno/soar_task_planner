# Theoretical Innovations: Mathematical Formulations

## Overview

This document presents the three core mathematical innovations of the system:
(A) a joint conformal prediction framework that provides coverage guarantees over two simultaneous uncertainty types; (B) a Bayesian traversability update that incorporates noisy user feedback into the robot's terrain model; and (C) an instruction uncertainty scoring function with calibrated decision threshold. Together these form the theoretical backbone connecting the perception pipeline to statistically grounded ask-or-act decisions.

---

## A. Joint Conformal Prediction for Two-Type Uncertainty

### A.1 Background: Single-Branch Conformal Prediction

We first establish notation for the standard conformal prediction setup used by KnowNo [Ren et al., 2023].

**Setting.** Let $\mathcal{D}$ be a distribution over scenarios $\xi = (e, \ell, g)$, where $e$ is an environment (POMDP), $\ell$ is a user instruction (possibly null), and $g$ is the correct behavior label $y^* \in \{A, B\}$ (act directly / ask user). A calibration set $\mathcal{Z} = \{z_i = (\tilde{x}_i, y_i)\}_{i=1}^N$ of $N$ i.i.d. samples is drawn from $\mathcal{D}$, where $\tilde{x}_i$ is the augmented context (instruction + scene description + LLM-generated candidate options) and $y_i$ is the ground-truth label.

**Non-conformity score.** The LLM produces normalized option confidences $\hat{f}(\tilde{x})_y \in [0,1]$ with $\sum_y \hat{f}(\tilde{x})_y = 1$. The non-conformity score for the $i$-th calibration example is:

$$\kappa_i = 1 - \hat{f}(\tilde{x}_i)_{y_i} \in [0, 1]$$

A high score indicates the LLM is not confident in the correct answer — the scenario does not "conform" to what the model has learned.

**Calibration and prediction set.** Given a desired coverage level $1 - \varepsilon \in (0, 1)$, the calibration quantile is:

$$\hat{q} = \text{Quantile}\!\left(\{\kappa_i\}_{i=1}^N;\; \frac{\lceil (N+1)(1-\varepsilon) \rceil}{N}\right)$$

At test time, the prediction set for a new scenario $\tilde{x}_{\text{test}}$ is:

$$C(\tilde{x}_{\text{test}}) = \left\{y \in \{A, B\} \;\middle|\; \hat{f}(\tilde{x}_{\text{test}})_y \geq 1 - \hat{q}\right\}$$

**Coverage guarantee** (KnowNo Proposition 1): With probability at least $1 - \delta$ over the sampling of $\mathcal{Z}$:

$$P\!\left(y_{\text{test}} \in C(\tilde{x}_{\text{test}})\right) \geq 1 - \varepsilon$$

**Ask decision.** The robot asks the user if and only if $|C(\tilde{x}_{\text{test}})| > 1$, i.e., both options survive the threshold.

---

### A.2 Novel Extension: Joint Non-Conformity Score

The single-branch formulation handles instruction uncertainty but provides no mechanism for environmental uncertainty to influence the ask decision. We extend CP to jointly reason over both branches.

**Environmental confidence function.** Define $g : \mathcal{X}^E \to [0, 1]$ as the traversability confidence of the best available trajectory, where $\mathcal{X}^E$ is the space of terrain observations. Concretely:

$$g(x^E) = \min_{\tau \in \mathcal{T}^*} \bar{\tau}(\text{path}_\tau)$$

where $\mathcal{T}^* = \{\tau : \tau \text{ does not pass through an unknown region}\}$ is the set of trajectories avoiding all unknown regions, and $\bar{\tau}(\text{path}_\tau)$ is the mean traversability score along trajectory $\tau$. If $\mathcal{T}^*$ is empty (all paths cross unknown terrain), we set $g(x^E) = 0$.

Thus $g(x^E) = 1$ when all candidate trajectories pass through fully known safe terrain (no need to ask), and $g(x^E) = 0$ when all paths are blocked by unknown or impassable terrain (definitely ask).

**Environmental non-conformity score.** By analogy with the instruction branch:

$$\kappa^E = 1 - g(x^E) \in [0, 1]$$

**Instruction non-conformity score.** For the instruction branch, $\kappa^I = 1 - \hat{f}(\tilde{x}^I)_{y^*}$ as before. When no instruction is present, $\kappa^I = 0$ (no instruction ambiguity).

**Joint non-conformity score.** We define the joint score as the element-wise maximum over both branches:

$$\kappa^{\text{joint}} = \max\!\left(\kappa^I,\; \kappa^E\right)$$

**Intuition.** The max operation is conservative: if either branch signals high non-conformity (high uncertainty), the joint score is high, which pushes the calibration quantile upward and expands the prediction set. This means the robot becomes more likely to ask whenever either its instruction understanding OR its terrain perception is uncertain. The robot asks when it needs to — not only when the instruction is ambiguous.

**Joint calibration.** A joint calibration set requires paired examples with both instruction and environmental context:

$$\mathcal{Z}^{\text{joint}} = \left\{z_i = \left(\tilde{x}_i^I, x_i^E, y_i\right)\right\}_{i=1}^N$$

where $y_i = B$ (ask) whenever $\kappa_i^I > \theta$ or $\kappa_i^E > \theta$ for some pre-specified threshold, and $y_i = A$ otherwise. The joint non-conformity score for each calibration example is $\kappa_i^{\text{joint}} = \max(\kappa_i^I, \kappa_i^E)$, and $\hat{q}$ is computed from $\{\kappa_i^{\text{joint}}\}_{i=1}^N$.

**Coverage guarantee (extended).** The standard CP coverage guarantee holds for the joint score under exchangeability of $\mathcal{Z}^{\text{joint}} \cup \{z_{\text{test}}\}$:

$$P\!\left(y_{\text{test}} \in C^{\text{joint}}(\tilde{x}_{\text{test}}^I, x_{\text{test}}^E)\right) \geq 1 - \varepsilon$$

**Proof sketch.** The joint prediction set $C^{\text{joint}}$ is formed by including option $y$ if $\hat{f}(\tilde{x}^I)_y \geq 1 - \hat{q}$ AND $g(x^E) \geq 1 - \hat{q}$. Since $\kappa^{\text{joint}} \geq \kappa^I$ and $\kappa^{\text{joint}} \geq \kappa^E$, the joint calibration quantile $\hat{q}^{\text{joint}}$ is at least as large as either single-branch quantile, making $C^{\text{joint}}$ at least as conservative as either single-branch set. Coverage follows from the standard CP argument applied to the joint scores. $\square$

**Backward compatibility.** When no instruction is present, $\kappa^I = 0$, so $\kappa^{\text{joint}} = \kappa^E$. The joint formulation reduces to pure environmental branch calibration, recovering the original environmental uncertainty decision rule without modification.

---

### A.3 Dataset-Conditional Guarantee (Practical Use)

Following KnowNo's Eq. (2), we apply the dataset-conditional guarantee that avoids re-calibration for new test data:

$$P\!\left(y_{\text{test}} \in C(\tilde{x}_{\text{test}}) \;\middle|\; \{z_1, \ldots, z_N\}\right) \geq \text{Beta}_{N+1-v,\, v}^{-1}(\delta)$$

where $v = \lfloor (N+1) \hat{\varepsilon} \rfloor$, and $\text{Beta}_{a,b}^{-1}(\delta)$ is the $\delta$-quantile of the Beta$(a, b)$ distribution. We use $N = 57$ (calibration set size), $\delta = 0.01$, and adjust $\hat{\varepsilon}$ to achieve the desired $1 - \varepsilon$ coverage with probability $1 - \delta = 0.99$.

---

## B. Bayesian Traversability Update After User Feedback

### B.1 Motivation

The current `MapUpdater.apply_user_feedback()` hard-codes updated traversability to 0.9 (safe) or 0.0 (unsafe). This is incorrect for two reasons. First, it ignores the robot's prior knowledge: a region labeled "gravel" that was unknown before feedback has a different prior than one that appeared to be mud. Second, it treats user responses as noiseless oracles, ignoring the possibility of user error or ambiguity in the response.

### B.2 Model Definition

Let $T_r \in \{0, 1\}$ be the binary traversability of a terrain region $r$ (1 = traversable, 0 = impassable). Before receiving user feedback, the robot holds a prior:

$$P(T_r = 1) = \tau_0^{(r)}$$

where $\tau_0^{(r)}$ is determined as follows:
- If the region is SAM3-labeled: $\tau_0^{(r)}$ = static table value for that class
- If the region is SAM2-only (unknown): $\tau_0^{(r)} = 0.5$ (maximum entropy prior — no information)
- If the region has been encountered before: $\tau_0^{(r)}$ = posterior from most recent interaction

The user's response $R \in \{\text{safe}, \text{unsafe}\}$ is modeled as a binary noisy channel with known error rates derived from user study data:

| True state | $P(R = \text{safe})$ | $P(R = \text{unsafe})$ |
|---|---|---|
| $T_r = 1$ (traversable) | $p_{\text{tp}} = 0.95$ | $1 - p_{\text{tp}} = 0.05$ |
| $T_r = 0$ (impassable) | $p_{\text{fp}} = 0.10$ | $1 - p_{\text{fp}} = 0.90$ |

These values reflect the observation that users rarely call safe terrain unsafe (false negative rate 5%), but may occasionally call unsafe terrain safe due to overconfidence or poor visibility (false positive rate 10%).

### B.3 Posterior Update Equations

**After $R = \text{safe}$:**

$$P(T_r = 1 \mid R = \text{safe}) = \frac{p_{\text{tp}} \cdot \tau_0}{p_{\text{tp}} \cdot \tau_0 + p_{\text{fp}} \cdot (1 - \tau_0)} = \frac{0.95\,\tau_0}{0.95\,\tau_0 + 0.10\,(1 - \tau_0)}$$

**After $R = \text{unsafe}$:**

$$P(T_r = 1 \mid R = \text{unsafe}) = \frac{(1 - p_{\text{tp}}) \cdot \tau_0}{(1 - p_{\text{tp}}) \cdot \tau_0 + (1 - p_{\text{fp}}) \cdot (1 - \tau_0)} = \frac{0.05\,\tau_0}{0.05\,\tau_0 + 0.90\,(1 - \tau_0)}$$

**General form** (for any response $r$ with likelihood $L_r$):

$$\tau_1 = \frac{L_r(\text{traversable}) \cdot \tau_0}{L_r(\text{traversable}) \cdot \tau_0 + L_r(\text{impassable}) \cdot (1 - \tau_0)}$$

The posterior $\tau_1$ is the updated traversability score stored in the traversability map for that region.

### B.4 Sequential Update (Multi-Turn)

If the robot encounters the same terrain type across multiple frames or interactions, the posterior from turn $k$ becomes the prior for turn $k+1$:

$$\tau_0^{(k+1)} = \tau_1^{(k)}$$

This enables the robot to progressively sharpen its estimate of a terrain type's traversability through repeated exposure. After $K$ consistent "safe" responses, the traversability score converges toward 1.0 asymptotically, and after $K$ consistent "unsafe" responses it converges toward 0.0.

**Numerical examples:**

| Prior $\tau_0$ | Response | Posterior $\tau_1$ |
|---|---|---|
| 0.50 (unknown) | safe | 0.905 |
| 0.50 (unknown) | unsafe | 0.053 |
| 0.80 (grass, prior) | safe | 0.974 |
| 0.80 (grass, prior) | unsafe | 0.182 |
| 0.10 (mud, prior) | safe | 0.514 |
| 0.10 (mud, prior) | unsafe | 0.006 |

The update is weakly informative for high-prior safe classes (grass: 0.80 → 0.97 after "safe") and strongly informative for unknown regions (0.50 → 0.91 after "safe"), which is the desired behavior.

### B.5 Implementation in `map_updater.py`

The `MapUpdater.apply_feedback()` method receives the `user_response` string and the `prior_score` from the current traversability map. Instead of hard-coding 0.9 / 0.0:

```python
def _bayesian_update(prior: float, is_safe: bool,
                     p_tp: float = 0.95, p_fp: float = 0.10) -> float:
    if is_safe:
        num = p_tp * prior
        den = p_tp * prior + p_fp * (1.0 - prior)
    else:
        num = (1.0 - p_tp) * prior
        den = (1.0 - p_tp) * prior + (1.0 - p_fp) * (1.0 - prior)
    return num / den if den > 0.0 else 0.0
```

The `prior` is read from `tmap.score_at()` over the target region before updating.

---

## C. Instruction Uncertainty Scoring

### C.1 Ambiguity Detection via Slot-Filling

Instruction uncertainty is detected by an LLM parser that identifies missing or underspecified semantic slots in the user command. Given instruction $\ell$ and scene description $s$, the parser produces a JSON output:

```json
{
  "ambiguity_type": "missing_object" | "missing_direction" | "missing_distance" |
                    "missing_action" | "ambiguous_target" | "ambiguous_action" | "none",
  "ambiguity_score": <float in [0,1]>,
  "missing_slot":    "object" | "direction" | "distance" | "action" | null,
  "affected_trajectories": "all" | "some" | "none"
}
```

### C.2 Ambiguity Score Definition

The ambiguity score $u_I \in [0, 1]$ combines the LLM's confidence in detecting an ambiguity and the severity of the identified slot gap:

$$u_I(\ell, s) = w_{\text{type}}(t^*) \cdot P(\text{ambiguous} \mid \ell, s)$$

where:
- $P(\text{ambiguous} \mid \ell, s) \in [0, 1]$ is the LLM's probability that the command is ambiguous (obtained from the normalized token probability of the "ambiguous" vs. "unambiguous" classification)
- $w_{\text{type}}(t^*) \in \{0.25, 0.50, 0.75, 1.0\}$ is a severity weight for the identified ambiguity type:
  - $w = 1.00$ for missing_action (no executable verb — fully execution-blocking)
  - $w = 0.75$ for missing_object, ambiguous_target (referent unknown — trajectory selection blocked)
  - $w = 0.50$ for ambiguous_action, missing_direction (path-blocking but partially recoverable with context)
  - $w = 0.25$ for missing_distance (can estimate or proceed cautiously)
  - $w = 0.00$ for "none" (no ambiguity)

### C.3 Threshold Calibration

The robot asks when $u_I > \theta_{\text{ask}}$. The threshold is calibrated using CP on the instruction-only calibration set. Each Type-1 entry in `nav_calibration.json` provides a ground-truth label $y_i \in \{A, B\}$. The non-conformity score is:

$$\kappa_i^I = 1 - \hat{f}(\tilde{x}_i^I)_{y_i}$$

where $\hat{f}(\tilde{x}_i^I)_B = \sigma(u_I(\ell_i, s_i) - \theta)$ is the model's probability of the ASK option (using a sigmoid transform of the ambiguity score relative to the current threshold). The CP quantile $\hat{q}$ calibrates the threshold to achieve the desired coverage.

In practice, $\theta_{\text{ask}}$ is found by binary search over the calibration set such that the empirical FPR (asking when $y_i = A$) does not exceed a user-specified false-positive budget $\alpha$, subject to the coverage constraint $P(y_i \in C) \geq 1 - \varepsilon$.

### C.4 Action Correlation via Constraint Violation Count

After detecting ambiguity type $t^*$, the system counts how many of the $K = 3$ candidate trajectories are infeasible under the ambiguous instruction:

$$\text{vc}(\ell, t^*) = |\{\tau_k \in \mathcal{T} : \tau_k \text{ is infeasible given } t^*\}|$$

The corrected prediction set size accounts for this:

$$|C(\tilde{x})| = \begin{cases} 1 & \text{if } u_I \leq \theta_{\text{ask}} \text{ and } \text{vc} = 0 \\ 2 & \text{if } u_I > \theta_{\text{ask}} \text{ or } \text{vc} > 0 \end{cases}$$

When $|C| > 1$, the robot enters the ASK state. This formulation ensures that even low-ambiguity-score commands that constrain the action space (e.g., "go to the rock" when three rocks are visible) correctly trigger clarification.

---

## D. Relationship to Prior Work

| Component | KnowNo | WhenToAsk/UPS | This work |
|-----------|--------|---------------|-----------|
| Uncertainty source | Instruction (LLM confidence) | Instruction + policy incapability | Instruction + environmental perception |
| Calibration framework | Conformal prediction (CP) | CP over VLM verifier | Joint CP over max-pooled non-conformity |
| Score function | $1 - \hat{f}(\tilde{x})_{y^*}$ | VLM verifier score | $\max(\kappa^I, \kappa^E)$ |
| Environment branch | None | None | $\kappa^E = 1 - g(x^E)$ |
| Update rule | None (single-turn) | Residual learning | Bayesian posterior update |
| Coverage guarantee | Marginal, $1-\varepsilon$ | Marginal, $1-\varepsilon$ | Marginal, $1-\varepsilon$ (conservative) |

The key theoretical advance is the joint non-conformity score, which is the first formulation we are aware of that provides CP coverage guarantees when uncertainty can arise from two structurally different sources simultaneously.
