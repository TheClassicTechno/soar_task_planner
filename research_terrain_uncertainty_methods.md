# Novel Non-LLM Methods for Terrain Uncertainty in the 6-Step Pipeline
## Research Notes — May 6, 2026

---

## 0. Starting Point: What the Mentor Said (Exact Criticisms)

From `may6meeting.txt` and `may6slides.txt`, the mentor's criticisms were:

1. **VLM/LLM scoring is unreliable.** Scores are inconsistent across prompts, sensitive to wording, not suitable
   as numerical control signals (may6meeting §6, §13).
2. **Current innovations mostly repeat KnowNo + WhenToAsk.** Using conformal prediction on LLM probabilities
   for instruction ambiguity was already done. No real innovation yet (may6meeting §12).
3. **Personalization is product design, not research.** Drop verbosity/user profile unless it directly helps
   uncertainty resolution (may6meeting §7, §19).
4. **LLMs are bad at geometric/numerical reasoning.** Can't feed raw (x,y) coordinates to an LLM and get reliable
   outputs (may6meeting §18).
5. **No prior work handles environmental/terrain uncertainty with a language clarification loop.**
   This is the open gap (may6slides, prior work table).

**Lines 396–414 of may6slides.txt** (the specific action items):
- Line 396: Find papers on whether verbosity level affects response quality.
- Lines 399–402: Use syngraph/object graph + traversability score. "Previous work updates instruction uncertainties
  but **no ideas on environment uncertainties**. Use traversability score to make this more unique/novel."
- Line 412: "We have to do something for terrain uncertainty that is new, novel, and **feasible**."

---

## 1. The Gap: What Prior Work Actually Does vs. Does Not Do

| Paper | Core Method | Handles | Gap |
|---|---|---|---|
| KnowNo [Ren et al., CoRL 2023] | CP on LLM MCQA softmax scores. s_j = 1 − p_{y*}(x_j). C(x) = {y : p_y(x) ≥ 1−q̂}. |C|>1 → ASK | Instruction ambiguity only | No spatial map; no terrain; score is LLM logit |
| WhenToAsk/UPS [Yuan et al., arXiv 2026] | CP on VLM verifier over action outcome narrations. Three-way: act/ask/learn via residual policy. κ(x̄,ȳ) = 1 − min p^VLM | Instruction ambiguity + policy incapability | No terrain; still uses VLM scoring |
| GA-Nav | Fixed traversability lookup table | Terrain traversability | No language instructions; no ask/act; static table |
| IntroPlan | CP + memory on instruction uncertainty | Instruction ambiguity + memory | No environmental perception |

**The actual gap:** No paper (a) uses conformal prediction or Bayesian methods on **visual/geometric terrain features**,
(b) builds a **spatially indexed uncertainty map** that can be updated from user language feedback,
or (c) combines terrain and instruction uncertainty in one pipeline with formal guarantees.

---

## 2. Why LLM/VLM Scoring Fails for Terrain

The mentor's concern is backed by the literature. VLMs are well-calibrated on language tasks but systematically
overconfident or inconsistent on numerical/geometric outputs [Angelopoulos et al., ICLR 2021]. Specifically:

- LLM softmax scores for terrain descriptors ("mud", "gravel") vary with prompt wording.
- They cannot reason reliably over 2D spatial coordinates [may6meeting §18].
- A score of 0.7 from one prompt vs. 0.5 from a rephrased prompt is the same physical terrain — the VLM
  didn't observe anything different.

**Consequence for the pipeline:** The nonconformity score in conformal prediction must be **stable and reproducible**
for the CP coverage guarantee to be meaningful. LLM logits for terrain are not stable. Therefore, we need
a different uncertainty signal for the terrain branch.

---

## 3. Proposed Novel Technical Methods (Non-LLM, Non-VLM)

### 3.1 Step 1 (Detection): Self-Supervised Reconstruction Error as Terrain Anomaly Score

**Motivation:** Instead of asking an LLM "how unsafe is this terrain?", use the robot's own
traversal history. Terrain the robot has safely driven through is by definition known. Any terrain
visually unlike what it has driven through is uncertain.

**Method (Schmid et al., IROS 2022):**

The robot collects image patches from its own driven trajectory (self-supervised, no human labels).
An autoencoder A = Dec ∘ Enc is trained exclusively on these **known-traversed patches**:

```
L_train = E_{p ∈ traversed} [ ||p - Dec(Enc(p))||² ]
```

At inference, for any new terrain patch p_new:

```
Score_E(p_new) = ||p_new - Dec(Enc(p_new))||² / σ²_baseline
```

where σ²_baseline is the variance of reconstruction errors on the training set. High Score_E means the
terrain is unlike anything the robot has traversed before → unknown/uncertain.

**Why this is novel for this pipeline:** Schmid et al. use it for binary safe/unsafe classification.
This work proposes applying **conformal prediction on top of Score_E** to get a formal guarantee:

```
Calibration set:  {(p_i, label_i)} where label_i ∈ {known, unknown}
Nonconformity score:  s_i = Score_E(p_i)  for known terrain patches
Threshold:  q̂ = ⌈(N+1)(1-α)⌉/N quantile of {s_i}

Test: if Score_E(p_test) > q̂  →  terrain is "unknown" with coverage guarantee P ≥ 1-α
```

This gives the **same formal guarantee as KnowNo** but derived from a **visual reconstruction signal,
not LLM logits**. The nonconformity score is stable and reproducible because it is computed
deterministically from the autoencoder weights and the image patch.

**References:**
- Schmid et al., "Self-Supervised Traversability Prediction by Learning to Reconstruct Safe Terrain," IROS 2022.
- Angelopoulos et al., "Uncertainty Sets for Image Classifiers using Conformal Prediction," ICLR 2021.

---

### 3.2 Alternative Detection: Mahalanobis Distance on Visual Features

A second non-LLM detection approach: extract visual features φ(patch) from a frozen vision encoder
(e.g., DINOv2 ViT-S/14). Fit a multivariate Gaussian to features of known terrain:

```
μ_known = (1/N) Σᵢ φ(pᵢ)
Σ_known = (1/N) Σᵢ (φ(pᵢ) − μ_known)(φ(pᵢ) − μ_known)ᵀ
```

Mahalanobis distance for a test patch:

```
D_M(p) = sqrt[ (φ(p) − μ_known)ᵀ Σ_known⁻¹ (φ(p) − μ_known) ]
```

Under the null hypothesis (known terrain), D_M follows a chi-squared distribution with k degrees of freedom.
A calibrated threshold τ_M can be derived from χ²(k, 1-α) or via conformal prediction on calibration data.

If D_M(p) > τ_M → terrain is statistically out-of-distribution → uncertain.

**Advantages over VLM scoring:**
- Fully deterministic (same image = same D_M every run)
- No prompting, no API call latency
- Chi-squared distribution gives principled threshold

**References:**
- Lin et al., "Detecting Anomalies in Unmanned Vehicles Using the Mahalanobis Distance," ICRA 2010.
- Yang et al., "Unsupervised Anomaly Detection via Mahalanobis SVDD," arXiv 2025.
- Arafin et al., "Advances and Trends in Terrain Classification for Off-Road Perception," JFR 2025.

---

### 3.3 Step 3 (Correlation): Gaussian Process Traversability Map with Trajectory Overlap Test

**Motivation:** Once terrain is flagged as uncertain, we need to correlate it to robot actions
(which trajectories are affected). The mentor noted this must be spatial — LLMs can't reason over (x,y).

**Method (Leininger et al., ICRA 2024; Hou et al., IROS 2025):**

Model terrain traversability as a function over 2D positions using a Sparse Gaussian Process:

```
T(x, y) ~ GP( μ_0, k_SE((x,y), (x',y')) )

k_SE(p, p') = σ²_f · exp( −||p − p'||² / (2ℓ²) )
```

where σ²_f is signal variance, ℓ is the length-scale (how far the correlation extends spatially).

After incorporating sensor observations {(p_i, t_i)}:

```
μ_post(p*) = K(p*, X) [K(X,X) + σ²_n I]⁻¹ t
σ²_post(p*) = k(p*,p*) − K(p*,X) [K(X,X) + σ²_n I]⁻¹ K(X,p*)
```

**Trajectory uncertainty score (novel formula):**

For a candidate trajectory τ = {p_1, p_2, ..., p_T} (a sequence of 2D waypoints):

```
U_traj(τ) = max_{t} σ_post(p_t)      [maximum GP uncertainty along path]

LCB_traj(τ) = min_{t} [ μ_post(p_t) − β · σ_post(p_t) ]   [lower confidence bound]
```

where β controls conservatism (e.g., β = 2 for ~95% confidence lower bound).

If `LCB_traj(τ) < τ_safe` (threshold), trajectory τ passes through uncertain terrain → enter ASK.

**Why novel:** KnowNo and WhenToAsk have no spatial map at all. The GP uncertainty map:
1. Is spatially indexed (each position has a mean and variance)
2. Is updated as the robot observes terrain
3. Gives a principled traversability lower-bound per trajectory

---

### 3.4 Step 5 (Update): GP Bayesian Posterior Update from User Language Feedback

**Motivation:** After user says "yes, that terrain is safe" or "no, avoid it", we want to update
the traversability map in the affected region. This is non-trivial because the user describes
a spatial region in language, not as exact coordinates.

**Proposed update method:**

1. The system identifies the terrain patch in question (from Detection step — it knows which patch
   triggered ASK, and it knows the approximate 2D footprint of that patch on the map).

2. User responds: "yes, safe" → y_obs = 1 (traversable); "no, avoid" → y_obs = 0 (non-traversable).

3. Add observation (p_center, y_obs) to the GP with lower noise variance (σ²_n_user < σ²_n_sensor)
   to reflect the user's explicit confirmation:

```
GP posterior update with new point (p_user, y_obs):
μ_post_new = K(p*, [X; p_user]) [K([X;p_user],[X;p_user]) + σ²_n I]⁻¹ [t; y_obs]
```

4. The GP posterior uncertainty σ²_post contracts in the neighborhood of p_user:
   **the robot gets more certain about that terrain**.

5. For the Bayesian traversability posterior (existing slide formula):

```
τ₁ = p_tp · τ₀ / (p_tp · τ₀ + p_fp · (1 − τ₀))

where τ₀ = μ_post(p_user)  [GP mean at user-pointed location]
      p_tp = 0.95           [true positive rate of user saying "safe" when safe]
      p_fp = 0.10           [false positive rate]
```

This connects the existing Bayesian update formula to the GP map: **the prior τ₀ comes from the GP
mean, not from a hard-coded value**. After update, write back τ₁ to the GP as a new observation.

**Why novel:** No prior paper (KnowNo, WhenToAsk, GA-Nav) maintains a spatially indexed GP map
and updates it from user language feedback. Wellhausen et al. (WVN) do online self-supervised update
from robot motion, but not from user language responses. This is a new combination.

**References:**
- Leininger et al., "Gaussian Process-based Traversability Analysis," ICRA 2024.
- Hou et al., "Real-time Spatial-temporal Traversability Assessment via Feature-based Sparse GP," IROS 2025.
- Stephens et al., "Planning under Uncertainty for Safe Robot Exploration using GP Prediction," Autonomous Robots 2024.
- Cai et al., "Probabilistic Traversability Model for Risk-Aware Motion Planning," IROS 2023.

---

### 3.5 Environment Representation: Scene Graph with Probabilistic Traversability Nodes

**Motivation:** Mentor recommended scene graph or object graph over semantic Gaussian maps
(too complex) but above a simple SQL database (no spatial structure). We need something
in between.

**Proposed: Object Graph with GP Traversability Nodes**

Each node in the graph represents an object or terrain region with:

```python
Node = {
    "id": unique_id,
    "label": terrain_class,          # e.g., "gravel", "unknown_surface_1"
    "position": (x, y),              # 2D map position
    "traversability_mean": μ,        # GP posterior mean at this position
    "traversability_var": σ²,        # GP posterior variance
    "certainty_level": "unknown" | "clarified" | "observed",
    "user_confirmed": bool,
    "adjacent_trajectories": [traj_ids]  # which planned trajectories intersect this node
}
```

Edges represent spatial adjacency or trajectory connectivity.

**Update after user clarification:**

```
Node.traversability_mean ← τ₁  (from Bayesian update above)
Node.traversability_var  ← σ²_post after GP update
Node.certainty_level     ← "clarified"
Node.user_confirmed      ← True
```

**Why this is simpler than semantic Gaussian maps:**
- No 3D reconstruction required
- No GPU for 3D-GS rendering
- Runs on a wheeled robot with a 2D map
- Spatial update is a simple GP posterior computation

**Why more structured than SQL:**
- Encodes spatial relationships between terrain regions
- Can be queried: "which nodes intersect trajectory τ?"
- Can be updated incrementally

**References:**
- Ginting et al., "SEEK: Semantic Reasoning for Object Goal Navigation," RSS 2024.
- Yang et al., "Probabilistic Modeling of Semantic Ambiguity for Scene Graphs," CVPR 2021.
- Wilson et al., "Modeling Uncertainty in 3D Gaussian Splatting through Continuous Semantic Splatting," arXiv 2024.
  [This is most likely the paper the mentor referenced — uses Dirichlet conjugate prior over Gaussian ellipsoids]

---

## 4. The Joint Ask/Proceed/Stop Decision with Formal Guarantees

**This is the novel mathematical core connecting both uncertainty types.**

KnowNo: κ_I = 1 − p_{y*}(instruction) (LLM logit-based, instruction only)
WhenToAsk: κ_UPS = 1 − min p^VLM(action|narration) (VLM-based, instruction + policy failure)

**Proposed joint score for both types:**

```
κ_I(ℓ, s)     = Instruction nonconformity score (can use LLM slot-fill, KnowNo-style)
κ_E(patch, τ)  = max( Score_E(patch), U_traj(τ) / U_max )   [terrain score, normalized]

κ_joint = max(κ_I, κ_E)
```

With a single CP threshold q̂ calibrated on a combined calibration set:

```
q̂ = ⌈(N+1)(1-α)⌉/N quantile of {max(κ_I_j, κ_E_j)}

Decision:
  if κ_joint ≤ q̂:     PROCEED  (robot is confident on both branches)
  if κ_joint > q̂:     ASK      (uncertainty in at least one branch)
  if LCB_traj(τ*) < τ_danger:  STOP  (no safe trajectory exists)
```

**Formal coverage guarantee (from conformal prediction theory):**

```
P( correct action ∈ C(x_test) ) ≥ 1 − α
```

**What's new vs. KnowNo:** The nonconformity score κ_E comes from reconstruction error or GP variance —
not from LLM logits. This gives stability to the CP threshold.

**What's new vs. WhenToAsk:** WhenToAsk does not have a spatially indexed terrain map.
It handles policy incapability (the policy itself fails) but not terrain uncertainty
(the policy could execute, but the terrain is unknown). These are different problems.

---

## 5. Non-AI Methods Summary Table

| Pipeline Step | LLM/VLM Approach (current) | Proposed Non-AI Alternative | Key Paper |
|---|---|---|---|
| Step 1: Detect unknown terrain | VLM scores terrain images (unreliable) | Autoencoder reconstruction error Score_E | Schmid et al., IROS 2022 |
| Step 1: Detect unknown terrain (alt) | VLM scores terrain images | Mahalanobis distance on DINOv2 features | Lin et al., ICRA 2010; Wellhausen et al., RA-L 2020 |
| Step 3: Correlate to trajectories | LLM reasons over (x,y) (unreliable) | GP posterior variance + trajectory LCB | Leininger et al., ICRA 2024 |
| Step 3: ASK/PROCEED decision | LLM threshold (inconsistent) | CP on Score_E or GP variance | Angelopoulos et al., ICLR 2021 |
| Step 5: Update terrain knowledge | Hard-coded τ = 0.9/"safe" | GP Bayesian posterior update | Hou et al., IROS 2025; Stephens et al., 2024 |
| Step 5: Memory representation | SQL/JSON (no spatial structure) | Scene graph with GP traversability nodes | Ginting et al., RSS 2024 |
| Step 6: Trajectory selection | FM Scorer (VLM, latency issues) | Max min-LCB trajectory from GP map | Cai et al., IROS 2023 |

---

## 6. The Verbosity Question (Lines 396 of may6slides.txt)

**Question from slides:** "If we have different levels of verbosity, all different levels matter to user — can we
get clearer responses or same responses? Find more papers on this."

**What the literature says:**

The HRI literature does not have a simple "more verbose = better" or "shorter = better" result.
The key variable is **information content**, not raw word count.

- **Rosenthal et al. (RO-MAN 2009, IJSR 2012):** Adding meaningful grounding content to questions
  (robot's prediction, uncertainty level, sensor context) improved response accuracy significantly.
  But this is adding *relevant* content — not just making questions longer.
  
- **Deits et al. (JHRI 2013):** Information-theoretic question selection. Select question q* that
  maximizes entropy reduction: q* = argmax_q [H(M) − H(M|answer(q))]. More targeted/constrained
  questions asked fewer times but achieved higher task accuracy. This argues for **precision over verbosity**.

- **Cakmak & Thomaz (HRI 2012):** In a teaching paradigm, highly specific constrained questions
  (feature queries) extracted more useful signal per interaction than open-ended queries (label queries).

**Conclusion for the pipeline:** The 9-style verbosity scheme (3 levels × 3 formats) is probably
over-engineered for the research goal. The more scientifically defensible argument is:

> Questions that include the robot's uncertainty level, the terrain label, and the candidate actions
> as grounding context produce more accurate user responses — not because they are longer or shorter,
> but because they provide the **minimum sufficient information** for the user to resolve the uncertainty.

If the team keeps a verbosity study, it should frame it as: *does grounding context (terrain label +
candidate trajectories) improve response accuracy vs. a generic "I'm not sure, should I proceed?"*
That's a testable hypothesis backed by Rosenthal et al.'s framework.

---

## 7. What is Still LLM/VLM (And Where That's Acceptable)

Not everything needs to be replaced. The mentor's concern was about **LLM scoring** being unreliable.
Using LLMs for language generation (converting structured inputs to natural-language questions) is
acceptable, because it is a generation task, not a scoring task.

| Component | LLM/VLM acceptable? | Reason |
|---|---|---|
| Terrain anomaly score | NO | Inconsistent; not reproducible |
| Trajectory traversability score | NO | Can't reason over (x,y) reliably |
| Instruction ambiguity classification | CONDITIONAL | Use structured slot-fill, test consistency on calibration set |
| Question generation from structured input | YES | LLM generates language, not numbers |
| Terrain label description (what is this region?) | YES | Semantic description, not score |

---

## 8. Summary of Truly Novel Contributions (vs. KnowNo and WhenToAsk)

| Contribution | vs. KnowNo | vs. WhenToAsk | What's new |
|---|---|---|---|
| Reconstruction-error CP for terrain detection | KnowNo: CP on LLM logits | UPS: CP on VLM narration scores | Different nonconformity score source: visual geometry, not language model |
| GP spatial traversability map | No spatial map | No spatial map | First principled spatial uncertainty field in this pipeline type |
| GP Bayesian update from user language response | No update mechanism | Residual policy update (different) | Bridges user natural language → spatial map update |
| Scene graph with GP uncertainty nodes | No environment representation | No environment representation | Structured spatial memory enabling repeated-scene recognition |
| Joint CP decision with terrain + instruction | Instruction only | Instruction + policy capability | Terrain as first-class uncertainty source with CP guarantee |

---

## 9. Immediate Next Steps (Research, Not Implementation Yet)

1. **Verify the GP traversability approach is feasible given the current dataset** — the calibration
   set has 26 terrain examples (Type 2). Are these enough to calibrate a CP threshold on Score_E?
   Answer: Yes for CP calibration. Angelopoulos et al. (ICLR 2021) show CP is valid for small N.
   With N=26, the bound is q̂ = (27)(0.9)/26 = 93.5th percentile — still valid.

2. **Define the autoencoder architecture** — a small VAE (Conv3 encoder, 64-dim latent) trained
   on RUGD terrain patches is feasible on a single GPU. Schmid et al. (IROS 2022) used a U-Net style
   autoencoder; a smaller architecture is fine for this scope.

3. **Decide whether to implement full GP or simplified version** — A sparse GP with 50-100 inducing
   points is computationally tractable in real-time (Leininger et al., ICRA 2024 showed this).
   A simpler alternative: a 2D grid of Bernoulli cells with Bayesian updates (Fankhauser et al.,
   RA-L 2018) — lower math complexity, still principled.

4. **Build the scene graph data structure** — this is a pure engineering task (Python dict of nodes
   with GP-valued fields). No new libraries needed.

5. **Run baseline comparison** — compare reconstruction-error CP against static traversability
   table (GA-Nav-style) and VLM scoring on RUGD test set. Metric: ask/proceed accuracy (true
   positive rate of entering ASK state when terrain is actually uncertain).

---

---

## 11. Gap Audit — Items Not Fully Covered in First Pass

### Gap 1 (HIGH): FM Scorer must be removed as a claimed innovation

**Problem:** Slide Innovation 3 says "Foundation Model traversability scorer (FM Scorer): Claude/Gemini
as judge per terrain class" is a novel contribution. But the mentor explicitly said VLM/LLM scoring
is inconsistent and unreliable (may6meeting §6, §13). These two statements directly contradict each other.

**Fix:** Remove FM Scorer as a claimed innovation. Replace with:

> **Innovation 3 (revised):** A Gaussian Process traversability lower-confidence-bound (GP-LCB)
> replaces the FM Scorer for terrain safety decisions. The GP-LCB provides a mathematically principled,
> deterministic uncertainty signal derived from spatial observations — no API call, no prompt sensitivity,
> guaranteed reproducibility.

The FM Scorer (Claude/Gemini as judge) may still be used for **question generation** (producing
natural-language descriptions of what terrain was detected), because that is a language task, not
a numerical scoring task. But it should not appear as a traversability confidence signal.

---

### Gap 2 (HIGH): Innovation 4 (9 question styles) is product design

**Problem:** Slide Innovation 4 claims "Questions change based on who the user is" (9 styles:
3 verbosity × 3 formats) is a research innovation. Mentor explicitly said this is product design,
not research (may6meeting §7, §19). It can only be included if it demonstrably helps uncertainty
resolution.

**Fix:** Reposition Innovation 4 as future work:

> *Future work: Whether personalized question verbosity (e.g., grounding context vs. minimal phrasing)
> improves user response accuracy in terrain uncertainty scenarios. Rosenthal et al. (RO-MAN 2009)
> show that meaningful grounding content improves accuracy; Deits et al. (JHRI 2013) show that
> targeted questions reduce clarification rounds. A controlled user study could validate whether
> one of these styles outperforms in the robot navigation context.*

---

### Gap 3 (HIGH): "Our approach" for terrain detection is still labeled ChatGPT/Gemini in diagram

**Problem:** The environmental branch diagram (lines 371–394 of may6slides.txt) shows:
- Baseline: SAM3
- Our approach: ChatGPT/Gemini to identify the uncertain part

But the mentor said VLM scoring is unreliable. ChatGPT/Gemini cannot be "our approach" for terrain.

**Fix:** Replace the diagram labeling:
- Baseline 1: Static traversability lookup table (GA-Nav style)
- Baseline 2: SAM3 segmentation + VLM scoring
- **Our approach: Reconstruction error CP (Schmid et al.) or Mahalanobis distance on DINOv2 features,
  with CP threshold calibrated from RUGD examples**

This directly addresses what the mentor said about needing something more concrete.

---

### Gap 4 (MEDIUM): Joint score κ_joint tension with mentor pushback

**Problem:** The slides propose κ_joint = max(κ_I, κ_E). The mentor pushed back (may6meeting §4):
"The robot handles one uncertainty type at a time. If instruction is ambiguous, that is the source.
If terrain is uncertain during navigation, that is the source. A joint score may not be needed."

**Resolution:** The mentor's pushback applies to the **common case** (sequential uncertainty).
The joint score is still valid for the **edge case** the mentor explicitly noted: "what if the robot
is already in uncertain terrain and the user gives an ambiguous instruction to continue?"
In that case both branches are simultaneously active and max(κ_I, κ_E) is the correct conservative
decision. The document should explicitly define when each applies:

```
Normal case:   Only one branch is active at a time → use single-branch CP score
Edge case:     Both active simultaneously → use κ_joint = max(κ_I, κ_E) conservatively
```

Do NOT claim the joint score as a novel innovation — it follows directly from standard CP algebra.
It is an engineering design choice, not a mathematical contribution.

---

### Gap 5 (MEDIUM): Bernoulli/Beta-Bernoulli canonical reference missing

The slides correctly state that terrain uncertainty is binary (Bernoulli). The canonical paper for
Bayesian per-cell Bernoulli traversability is:

**Shan et al., "Bayesian Generalized Kernel Inference for Terrain Traversability Mapping," CoRL 2018.**

Formula (Beta conjugate prior over per-cell traversability θ):
```
Prior:    θ ~ Beta(α₀, β₀)
Update:   θ | n traversable, m non-traversable ~ Beta(α₀ + n, β₀ + m)
Posterior mean:  P(traversable) = (α₀ + n) / (α₀ + β₀ + n + m)
```

For the binary "safe/unsafe" user response:
- User says "safe" → add n=1, m=0 → posterior mean increases toward 1
- User says "unsafe" → add n=0, m=1 → posterior mean decreases toward 0
- Unknown terrain prior: α₀ = β₀ = 1 → P = 0.5 (maximum entropy, consistent with the slides)

This is the mathematical foundation for the binary case, different from the continuous GP approach.
Use Bernoulli/Beta when terrain is clearly binary (traversable/not), and GP when continuous
traversability scores are needed (e.g., for trajectory cost functions).

**Reference:** Shan, T., Wang, J., Englot, B., Doherty, K. CoRL 2018. PMLR 87:829–838.

---

### Gap 6 (MEDIUM): SAM3 actual speed numbers

SAM3 is a real released model (Meta, November 2025, arXiv:2511.16719 "SAM 3: Segment Anything
with Concepts"). Actual speed measurements:

- SAM2 (A100): ~44 FPS — **acceptable for real-time robot navigation**
- SAM3 (H200, user-reported GitHub issue #425): **5–6 FPS** — too slow for real-time RGB input
- SAM3.1 (H100, released March 2026): ~32 FPS — borderline, only with multiplexed tracking

The mentor was correct: SAM3 may not be fast enough for real-time navigation. This supports using
reconstruction error or Mahalanobis distance as faster, deterministic alternatives:
- Autoencoder reconstruction error: ~50ms for a patch on CPU (effectively real-time)
- Mahalanobis distance on frozen DINOv2 features: <5ms per patch (deterministic matrix multiply)

If SAM3 is used, it should be part of a **local map update pipeline** (not per-frame analysis):
observe from distance → segment into map → update object graph → plan. This is consistent with
the mentor's note that a local map can tolerate lower segmentation frequency.

---

### Gap 7 (MEDIUM): Terrain memory — avoiding re-asking about confirmed terrain

**Problem:** The slides say "confirmed intent stored in memory keyed by (instruction_pattern,
scene_hash)". This is for instruction uncertainty. The terrain analog is missing: how does the
robot remember that it already confirmed a specific terrain patch is traversable?

**Proposed terrain memory key:**
```
terrain_memory[(terrain_label, position_cell_id)] = {
    "traversability": τ₁,   # posterior after user update
    "certainty": "confirmed",
    "timestamp": t
}
```

At runtime: before entering ASK mode for terrain, query `terrain_memory` with the terrain label
and approximate cell position. If a confirmed entry exists and the cell is nearby, skip ASK and
use the stored traversability value.

This is directly analogous to the instruction memory:
- Instruction: `(instruction_pattern, scene_hash)` → confirmed intent
- Terrain: `(terrain_label, position_cell_id)` → confirmed traversability

---

### Gap 8 (MEDIUM): Non-LLM alternatives for INSTRUCTION ambiguity detection

We focused all non-LLM work on terrain. For instruction ambiguity, the mentor also warned that
LLM scoring may be inconsistent. Pre-LLM structured alternatives:

**Tellex et al., "Understanding Natural Language Commands for Robotic Navigation," AAAI 2011.**
Method: Generalized Grounding Graphs (G³). Stanford dependency parse → Spatial Description Clauses
(SDCs). If an SDC slot cannot be grounded, it triggers a clarification question. No LLM.
Gap: requires a fixed action/object vocabulary; brittle to novel phrasings.

**Thomason et al., "Improving Grounded NLU through Human-Robot Dialog," ICRA 2019, JAIR 2020.**
Method: Learned CCG semantic parser → formal logical form. Slot confidence below threshold →
clarification. Learned from 1,500 dialog examples.
Gap: still requires labeled dialog data to train the parser.

**Practical recommendation for this project:**
Use LLM slot-fill as the primary approach but add a **consistency check**:
run the same instruction through the LLM 3× with temperature=0; if the ambiguity_type differs
across runs, flag the output as unreliable and fall back to a conservative "ASK" default.
This does not require pre-LLM infrastructure but directly addresses the consistency concern.
Back it with: Angelopoulos et al. (ICLR 2021) who note that CP requires a stable score function.

---

### Gap 9 (MEDIUM): Evaluation metrics not specified

The slides mention "success rate" and qualitative results but no concrete evaluation design.
From the mentor: evaluate each innovative part separately.

**Proposed evaluation design (backed by prior work):**

For terrain detection (Step 1):
- Metric: F1 score on ask/proceed/stop classification using RUGD ground-truth annotations
- Baseline: Static table (GA-Nav style); Our method: Reconstruction-error CP
- Standard: Wellhausen et al. (RA-L 2020) uses AUROC on anomaly detection

For GP update (Step 5):
- Metric: Traversability estimate MAE before vs. after user update on held-out positions
- Baseline: Bayesian traversability posterior with hard-coded τ₀=0.5 vs. GP-prior τ₀

For instruction detection (Step 1, instruction branch):
- Metric: Ambiguity type classification accuracy on KnowNo navigation examples
- Baseline: Majority-class; Our method: LLM slot-fill with consistency check

For ask/proceed decision (Step 3):
- Metric: Coverage guarantee validation — does P(correct ∈ C) ≥ 1-α hold empirically?
- Standard: Exact same validation used by KnowNo and WhenToAsk

For end-to-end pipeline:
- Metric: Task completion rate (robot reaches goal without collision or user over-correction)
- Compare: KnowNo only vs. our unified pipeline on combined instruction + terrain test set

---

### Gap 10 (LOW): IntroPlan vs. this work

IntroPlan (referenced in prior work table) handles instruction ambiguity via CP + memory retrieval
but assumes the robot policy can always execute once it knows the instruction. It does not handle
cases where terrain blocks execution even with a clear instruction. This is the gap our pipeline
fills on the environmental side.

IntroPlan is closer to this work than KnowNo (it has memory, not just single-step CP), so the
key differentiation is: IntroPlan has no environment representation, no terrain update, and no
terrain-triggered ASK. We add all three.

---

### Gap 11 (LOW): Semantic segmentation map as middle-ground option

The mentor listed semantic segmentation map as an option between SQL and full Gaussian splatting.
This corresponds to: a 2D grid where each cell stores a semantic label + uncertainty score.
Relevant reference: Fankhauser et al. (RA-L 2018) for probabilistic cell-level estimates.
This is simpler than the GP approach (no covariance kernel) but loses spatial correlation.
Recommended only if GP is too slow at runtime.

---

## 10. References (Formatted for Paper)

1. Ren, A.Z., et al. "Robots That Ask For Help: Uncertainty Alignment for Large Language Model Planners." CoRL 2023.
2. Yuan, J., Wu, Y., Bajcsy, A. "When to Act, Ask, or Learn: Uncertainty-Aware Policy Steering." arXiv 2026.
3. Schmid, R., et al. "Self-Supervised Traversability Prediction by Learning to Reconstruct Safe Terrain." IROS 2022.
4. Leininger, A., et al. "Gaussian Process-based Traversability Analysis for Terrain Mapless Navigation." ICRA 2024.
5. Hou, Z., et al. "Real-time Spatial-temporal Traversability Assessment via Feature-based Sparse GP." IROS 2025.
6. Stephens, A., et al. "Planning under Uncertainty for Safe Robot Exploration using GP Prediction." Autonomous Robots 2024.
7. Angelopoulos, A., et al. "Uncertainty Sets for Image Classifiers using Conformal Prediction." ICLR 2021.
8. Lin, R., et al. "Detecting Anomalies in Unmanned Vehicles Using the Mahalanobis Distance." ICRA 2010.
9. Wellhausen, L., et al. "Safe Robot Navigation via Multi-Modal Anomaly Detection." RA-L 2020.
10. Fankhauser, P., Bloesch, M., Hutter, M. "Probabilistic Terrain Mapping for Mobile Robots with Uncertain Localization." RA-L 2018.
11. Cai, X., et al. "Probabilistic Traversability Model for Risk-Aware Motion Planning in Off-Road Environments." IROS 2023.
12. Oh, M., et al. "TRIP: Terrain Traversability Mapping With Risk-Aware Prediction." arXiv 2024.
13. Ginting, M.F., et al. "SEEK: Semantic Reasoning for Object Goal Navigation." RSS 2024.
14. Yang, G., et al. "Probabilistic Modeling of Semantic Ambiguity for Scene Graph Generation." CVPR 2021.
15. Wilson, J., et al. "Modeling Uncertainty in 3D Gaussian Splatting through Continuous Semantic Splatting." arXiv 2024.
16. Mattamala, M., et al. "Wild Visual Navigation: Fast Traversability Learning via Pre-Trained Models." Autonomous Robots 2025.
17. Rosenthal, S., Dey, A.K., Veloso, M. "How Robots' Questions Affect the Accuracy of Human Responses." RO-MAN 2009.
18. Deits, R., et al. "Clarifying Commands with Information-Theoretic Human-Robot Dialog." JHRI 2013.
19. Xu, Y., et al. "Seeing with Partial Certainty: Conformal Prediction for Robotic Scene Recognition." arXiv 2025.
20. Wigness, M., et al. "RUGD Dataset." IROS 2019.
21. Shan, T., Wang, J., Englot, B., Doherty, K. "Bayesian Generalized Kernel Inference for Terrain Traversability Mapping." CoRL 2018. PMLR 87:829–838. [Canonical Beta-Bernoulli binary traversability update]
22. Zhou, C., et al. "LIMA: Less Is More for Alignment." NeurIPS 2023. [1,000 samples sufficient for narrow instruction-tuning]
23. Chen, L., et al. "AlpaGasus: Training A Better Alpaca with Fewer Data." ICLR 2024. [Quality over quantity; 9k filtered > 52k raw]
24. Tellex, S., et al. "Understanding Natural Language Commands for Robotic Navigation and Mobile Manipulation." AAAI 2011. [G³: non-LLM slot detection via dependency parse + grounding graph]
25. Thomason, J., et al. "Improving Grounded Natural Language Understanding through Human-Robot Dialog." ICRA 2019 / JAIR 2020. [CCG semantic parser: slot confidence threshold triggers clarification without LLM]
26. Meta AI. "SAM 3: Segment Anything with Concepts." arXiv:2511.16719, November 2025. [SAM3 real speed: 5–6 FPS on H200, too slow for real-time]
27. Oh, M., et al. "TRIP: Terrain Traversability Mapping With Risk-Aware Prediction." arXiv 2024. [T-BGK kernel + Kalman filter with Mahalanobis outlier rejection]
28. Jiang, R., et al. "RELLIS-3D Dataset." ICRA 2021. [20 outdoor terrain classes, lidar + camera]



Implemented Components

1. LLM Consistency Check (system/instruction_uncertainty/consistency_check.py)

The existing ambiguity detector queries an LLM once per instruction and produces a nonconformity score κ_I for conformal prediction. However, LLM outputs are sensitive to prompt context — the same instruction can yield different ambiguity classifications across independent forward passes.
We exploit this instability as a free signal: run the detector 3× with independent LLM calls (fresh in-session caches each time), collect the three predicted ambiguity types, and test for agreement.
If all three runs agree on the same type, κ_I is computed normally: κ_I = w(type) · avg(p_ambiguous), where w is the severity weight and p_ambiguous is averaged across runs.
If any run disagrees (i.e. |{types}| > 1), the instruction is flagged as unstable and κ_I is set to its maximum value of 1.0, unconditionally triggering the clarification request.
This requires no additional training data, no new model, and no change to the conformal prediction framework — it is purely a consistency audit layer on top of the existing detector.
2. Gaussian Process Traversability Map (system/env_uncertainty/gp_traversability.py)

Replaces FM-based terrain scoring (which the mentor identified as unreliable) with a Gaussian Process regression model over traversability scores in normalized pixel space [0,1]².
Kernel is a fixed RBF + WhiteKernel (no optimizer restarts), making inference deterministic and reproducible across runs — a property that LLM logit-based scorers lack.
Each terrain observation adds a point (y_n, x_n, τ) to the GP dataset; the posterior mean μ(p) and variance σ²(p) are updated analytically.
Cold start (zero observations) returns an uninformative uniform prior: μ = 0.5, σ = 0.4.
Trajectory safety is scored using the Lower Confidence Bound: LCB(τ) = min_t [μ_post(p_t) − β·σ_post(p_t)]. A low LCB signals either low expected traversability or high epistemic uncertainty at some point along the path — both are grounds for querying the user.
User language feedback is incorporated via a Bayesian update: τ₁ = p_tp·τ₀ / (p_tp·τ₀ + p_fp·(1−τ₀)) for a "safe" response, with p_tp = 0.95, p_fp = 0.10. The posterior τ₁ is then injected as a new GP observation, enabling sequential Bayesian refinement across turns.
3. Scene Graph with Terrain Memory (system/env_uncertainty/scene_graph.py)

Implements the "object/scene graph" representation recommended by the mentor, replacing the continuous traversability map as the primary memory structure.
The image plane is discretized into a coarse 10×10 grid. Each occupied cell is described by a TerrainNode keyed on (terrain_label, cell_id), storing the GP posterior mean, GP variance, and a user-confirmation flag.
Node certainty follows a three-level progression: UNKNOWN (no GP data) → INFERRED (GP posterior available) → CONFIRMED (user has explicitly responded).
The should_skip_asking() predicate prevents redundant clarification requests: if a cell is user-confirmed and its GP posterior variance σ² < 0.04 (i.e. σ < 0.2, indicating the model is confident), the robot does not re-query the user for that terrain region in subsequent frames.
update_from_gp() propagates GP posterior values into all nodes after each new observation, keeping the scene graph and the GP model synchronized. Trajectory names are also recorded on nodes whose grid cells the trajectory passes through, enabling later reasoning about which paths are affected by uncertain terrain.