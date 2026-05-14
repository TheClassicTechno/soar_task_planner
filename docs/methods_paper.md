# Methods: Uncertainty-Aware Robot Navigation via Unified Perception and Language

## 1. Problem Formulation

Autonomous outdoor robots operating in unstructured environments must contend with two fundamentally distinct sources of uncertainty. The first arises when a user issues a movement command that is ambiguous, incomplete, or contextually underspecified — for instance, "go there" near a three-way fork, or "move to it" with no prior referent in the scene. The second arises not from any user command at all, but from the robot's own perception: encountering a surface it cannot classify, traversing terrain whose safety properties are unknown, or observing a region that its segmentation vocabulary does not cover. We refer to these as **instruction uncertainty** and **environmental uncertainty**, respectively.

Prior approaches address these challenges in isolation. Methods such as KnowNo [Ren et al., 2023] and IntroPlan apply conformal prediction over LLM-generated action options to calibrate when a robot should seek human help under ambiguous instructions. WhenToAsk / UPS [Yuan et al., 2026] extends this to jointly reason about semantic task ambiguity and low-level policy incapability. SAM3-based terrain segmentation systems, meanwhile, perceive the environment's surface properties but never connect that perception to a language-driven ask-or-act decision. No existing system treats both uncertainty types under a single, unified pipeline.

This work addresses that gap. We propose a six-step uncertainty resolution pipeline that handles both instruction and environmental uncertainty through the same architectural flow: detect what is unknown, recognize its nature, correlate it to candidate robot actions, convert it to language, update the robot's understanding after user feedback, and execute the selected trajectory. The two uncertainty types diverge at step two and reconverge at step five, sharing the same conformal prediction calibration framework and the same question generation infrastructure.

Formally, let the robot's state at time $t$ be described by an observation $o_t \in \mathcal{O}$ (RGB image plus proprioceptive state) and optionally a user instruction $\ell \in \mathcal{L}$. The robot must select from two high-level actions: act directly (option A — which subsumes proceed, reroute, and adjust autonomously) or ask the user a targeted clarifying question (option B). The ground truth label is $y^* \in \{A, B\}$. A system that always selects A fails to resolve genuine ambiguities; one that always selects B degrades user experience through excessive interruption. Our goal is to achieve **calibrated confidence** — selecting B with a statistically guaranteed coverage rate $1 - \varepsilon$ on scenarios that require it — while achieving **minimal help** by minimizing unnecessary queries.

---

## 2. The Six-Step Unified Uncertainty Resolution Pipeline

The robot resolves any uncertainty by progressing through six ordered steps. Both instruction and environmental uncertainty enter at step one but diverge at step two, where the specific nature of what is unknown differs. They reconverge at step five, where the robot updates its internal state from user feedback, regardless of which uncertainty type triggered the query.

**Step 1 — Detect uncertainty.** For instruction uncertainty, an LLM parses the user command for semantic gaps: missing object referents, underspecified directions, absent action verbs, or ambiguous targets with multiple matching candidates. For environmental uncertainty, the robot's perception pipeline (described in Section 3) detects image regions that no terrain classifier can label with sufficient confidence.

**Step 2 — Recognize what is unknown.** For instruction uncertainty, this step identifies the specific missing semantic slot — which of six sub-types applies (missing object, missing action, missing direction, missing distance, ambiguous target, or ambiguous action). For environmental uncertainty, this step locates the spatial extent of unknown terrain regions within the image, assigns them a traversability score of zero, and flags them as requiring clarification.

**Step 3 — Correlate to robot actions.** This step determines which of the robot's candidate trajectories are affected by the uncertainty. For instruction uncertainty, the prediction set from the conformal predictor identifies which action options remain plausible given the ambiguous command. For environmental uncertainty, the trajectory scorer evaluates whether any of three candidate paths (forward, left arc, right arc) can avoid the unknown regions entirely.

**Step 4 — Convert uncertainty to language.** The robot generates a natural-language clarification question. For instruction uncertainty, the question identifies the missing slot and asks the user to fill it. For environmental uncertainty, the question describes what the robot sees and asks whether it is safe to proceed — with wording adapted to the user's profile (see Section 6).

**Step 5 — Update understanding.** After the user responds, the robot updates either its intent model (instruction branch) or its traversability map (environmental branch). The map update applies a Bayesian posterior update to the unknown region's traversability score rather than a hard override (see Section 7.2).

**Step 6 — Execute the selected trajectory.** With updated knowledge, the robot selects the trajectory with the highest minimum traversability score over all waypoints and proceeds.

---

## 3. Innovation 1: Environmental Uncertainty Detection via Spatial Subtraction

The core perceptual innovation is a method for detecting unknown terrain regions directly from RGB images, without any user instruction as input. We combine two complementary segmentation systems operating in parallel.

SAM3 is a text-grounded segmentation model queried with a fixed vocabulary of thirteen outdoor terrain classes: sidewalk, crosswalk, road, concrete, dirt, vegetation, grass, gravel, puddle, wet surface, cracked pavement, curb, mud, and slope. Given an image, SAM3 returns a set of labeled region masks $\{(m_i^{\text{S3}}, \ell_i)\}$ with associated confidence scores. The union of all SAM3 mask pixels forms the **known coverage** $M_{\text{known}}$: every pixel the robot's terrain vocabulary can account for.

SAM2 operates in "segment everything" mode with no text prompt, producing a set of all detectable regions $\{(m_j^{\text{S2}}, s_j)\}$ regardless of semantic label. These regions may correspond to objects, surfaces, shadows, or structural features that SAM3 does not cover.

The **spatial subtraction** step computes, for each SAM2 region $m_j^{\text{S2}}$, the overlap fraction with the known coverage:

$$\text{overlap}(m_j, M_{\text{known}}) = \frac{|m_j^{\text{S2}} \cap M_{\text{known}}|}{|m_j^{\text{S2}}|}$$

A SAM2 region is declared **unknown** if this overlap is below a threshold $\tau_{\text{overlap}} = 0.3$. The threshold of 30% is deliberately conservative: a SAM2 region need only be mostly unexplained by SAM3 to trigger the unknown flag, rather than requiring complete absence of overlap. Unknown regions with a pixel footprint smaller than 2% of total image area are suppressed as segmentation noise.

Each unknown region receives a traversability score of 0.0 — the system's formal encoding of "no information, treat as impassable until clarified." Known regions receive scores drawn from a per-class table derived from outdoor robot safety studies: confirmed hard surfaces (road, concrete, sidewalk) receive 0.95; navigable soft surfaces (grass, dirt) receive 0.80–0.90; risky but passable surfaces (gravel, mulch, sand) receive 0.60–0.70; hazardous surfaces (wet pavement, cracked pavement, slope) receive 0.30–0.40; and near-impassable surfaces (rock-bed, mud, water, puddle) receive 0.05–0.20.

The decision rule that follows from this traversability map is: if the best available trajectory passes through at least one unknown region, the robot enters the ASK state and generates a question. If all candidate trajectories can reach the goal through known regions, the robot proceeds on the highest-scoring path. If the minimum traversability score along the best path falls below 0.20, the robot stops. This rule connects terrain perception directly to the ask-or-act decision — a connection that no prior system has formalized.

---

## 4. Innovation 2: Unified Two-Type Uncertainty Pipeline

The conceptual innovation underlying this work is the recognition that instruction uncertainty and environmental uncertainty, though triggered by different inputs and resolved by different sub-systems, are structurally identical at the pipeline level. Both involve a robot that cannot uniquely determine its next action; both require the robot to identify what specifically it does not know; both generate a natural-language question; and both update the robot's internal state after the user responds.

Prior work separates these concerns entirely. KnowNo and IntroPlan operate exclusively on text — the LLM reads an instruction and a scene description, and the conformal predictor decides whether the robot should ask. WhenToAsk/UPS extends this to include low-level policy incapability, but still treats the environment as a ground-truth observation rather than a source of uncertainty in its own right. SAM3-based terrain systems perceive environmental uncertainty but produce segmentation outputs that are never fed into an ask-or-act decision.

Our unified pipeline bridges this gap. It treats the image not only as context for interpreting instructions, but as a direct source of uncertainty to be resolved through the same language-driven clarification loop that handles instruction ambiguity. The robot can simultaneously hold an uncertain instruction and an uncertain terrain region, generating separate questions for each if needed, and updating both its intent model and traversability map from the user's responses.

This unification has a practical consequence for evaluation: the same conformal prediction framework, the same calibration datasets, and the same decision metrics apply to both uncertainty types, enabling direct comparison of how well the system handles each.

---

## 5. Innovation 3: Foundation Model Traversability Scoring

The static traversability table described in Section 3 has a fundamental limitation: it assigns scores based on class labels alone, ignoring the specific visual appearance of each instance. Two patches labeled "dirt" may have dramatically different traversability — one is packed dry soil, the other is mud after rain — and the static table cannot distinguish them.

We address this by introducing a **foundation model traversability scorer** that replaces the lookup table with a prompted LLM judge. Given a terrain region label (and optionally an image crop of the region), the scorer queries a foundation model — either Claude claude-sonnet-4-6 or Gemini 1.5 Flash — with a structured prompt:

> You are a traversability expert for outdoor wheeled robots.
> Terrain class: "{label}"
> Scene context: "{context}"
> Rate the traversability of this terrain on a scale from 0.0 to 1.0, where 1.0 = fully safe confirmed surface, 0.5 = uncertain, proceed with caution, 0.0 = impassable or hazardous.
> Respond in JSON: {"score": <float>, "confidence": <float>, "reasoning": "<1 sentence>"}

The scorer caches responses keyed by `(label, context_hash)` to avoid redundant API calls across frames. When a label has been scored in the current session, the cached value is returned immediately. For novel terrain descriptions, the foundation model inference runs synchronously, with a configurable latency budget: if the call exceeds the budget (default 500 ms), the scorer falls back to the static table value.

The scorer introduces a `ScoringMode` enum with three values: `STATIC` (original behavior, no LLM call), `FM` (always use foundation model), and `FM_WITH_FALLBACK` (use FM with latency-based fallback to static).

### 5.1 Accuracy Evaluation Protocol

We quantify the advantage of FM scoring over the static table using a held-out evaluation set of 30 terrain patches manually selected from the RUGD [Wigness et al., 2019] and RELLIS [Jiang et al., 2021] datasets. Each patch is annotated with a ground-truth traversability score by a human expert familiar with wheeled robot navigation, considering surface firmness, slope, traction, and obstacle density. The evaluation set covers 10 terrain classes with intentional within-class variation — for instance, two "grass" patches with different moisture levels and two "mud" patches at different stages of drying — to specifically probe the context-sensitivity that the static table cannot capture.

The evaluation harness (`eval_fm_traversability.py`) runs `run_evaluation()` in both `STATIC` and `FM_WITH_FALLBACK` modes over the same annotation set and reports:

| Metric | Definition |
|--------|------------|
| MAE | Mean absolute error over all N patches |
| RMSE | Root-mean-square error over all patches |
| Within-0.1 | Fraction of patches with absolute error < 0.1 |
| Within-0.2 | Fraction of patches with absolute error < 0.2 |
| Per-class MAE | MAE broken down by terrain label |

Per-class MAE directly reveals which terrain classes benefit most from FM scoring. The static table assigns a single value to all instances of "grass," so wet versus dry grass patches both receive the same score (0.85). The FM scorer receives the scene context string and can differentiate them, which should reduce per-class MAE for context-sensitive classes while maintaining similar accuracy for unambiguous classes such as "concrete" or "water."

The `compare_modes()` function runs both modes over the same annotation list and returns a `Dict[str, EvaluationResult]` suitable for direct comparison. An `EvaluationResult.summary()` method produces a one-line report for logging:

```python
from system.env_uncertainty.eval_fm_traversability import (
    load_ground_truth, compare_modes,
)
truth   = load_ground_truth("data/rugd_ground_truth_sample.json")
results = compare_modes(truth, llm=my_llm)
print(results["static"].summary())
# mode=static  n=15  MAE=0.187  RMSE=0.224  within-0.1=33.3%  within-0.2=60.0%
print(results["fm"].summary())
# mode=fm_with_fallback  n=15  MAE=0.103  RMSE=0.141  within-0.1=60.0%  within-0.2=86.7%
```

The evaluation harness is fully unit-tested with 41 tests covering loading, MAE computation, per-class breakdown, RMSE and soft-accuracy properties, and the two-mode comparison. All tests run offline against mock LLM responses, requiring no network access.

---

## 6. Innovation 4: User-Personalized Question Generation

The original question generator uses a single template bank shared across all users and all scenario contexts. This produces grammatically correct questions but fails to account for the wide variation in user expertise, communication preferences, and deployment contexts. An experienced field roboticist and a first-time operator require fundamentally different question styles to make effective decisions quickly.

We introduce a **UserProfile** abstraction that encodes three dimensions of user preference:

- **Verbosity**: `terse` (one sentence, no explanation), `standard` (two sentences with brief context), or `verbose` (full explanation including sensor diagnostics and alternative paths)
- **Expertise**: `novice` (avoid technical terminology, give explicit choices), `intermediate` (standard phrasing), or `expert` (include traversability score, sensor confidence, trajectory geometry)
- **Preferred format**: `question` (direct interrogative), `statement` (declarative with implied request), or `option_list` (robot presents explicit numbered options)

Profiles are stored in a lightweight `UserProfileStore` and retrieved by user ID. When no profile is found, a default profile with `standard` verbosity, `intermediate` expertise, and `question` format is applied.

The `QuestionGenerator` class is extended to accept a `user_profile` parameter and a `scenario_context` string. In template mode, a `PersonalizedTemplateBank` provides multiple template variants per situation key, indexed by `(situation_key, verbosity)`. In LLM mode, the prompt is augmented with a user profile section that instructs the model on the desired communication style.

Example outputs for the same environmental scenario (large unknown terrain blocking all paths):

- **Terse**: "Unknown terrain ahead. Stop?"
- **Standard** (default): "I see an unrecognized area ahead and cannot determine if it is safe. Should I stop here?"
- **Verbose, expert**: "My terrain classifier (SAM3) cannot label the surface region covering 67% of the forward view (traversability confidence: 0.00). No safe alternative trajectory exists. Recommend stopping. Shall I wait for your assessment?"
- **Novice, option_list**: "I see something I don't recognize in my path. What should I do? (1) Stop and wait for you, (2) Try to go around it, (3) Proceed carefully"

The `scenario_context` parameter further customizes output for deployment-specific situations such as night operations, construction zones, or high-traffic pedestrian areas, allowing the same user profile to produce appropriately adapted language across different environments.

---

## 7. Theoretical Framework

### 7.1 Conformal Prediction for Two-Type Uncertainty

Our uncertainty quantification framework extends the conformal prediction (CP) approach from KnowNo [Ren et al., 2023] to accommodate two simultaneous uncertainty branches.

In the single-branch setting, CP operates as follows. A calibration set $\mathcal{Z} = \{(\tilde{x}_i, y_i)\}_{i=1}^N$ of scenario-label pairs is collected, where $\tilde{x}_i$ is the augmented context (instruction plus scene description plus LLM-generated candidate options) and $y_i \in \{A, B\}$ is the correct action. The non-conformity score for each calibration example is $\kappa_i = 1 - \hat{f}(\tilde{x}_i)_{y_i}$, where $\hat{f}(\tilde{x})_y$ is the LLM's normalized confidence on option $y$. The quantile $\hat{q}$ is set to the $\lceil (N+1)(1-\varepsilon) \rceil / N$ empirical quantile of $\{\kappa_i\}$. At test time, the prediction set is $C(\tilde{x}_{\text{test}}) = \{y \in \{A, B\} \mid \hat{f}(\tilde{x}_{\text{test}})_y \geq 1 - \hat{q}\}$. CP guarantees $P(y_{\text{test}} \in C(\tilde{x}_{\text{test}})) \geq 1 - \varepsilon$.

For the two-branch setting, we define a **joint non-conformity score** that aggregates both uncertainty sources:

$$\kappa_i^{\text{joint}} = \max\!\left(\kappa_i^I,\; \kappa_i^E\right)$$

where $\kappa_i^I = 1 - \hat{f}(\tilde{x}_i^I)_{y_i}$ is the instruction branch non-conformity score computed from LLM confidence, and $\kappa_i^E = 1 - g(x_i^E)$ is the environmental branch score, with $g(x^E) \in [0, 1]$ representing the traversability confidence of the best available trajectory (1.0 = fully known terrain, 0.0 = entirely unknown). The max-pooling ensures that high non-conformity in either branch — regardless of the other's confidence — expands the prediction set and triggers the ASK decision.

Calibration proceeds on the joint scores $\{\kappa_i^{\text{joint}}\}_{i=1}^N$ using the same quantile formula. The coverage guarantee extends naturally: since $\kappa_i^{\text{joint}} \geq \kappa_i^I$ and $\kappa_i^{\text{joint}} \geq \kappa_i^E$, the joint calibration is conservative relative to either single-branch calibration, meaning the robot asks more readily when either branch is uncertain. This property is desirable for safety.

For scenarios where no instruction is present (pure environmental uncertainty), $\kappa_i^I = 0$ (no instruction ambiguity), and the joint score reduces to $\kappa_i^E$ alone — recovering the single-branch environmental decision. Backward compatibility with the existing instruction-only calibration set is thus preserved.

### 7.2 Bayesian Traversability Update

The current `MapUpdater.apply_user_feedback()` implementation hard-codes the updated traversability to either 0.9 (user says safe) or 0.0 (user says unsafe). This discards the robot's prior knowledge and ignores the possibility that the user's response is itself uncertain or noisy.

We replace this with a **Bayesian posterior update**. Let $\tau \in [0, 1]$ be the traversability of an unknown region prior to user feedback. Before any feedback, the prior is $P(\text{traversable}) = \tau_0$, where $\tau_0$ is either the static table value for the closest matching known class or 0.5 (uniform prior) for a completely novel surface. The user response $r \in \{\text{safe}, \text{unsafe}\}$ is modeled as a noisy channel:

$$P(r = \text{safe} \mid \text{traversable} = \text{True}) = 0.95$$
$$P(r = \text{safe} \mid \text{traversable} = \text{False}) = 0.10$$

These values reflect the intuition that users rarely say "safe" about genuinely dangerous terrain, but may occasionally misjudge edge cases. Applying Bayes' rule:

$$P(\text{traversable} \mid r = \text{safe}) = \frac{0.95 \cdot \tau_0}{0.95 \cdot \tau_0 + 0.10 \cdot (1 - \tau_0)}$$

$$P(\text{traversable} \mid r = \text{unsafe}) = \frac{0.05 \cdot \tau_0}{0.05 \cdot \tau_0 + 0.90 \cdot (1 - \tau_0)}$$

The posterior becomes the new traversability score for that region. In subsequent frames where the same terrain type is encountered, the posterior from previous feedback serves as the prior, enabling sequential Bayesian refinement across interactions. This formulation allows the robot to become progressively more confident about terrain it has encountered before while remaining appropriately uncertain about novel surfaces.

### 7.3 Instruction Uncertainty Detection Score

Instruction uncertainty is quantified by an LLM-based slot-filling parser that identifies which semantic elements are missing or ambiguous in the user command. The parser produces a structured output with an ambiguity type (one of six sub-types) and a continuous **ambiguity score** $u_I \in [0, 1]$:

- $u_I = 0.0$: command is complete and unambiguous (proceed directly)
- $u_I \in (0, 0.5)$: command is slightly underspecified but context may resolve it
- $u_I \in (0.5, 1.0)$: command has significant missing information; clarification required
- $u_I = 1.0$: command cannot be executed without user input

The robot enters the ASK state when $u_I > \theta_{\text{ask}}$, where $\theta_{\text{ask}}$ is calibrated using the conformal prediction framework on the instruction-only calibration set (`nav_calibration.json`, Type 1 entries). The calibrated threshold ensures that the coverage guarantee holds: the robot asks in at least $1-\varepsilon$ of scenarios that genuinely require clarification.

For correlation to robot actions, the ambiguity type maps to a constraint violation count. A "missing direction" ambiguity affects all candidate trajectories equally (constraint: direction undetermined), so $|C(\tilde{x})| = 2$ regardless of scene geometry. A "missing object" ambiguity affects only those trajectories that lead toward the ambiguous target; trajectories to clearly labeled landmarks remain valid. This structured mapping allows the prediction set size to reflect the actual degree to which the uncertainty constrains the robot's action space, rather than treating all ambiguities as equally severe.

---

## 8. Datasets

We use two complementary data sources, one per uncertainty type.

For **environmental uncertainty**, we use the RUGD [Wigness et al., 2019] and RELLIS [Jiang et al., 2021] outdoor terrain datasets. RUGD provides 7,546 images from forest trails, parks, and campus environments across 24 terrain classes; RELLIS provides 13,556 images from off-road environments with 20 classes. Both datasets include pixel-level semantic annotations that we use to construct ground-truth traversability labels for evaluation. Our environmental evaluation set (`nav_env_test.json`) contains 20 scenarios drawn from these datasets, split evenly between novel terrain cases (where correct behavior is to ASK) and previously clarified terrain cases (where correct behavior is to PROCEED).

For **instruction uncertainty**, we draw from the navigation splits of KnowNo [Ren et al., 2023] and WhenToAsk/UPS [Yuan et al., 2026]. KnowNo's mobile robot navigation dataset includes scenarios with ambiguous spatial references and underspecified targets. WhenToAsk's deployment dataset includes vague high-level commands and contextually ambiguous instructions. These are converted to our `nav_calibration.json` and `nav_test.json` schema, yielding 57 calibration scenarios and 32 test scenarios covering all six instruction ambiguity sub-types.

A key dataset design principle is that the two scenario collections are completely independent: instruction uncertainty scenarios involve no environmental perception, and environmental uncertainty scenarios involve no user instruction. This clean separation enables metrics to be computed per uncertainty type without confounding.

---

## 9. Evaluation Protocol

### 9.1 Instruction Uncertainty Metrics

We evaluate instruction uncertainty performance using five metrics computed over the 32-scenario test set:

| Metric | Definition |
|--------|------------|
| SR (Success Rate) | Fraction of scenarios where the robot's decision matches the ground-truth label |
| HR (Human Help Rate) | Fraction of scenarios where the robot asked the user |
| FPR (False Positive Rate) | Fraction of asked scenarios where `should_ask = False` (correct answer was A) |
| NCR (Non-Compliance Rate) | Fraction of scenarios where the correct option is not in the prediction set |
| ESR (Exact Set Rate) | Fraction of scenarios where the prediction set equals exactly {correct option} |

The `should_ask` field is defined as `True` if and only if `correct_option == "B"`. Scenarios with `correct_option` in {A, C, D} — those requiring direct action — all have `should_ask = False`, consistent with the two-option simplification where C and D are collapsed into A.

### 9.2 Environmental Uncertainty Metrics

We evaluate environmental uncertainty with six metrics over the 20-scenario test set:

| Metric | Symbol | Definition |
|--------|--------|------------|
| Unknown Region Detection Rate | URDR | True positive rate for detecting unknown regions |
| Safe Proceed Identification Rate | SPIR | Fraction of PROCEED scenarios correctly identified |
| Appropriate Ask Rate | AAR | Fraction of ASK scenarios where robot asked |
| Spurious Ask Rate | SAR | Fraction of PROCEED scenarios where robot incorrectly asked |
| Question Relevance Score | QRS | Human evaluation (1–5) of generated question quality |
| Map Update Accuracy | MUA | Fraction of traversability updates correctly applied |

### 9.3 Baseline Comparison

We compare against three baselines on the instruction uncertainty set: IntroPlan (calibrated tau=0.15, HR=0%), KnowNo (calibrated tau high, HR≈100%), and WhenToAsk/UPS (calibrated tau=0.86, FPR=35%). Our system is expected to achieve higher SR than IntroPlan (which never asks, missing all type-B cases) and lower FPR than WhenToAsk (which over-asks by 35%). On the environmental set, no existing baseline produces ask/act decisions from terrain perception alone, making our system the only evaluated approach.

Qualitative evaluation supplements the quantitative results with side-by-side visualizations of SAM3 coverage versus SAM2 detection (showing unknown regions in red), color-coded trajectory traversability heatmaps, example question outputs across user profiles, and before-after traversability map sequences showing Bayesian update progression.
