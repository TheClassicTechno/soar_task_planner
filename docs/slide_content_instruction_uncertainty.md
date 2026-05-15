# Slide Content: Instruction Uncertainty — Detection, Correlation, and Innovations

> This document fills in the blank sections from the May 4 slide deck
> (`may4fillindiagramplease.txt`). Copy these directly into the slides.

---

## How to Handle Instruction Uncertainty

### Detection

**General:** Uncertainty arises when the user's command does not uniquely
determine which robot trajectory is correct. The robot cannot determine
what to do next.

**We need to — Recognize what the robot doesn't know:**

- **Method:** LLM-based semantic slot-filling parser.
  The instruction $\ell$ and observed scene description $s$ are passed to
  a foundation model (Claude claude-sonnet-4-6 or Gemini 1.5) with a
  structured prompt requesting JSON output:
  ```json
  {
    "ambiguity_type": "missing_object" | "missing_direction" | "missing_distance" |
                      "missing_action" | "ambiguous_target" | "ambiguous_action" | "none",
    "ambiguity_score": <float 0.0–1.0>,
    "missing_slot": "object" | "direction" | "distance" | "action" | null
  }
  ```
- **Score:** $u_I = w_{\text{type}}(t^*) \cdot P(\text{ambiguous} \mid \ell, s)$
  where $P(\text{ambiguous})$ is the LLM's classification confidence and
  $w_{\text{type}}$ is a severity weight (1.0 for missing\_action;
  0.75 for missing\_object, ambiguous\_target;
  0.50 for ambiguous\_action, missing\_direction;
  0.25 for missing\_distance).
- **Fine-tune option:** LoRA on Gemma-3 9B using instruction ambiguity
  classification examples from the WhenToAsk/KnowNo navigation datasets
  (multi-label: ambiguity type + severity score as targets).

**We need to — Correlate that to robot actions:**

- Map each ambiguity type to a **constraint violation count** across the
  $K = 3$ candidate trajectories (forward, left arc, right arc):
  - *missing_direction*: all 3 trajectories are equally valid → must ask
  - *missing_object*: all trajectories leading to unidentified object are invalid → must ask
  - *ambiguous_target*: trajectories to each candidate object are all plausible → must ask
  - *missing_distance*: trajectory length is unconstrained → ask for distance
- If constraint violations > 0 **or** $u_I > \theta_{\text{ask}}$ (calibrated
  threshold): enter ASK state and generate clarification question.
- $\theta_{\text{ask}}$ is calibrated via conformal prediction on
  `nav_calibration.json` Type-1 entries to achieve 1−ε coverage.

---

## Innovations — Detection

### Recognize the areas/properties the robot doesn't know

**Technical innovation:**
- LLM semantic slot-filling with 6-class ambiguity taxonomy
  (missing object, missing action, missing direction, missing distance,
  ambiguous target, ambiguous action)
- Continuous ambiguity score $u_I \in [0,1]$ instead of binary ask/no-ask
- Score formula: $u_I = w_{\text{type}} \cdot P(\text{ambiguous} \mid \ell, s)$
  — weight × LLM confidence

**Theoretical innovation:**
- Joint conformal prediction framework: joint non-conformity score
  $\kappa^{\text{joint}} = \max(\kappa^I, \kappa^E)$ provides statistical
  coverage guarantee over both uncertainty types simultaneously
- Coverage guarantee: $P(y_{\text{test}} \in C(\tilde{x}_{\text{test}})) \geq 1 - \varepsilon$

### Correlate that to robot actions

**Technical innovation:**
- Ambiguity type → trajectory constraint violation mapping
  (not all ambiguities affect all trajectories equally)
- Prediction set size $|C(\tilde{x})|$ is directly proportional to the
  number of viable trajectory options that remain after applying constraints
- If any trajectory is uniquely determined: act directly (option A)
- If multiple trajectories remain plausible: ask (option B)

**Model:** existing `ConformalPredictor` extended with the joint
non-conformity score from `theoretical_innovations.md` §A.2

### Turn these to language context for questions

**Technical innovation (Innovation 4 — User Personalization):**
- `UserProfile` abstraction: verbosity × expertise × preferred_format
- `PersonalizedTemplateBank`: multiple question variants per situation key,
  indexed by (situation_key, verbosity)
- LLM mode: profile is injected into prompt via `describe_profile_for_prompt()`
  so the model generates style-matched output

**Innovation examples:**
- Terse + question format: "No referent for 'it'. What is it?"
- Standard: "Your instruction refers to 'it,' but I don't see a clear
  referent in the scene. What object should I navigate to?"
- Verbose + expert: "Instruction parse failure: pronoun 'it' has no prior
  referent in scene context (3 visible objects: bench, tree, sign).
  Ambiguity score: 0.92. Please specify the target object."
- Novice + option_list: "I'm not sure what 'it' means. Which one should
  I go to? (1) The bench, (2) The tree, (3) The sign"

---

## Innovations — Iterative Update

### Convert the generated language text to questions

**Technical innovation:**
- `QuestionGenerator.generate(result, trajectories, user_profile, scenario_context)`
  — unified interface for both uncertainty types
- Profile-aware template selection in template mode (terse/standard/verbose ×
  question/statement/option_list = 9 output styles per situation)
- Scenario context injection: "construction zone", "night operation",
  "pedestrian area" → LLM generates context-appropriate phrasing
- Instruction uncertainty question template examples:
  - Missing object: "Your instruction mentions [slot] but I don't see a clear
    referent. Can you point to [slot], or describe it more specifically?"
  - Missing direction: "Should I turn left or right here?"
  - Ambiguous target: "I see [N] possible [object]s. Which one do you mean?"

**Model:** Claude claude-sonnet-4-6 via `LLMInterface.predict_json()` in
LLM mode; template bank in template mode.

### Update the existing understanding

**Technical innovation (Innovation 3 — Bayesian Map Update):**
- Bayesian posterior update replaces hard-coded 0.9/0.0:

$$\tau_1 = \frac{L_r(\text{trav}=1) \cdot \tau_0}{L_r(\text{trav}=1) \cdot \tau_0 + L_r(\text{trav}=0) \cdot (1-\tau_0)}$$

- Likelihood parameters: $P(R=\text{safe} \mid \text{trav}=1) = 0.95$,
  $P(R=\text{safe} \mid \text{trav}=0) = 0.10$
- Sequential update: posterior from turn $k$ becomes prior for turn $k+1$
  (robot gets better at recognizing terrain it has seen before)

**For instruction uncertainty:**
- Confirmed intent stored in memory dict keyed by `(instruction_pattern, scene_hash)`
- On next encounter of the same instruction type in a similar scene,
  the remembered preference is applied without re-asking

### Move the robot accordingly

**Technical innovation:**
- After user response, re-run trajectory selection with updated knowledge:
  - Instruction confirmed → `correct_option` resolved to A → proceed on
    the trajectory that satisfies the user's clarified intent
  - Terrain feedback applied → traversability map updated via Bayesian rule
    → re-score all candidate trajectories → select highest minimum-traversability path
- Both uncertainty types converge at this step: the unified 6-step pipeline
  returns to the default navigation state after any clarification

**Foundation Model Traversability Scoring (Innovation 3 — Technical):**
- `FMTraversabilityScorer` queries Gemini/Claude for per-class traversability
  instead of using a static table
- In-session cache keyed by `(label, context_hash)` avoids redundant calls
- Latency budget fallback: if FM call exceeds 500 ms, static table is used
- Evaluated via MAE vs. hand-annotated RUGD/RELLIS ground-truth table

---

## Evaluation of the Methodology

### Detection

**Recognize the areas/properties the robot doesn't know:**
- Instruction uncertainty: ambiguity type classification accuracy
  (6-class precision/recall on Type-1 test entries)
- Environmental uncertainty: URDR = true positive rate for detecting
  unknown regions in nav_env_test.json

**Correlate that to robot actions:**
- Prediction set coverage: NCR = fraction where correct option NOT in
  prediction set (lower is better)
- ESR = fraction where prediction set exactly equals {correct option}
  (higher = less unnecessary asking)

**Turn these to language context for questions:**
- QRS = human evaluation (1–5) of generated question relevance and clarity
- Separately rated for terse / standard / verbose profile outputs

### Iteratively update the uncertain areas and move the robot

**Convert generated language to questions:**
- Cross-validation with LLM judge: GPT-4o asked to rate question quality
  on 1–5 scale across all profile variants

**Update existing understanding:**
- MUA = fraction of traversability updates correctly applied
  (Bayesian update vs. hand-labeled expected posterior)
- Sequential update accuracy: over 3-round interactions, does traversability
  score converge toward correct value?

**Move the robot accordingly:**
- SR = overall success rate (robot makes correct ask/act decision)
- HR = human help rate (fraction of scenarios where robot asked)
- FPR = false positive rate (asked when correct_option ∈ {A})

---

## Summary Table: Innovation Coverage

| Pipeline Step | Innovation | Type | Method |
|---|---|---|---|
| Detect (instruction) | Ambiguity score $u_I$ | Technical + Theoretical | LLM slot-fill + severity weight |
| Detect (environment) | SAM3+SAM2 spatial subtraction | Technical | Overlap threshold algorithm |
| Recognize | 6-class ambiguity taxonomy | Task-level | Unified pipeline for both types |
| Correlate | Joint CP non-conformity score $\kappa^{\text{joint}}$ | Theoretical | $\max(\kappa^I, \kappa^E)$ |
| Convert to language | User-personalized question gen | Technical | UserProfile × template/LLM |
| Update understanding | Bayesian traversability update | Theoretical | Posterior update with noise model |
| Execute | FM traversability scoring | Technical | Foundation model judge + cache |
