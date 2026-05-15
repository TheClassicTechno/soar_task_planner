# Solidified Pipeline, Innovations, and Evaluation Plan

## Accurate Implementation Status (as of May 2026)

### Already Implemented in Code — No Action Needed

| Component | File | What's There |
|-----------|------|-------------|
| FM traversability scorer | `system/env_uncertainty/fm_traversability.py` | ScoringMode enum, system context prompt, JSON score prompt, MD5 cache keyed on (label, context_hash), 500ms latency fallback, score_label / score_region / score_batch |
| Personalization templates | `system/env_uncertainty/question_generator.py` | terse/standard/verbose × 4 situations (has_alternative, no_alternative, large_unknown, multiple_unknowns) + option_list format; LLM mode with full profile injection |
| UserProfile abstraction | `system/env_uncertainty/user_profile.py` | verbosity × expertise × format dataclass, store, describe_profile_for_prompt() |
| Bayesian map updater | `system/env_uncertainty/map_updater.py` | apply_feedback() with Bayesian posterior, _parse_user_response() keyword matching, apply_feedback_to_region() |
| Environmental test data | `baselines/introplan/data/nav_env_test.json` | 20 scenarios in 4 subtypes: A (partial unknown on path), B (unknown off-path → PROCEED), C (mostly unknown scene), D (unknown with route alternative) |
| Instruction test data | `baselines/introplan/data/nav_test.json` + `nav_calibration.json` | 30 test + 57 calibration instruction scenarios |

### Newly Implemented (this session)

| Component | File | What's There |
|-----------|------|-------------|
| Intent memory | `system/instruction_uncertainty/intent_memory.py` | IntentMemory with Bayesian update, should_skip_asking(), recall(), purge_stale(), context hashing; AMBIGUITY_TYPES registry; ambiguity_score() u_I formula |
| Intent memory tests | `system/instruction_uncertainty/tests/test_intent_memory.py` | 48 tests covering math accuracy, update/confirm/contradict cycles, staleness, threshold behavior |

### Still Needs Human Action

| Task | Effort |
|------|--------|
| Annotate 30 RUGD patches with traversability scores [0,1] | ~2 hours |
| Run QRS user study (3 raters × 30 questions) | ~1 day |
| Decide: Introspective Planning baseline — implement or confirm skip | ~1 hour decision |

### Not Yet in Code (Future Work)

| Component | Notes |
|-----------|-------|
| LLM slot-fill parser for instruction uncertainty | Full module still needed; IntentMemory assumes slots are already filled |
| Constraint violation map | Needs candidate trajectory × constraint matrix implementation |
| Joint conformal predictor runner | Runner for the combined κ_I / κ_E pipeline |

---

## What This Doc Fills In (Slide Content)
- Evaluation plan that covers every innovation
- Ablation structure

---

## The 4 Innovations vs. Prior Work

### What Prior Work Misses

| Prior Work | Instruction Uncert. | Env. Uncert. | Personalization | Map Update |
|------------|:---:|:---:|:---:|:---:|
| KnowNo (Ren et al., NeurIPS 2023) | CP-binary | ❌ | ❌ | ❌ |
| Introspective Planning (Liang et al., NeurIPS 2024) | CP+retrieval | ❌ | ❌ | ❌ |
| WhenToAsk/UPS (2026) | action-level | ❌ | ❌ | ❌ |
| GA-Nav | ❌ | traversability only | ❌ | ❌ |
| **Ours** | 6-class CP | SAM3+SAM2+FM | 9 variants | Bayesian |

**Core novel claim:** We are the first to treat environmental uncertainty as a first-class perception problem (not an instruction problem) and couple it with instruction uncertainty under a single conformal guarantee, while personalizing clarification to the user profile.

---

## Slide Content: Environmental Uncertainty

### How to Handle — Detection

**Recognize the areas/properties the robot doesn't know:**
- SAM3 (13 terrain classes, e.g., road, grass, mud, gravel) runs on each camera frame → semantic coverage map
- SAM2 (segment-everything, class-agnostic) runs on same frame → finds all image segments
- Spatial subtraction: UNKNOWN = {region r ∈ SAM2 : overlap(r, SAM3_pred) < τ_overlap = 0.3}
  - Overlap ratio = |r ∩ SAM3_pred| / |r|; threshold τ=0.3 calibrated on RUGD val set
  - This tells the robot exactly what areas SAM3's semantic vocabulary cannot explain
- FM Traversability Scorer (FM-TS): assigns score trav(p) ∈ [0,1] to each patch p
  - Model: GPT-4o-mini (vision) or Gemini 1.5 Flash — fast, cheap per call
  - Prompt template: "Given this outdoor patch image and terrain context '{context}', rate traversability 0 (impassable) to 1 (clear path). Known terrain around it: {neighbor_labels}. Output a single float."
  - Caching: keyed on (semantic_label, context_hash); avoids re-querying same class in same scene
  - Fallback: if latency > 500ms → static lookup table (road=0.95, grass=0.90, wet=0.40, mud=0.10, unknown=0.00)
  - Optional fine-tune: LoRA on Gemma-3 9B using RUGD traversability labels (binary → regressed to [0,1] via human annotation of 200 patches); if fine-tuned, target MAE < 0.10

**Correlate unknown areas to robot's actions:**
- Candidate trajectory pool: 6 heading angles × 3 distances = 18 trajectories; each is a path of patches
- Trajectory-level non-conformity score: κ_E(τ) = 1 − min_{p ∈ τ} trav(p)
  - Worst-case patch along trajectory drives the score (conservative, safety-oriented)
- Optimal trajectory: τ* = argmax_{τ} min_{p∈τ} trav(p)
- Decision rule: if κ_E(τ*) < δ_ask → PROCEED; if δ_ask ≤ κ_E(τ*) → ASK
  - δ_ask = 0.80 (i.e., minimum traversability below 0.20 triggers asking)
  - δ_ask calibrated on nav_env_calibration.json at target FPR ≤ 0.15

**Turn unknown areas to language context for questions:**
- Trajectories labeled 1–N; each patch described: "path 2 passes through unclassified low-lying vegetation (FM score: 0.28)"
- FM generates contextual sentence for unknown patches: "Describe what an operator should know about this terrain patch in one sentence" → appended to question
- Option list generated dynamically from top-3 viable trajectories (not fixed A/B/C)

### Environmental Uncertainty — Innovation Bullets

**Detection: Recognize what the robot doesn't know**
- SAM3+SAM2 spatial subtraction is architecturally novel: prior nav systems use SAM3 alone (GA-Nav) or rely on GPS uncertainty, not vision-based semantic gaps
- FM-TS replaces brittle hard-coded lookup table with grounded visual reasoning: LLM sees the actual patch image, not just a class label → handles edge cases (wet concrete looks like dry mud, shadow-covered grass scored as dry)
- Uncertainty score κ_E = 1 − min_{p∈τ} trav(p): takes worst-case traversability over trajectory, not mean → safety-conservative; mean would mask single dangerous patch

**Detection: Correlate unknown areas to robot's actions**
- Trajectory-level risk propagation: each of 18 candidate trajectories gets its own κ_E, not a single global "scene uncertainty" signal → robot can select a safe alternate route rather than always stopping
- This enables a richer robot response: PROCEED on path 1 even when path 2 has unknown terrain (prior work would halt or always ask)

**Detection: Turn to language context for questions**
- Dynamic option generation: question options are the actual top-3 best trajectories from the candidate pool, not generic A/B/C → question is grounded in current perception state
- FM-generated patch descriptions give the user concrete information to make a decision (prior work gives abstract "uncertain terrain ahead")

### How to Handle — Iterative Update

**Convert the generated language text to questions:**
- Personalized environmental question templates (3 verbosity × 3 expertise = 9 variants):
  - terse + novice: "Unknown terrain at 11 o'clock. Safe? [Y/N]"
  - standard + intermediate: "I see terrain I don't recognize ahead-left (path 2, risk 0.82). Should I: (A) take path 1 instead, (B) proceed slowly on path 2, (C) stop here?"
  - verbose + expert: "Traversability uncertainty detected (κ_E=0.82, FM confidence=0.68) on path 2 due to unclassified vegetation (SAM2 segment ID 14, no SAM3 match). Available: path 1 (κ=0.12, clear gravel), path 3 (κ=0.55, wet grass). Recommend path 1 unless you have specific information about path 2."
- FM fallback for novel situations: "Given UserProfile {verbosity, expertise, format}, write a clarification question about: '{patch_description}' with options: {trajectory_descriptions}"

**Update the existing understanding:**
- Bayesian traversability map update per patch after user response r:
  - τ_{k+1}(p) = [P(R="safe" | trav=1) · τ_k(p)] / [P(R="safe"|trav=1)·τ_k(p) + P(R="safe"|trav=0)·(1−τ_k(p))]
  - P(R="safe"|trav=1) = 0.95, P(R="safe"|trav=0) = 0.10
  - Numerical: unknown τ_0=0.5, user says safe → τ_1=0.905; unknown + says unsafe → τ_1=0.053
- Persistent terrain memory: keyed on (GPS_cell_hash, semantic_label) → (trav_score, n_observations)
  - If patch reencountered with τ > 0.80 AND n_obs ≥ 2 → skip asking, PROCEED
  - If τ < 0.20 AND n_obs ≥ 1 → skip asking, route around

**Move the robot accordingly:**
- Re-run trajectory selector on updated map: τ* = argmax_τ min_{p∈τ} τ_{updated}(p)
- If updated τ*(τ*) ≥ 1 − δ_ask → PROCEED on τ*
- If all trajectories still have max min-traversability < 1 − δ_ask → re-ask (max 3 rounds)
- After 3 failed rounds: STOP + report "All visible paths have uncertain terrain. Please assist the robot."

### Environmental Uncertainty — Iterative Update Innovation Bullets

**Convert to questions:**
- 9-variant personalized template bank is novel: prior work (KnowNo, IntroPlan) has one fixed question style regardless of user; user-adapted HRI for navigation clarification is unexplored
- Dynamic option generation from actual trajectory pool: options are tied to real robot plans, not abstract "left/right/stop" → user answer directly selects a waypoint

**Update existing understanding:**
- Bayesian map update is novel over prior work: GA-Nav and WhenToAsk treat each scene as independent; this system accumulates per-patch traversability beliefs across the mission (persistent map)
- Noisy channel model (p_tp=0.95, p_fp=0.10) explicitly models user miscalibration — user might say "safe" for muddy grass; system doesn't blindly trust the response
- Terrain memory prevents asking the same question twice (efficiency metric: NCR — no-clarification reuse rate)

**Move robot accordingly:**
- Trajectory re-selection on updated map directly links user feedback to robot behavior (closed loop): prior work stops after asking and requires restart; this replans in the same session

---

## Slide Content: Instruction Uncertainty

### How to Handle — Detection

**Recognize the areas/properties the robot doesn't know:**
- LLM slot-fill parser (GPT-4o or Gemma-3 9B with optional LoRA fine-tune):
  - Input: instruction string + scene description (detected object list from SAM3)
  - Output JSON: {action_verb, object_ref, direction, distance_metric, location_constraint, ambiguity_type, ambiguity_score, missing_slots}
  - Example: "Go there" → {action_verb:"go", object_ref:"there"[AMBIGUOUS], direction:null[MISSING], ambiguity_type:"ambiguous_target", ambiguity_score:0.85}
- 6-class ambiguity taxonomy: missing_action, missing_object, missing_direction, missing_distance, ambiguous_target, ambiguous_action
- Continuous ambiguity score: u_I = w_type(t*) · P(ambiguous | ℓ, s)
  - Severity weights: missing_action=1.0, ambiguous_target=0.75, missing_direction=0.5, missing_distance=0.25
  - P(ambiguous | ℓ, s) from LLM log-probability under temperature=0 ("Is this instruction unambiguous given this scene? Yes/No" → extract P(No))
- Conformal calibration of θ_ask: binary search over nav_calibration.json (57 scenarios) to minimize FPR while holding SR ≥ 1−ε

**Correlate unknown slots to robot's actions:**
- Constraint violation map: 18 candidate trajectories × |extracted_constraints| matrix; cell = 1 if trajectory violates constraint
  - Directional constraints: "go left" → all rightward trajectories violated
  - Object constraints: "go to the bench" → all trajectories not terminating near detected bench violated
  - Distance constraints: "go 5 meters" → trajectories shorter or longer than 4–6m violated
- τ* = argmin_τ violations(τ); ask if violations(τ*) > 0 OR u_I > θ_ask
- Precise missing slot recovery: if all distance-type constraints violated → generate question specifically about distance (not generic "what do you mean")

### Instruction Uncertainty — Innovation Bullets

**Detection: Recognize what the robot doesn't know**
- 6-class taxonomy is novel over KnowNo/IntroPlan (binary ambiguous/not): fine-grained type enables targeted follow-up questions (ask about missing distance vs. missing direction vs. ambiguous object separately)
- Continuous score u_I with per-type severity weights (novel over hard threshold): allows graded responses — high u_I for missing_action (robot literally can't move) vs. low u_I for missing_distance (robot can estimate)
- Optional LoRA fine-tune on Gemma-3 9B: 500 synthetic instruction–JSON pairs generated by GPT-4o; target slot-fill F1 > 0.90 on held-out set; 4-bit quantization for inference on embedded hardware

**Detection: Correlate to robot's actions**
- Constraint violation map (matrix, not scalar): novel structured representation — can identify which specific slot is missing from the violation pattern rather than just "ask or not ask"
- Non-conformity score κ_I = 1 − f̂(x̃^I)_{y*} where f̂ is LLM, y* is intended action: extends CP framework to language space (KnowNo does this too, but without multi-type handling or severity weighting)
- Joint CP: κ_joint = max(κ_I, κ_E) — novel: guarantees P(y* ∈ C^joint) ≥ 1−ε even when both uncertainty types are simultaneously active; prior CP approaches handle only one type

**Detection: Turn to language context for questions**
- Slot-targeted question generation: system knows exactly which slot is missing from the violation pattern → question is specific ("Where should I go?" not "What did you mean?")
- Severity-ordered asking: if multiple slots missing, ask about highest-weight missing slot first (missing_action before missing_distance)
- 9-variant personalized templates + FM fallback (see iterative update section below)

### Instruction Uncertainty — Iterative Update Innovation Bullets

**Convert to questions:**
- 9 template variants per situation_key (slot_type × ambiguity_type):
  - terse+novice, missing_object: "Which object?"
  - standard+intermediate, missing_object: "Your instruction mentions 'it,' but I see multiple objects. Which one do you mean: the bench or the trash can?"
  - verbose+expert, missing_object: "Ambiguity detected (u_I=0.85, type: ambiguous_target). 3 objects match 'it': bench (10m), gate (15m), sign (8m). Which is the intended destination?"
- FM fallback prompt: "UserProfile: {verbosity, expertise, format}. Missing slot: {slot_type}. Scene context: {detected_objects}. Write a clarification question in {N_words} words."
- Novel: verbosity × expertise × format = 27 combinations cover HRI needs from a construction worker with a radio to a research engineer using a tablet

**Update existing understanding:**
- Intent memory: {(instruction_type, context_hash): (resolved_answer, confidence, timestamp)}
  - Posterior: P(intent=i | response r, history H) ∝ P(r|intent=i) · ∏_{h∈H} P(r_h|intent=i) · P_0(i)
  - Smoothing: P_0(i) = 1/|intent_classes| initially; updated after each interaction
  - Re-use condition: if P(intent=i|current_instruction) > 0.85 from history → skip asking, proceed with i
- Novel over all prior CP-based work: KnowNo, IntroPlan, WhenToAsk treat every query independently; intent accumulation enables zero-shot reuse ("go there" → bench is resolved after first interaction)

**Move robot accordingly:**
- After clarification, re-run slot-fill parser with user response appended to instruction → fill missing slot
- Recompute constraint violation map with filled slots → select τ* = argmin violations
- If violations(τ*) still > 0 after 2 clarification rounds: "I cannot resolve this instruction. Please rephrase from the beginning." + STOP
- Final action selection tied directly to updated slot values: closed loop replanning in-session

---

## Evaluation Plan

### Datasets

| Dataset | Purpose | Size | Source |
|---------|---------|------|--------|
| nav_calibration.json | CP threshold calibration for instruction | 57 scenarios | Synthetic (KnowNo/WhenToAsk style) |
| nav_test.json | Instruction uncertainty evaluation | 30 scenarios | Synthetic, held-out |
| nav_env_test.json | Environmental uncertainty evaluation | 20 scenarios | Synthetic (RUGD-style scenes) |
| RUGD | FM-TS accuracy + traversability scoring ground truth | 7,546 images, 24 classes | Real outdoor robot data |
| RELLIS | Additional traversability validation | 13,556 images, 20 classes | Real outdoor robot data |
| FM-TS annotation set | MAE evaluation of FM traversability scorer | 30 RUGD patches, human-labeled [0,1] | Hand-annotated subset |
| QRS human study | Question quality rating | 3 raters × 30 questions = 90 ratings | User study |

**Important distinction:**
- RUGD/RELLIS are used for evaluating FM-TS accuracy and the environmental uncertainty detection pipeline, NOT as a training dataset for this work
- Instruction uncertainty evaluation uses nav_test.json (synthetic scenarios), not RUGD

### Baseline Comparison Table

| Baseline | Handles Env. Uncert. | Handles Instr. Uncert. | Personalization | CP Guarantee | Ablation Purpose |
|----------|:---:|:---:|:---:|:---:|--|
| Always-Act | ❌ | ❌ | ❌ | ❌ | Lower bound: never ask |
| Always-Ask | ❌ | ❌ | ❌ | ❌ | Upper bound: always ask |
| GA-Nav | Traversability only | ❌ | ❌ | ❌ | Env. nav without asking |
| KnowNo | ❌ | CP binary | ❌ | ✅ (single type) | Best prior CP for instruction |
| Introspective Planning | ❌ | CP+retrieval | ❌ | ✅ (single type) | Best prior CP v2 for instruction |
| WhenToAsk/UPS | ❌ | Action-level | ❌ | ✅ (action-level) | Action-level baseline |
| Ours (No Personalization) | ✅ | ✅ | ❌ | ✅ (joint) | Ablation: isolate personalization effect |
| **Ours (Full)** | ✅ | ✅ | ✅ | ✅ (joint) | Full system |

### Metrics Per Innovation

| Innovation | Primary Metric | Secondary Metric | Dataset | Covered? |
|------------|---------------|-----------------|---------|:---:|
| SAM3+SAM2 spatial subtraction | URDR: |detected ∩ true_unknown| / |true_unknown| | FPR_env | RUGD subset | ✅ |
| FM traversability scorer | MAE vs. hand-labeled RUGD patches | Score correlation (Pearson r) | FM-TS annotation set | ✅ |
| Joint conformal prediction | SR (success rate), HR (help rate) | FPR (false positive ask rate) | nav_test.json | ✅ |
| 6-class ambiguity taxonomy | Ambiguity type classification accuracy (F1 per class) | — | nav_test.json | ✅ |
| User personalization | QRS: 1–5 Likert rating by 3 raters | Verbosity compliance (word count within bounds) | User study | ✅ (needs human) |
| Bayesian map update | MUA: fraction of patches updated within ε of truth | Sequential convergence (n updates to τ > 0.8) | nav_env_test.json | ✅ |
| Intent memory reuse | NCR: fraction of re-encounters where asking was skipped correctly | — | nav_env_test.json | ✅ |

### Evaluation Coverage: Does It Cover All 4 Innovations?

**Innovation 1 — Environmental uncertainty detection (SAM3+SAM2+FM-TS):**
- URDR on RUGD tests detection accuracy
- MAE on 30 annotated patches tests FM-TS quality
- Coverage: ✅ both detection and scoring sub-components tested

**Innovation 2 — Joint conformal prediction (max-pooling κ):**
- SR/HR/FPR on nav_test.json tests prediction set quality for instruction uncertainty
- SAR (safe arrival rate) on nav_env_test.json tests environmental branch
- Ablation: Ours-NoPers vs. Always-Act/KnowNo isolates the joint CP contribution
- Coverage: ✅

**Innovation 3 — FM traversability scorer:**
- MAE vs. hand-labeled ground truth tests accuracy of scorer itself
- SAR difference between Ours-NoPers and GA-Nav tests downstream benefit
- Coverage: ✅

**Innovation 4 — User personalization:**
- QRS human rating tests question quality
- Ablation Ours-Full vs. Ours-NoPers tests whether personalization improves user satisfaction
- NCR tests intent memory benefit
- Coverage: ✅ (requires user study, even if small — 3 raters is sufficient for a research paper)

### Ablation Structure (What to Report in Paper)

1. Always-Act: establishes that asking is necessary (SR baseline without any asking)
2. Always-Ask: establishes that asking too much hurts efficiency (overcaution cost)
3. KnowNo → Ours-NoPers: shows joint CP + env. uncertainty adds accuracy
4. Ours-NoPers → Ours-Full: shows personalization adds question quality
5. Static lookup table → FM-TS: shows foundation model scoring improves traversability accuracy

---

## What Still Needs To Be Done (Action Items)

| Task | Priority | Effort |
|------|----------|--------|
| Implement Introspective Planning baseline (or confirm skip) | High | ~1 day |
| Annotate 30 RUGD patches with human traversability scores | High | ~2 hours |
| Write PersonalizedTemplateBank (9 × ~20 situation keys) | Medium | ~3 hours |
| Design nav_env_test.json (20 environmental scenarios) | High | ~2 hours |
| Run QRS user study (3 raters, 30 questions each) | Medium | ~1 day |
| Finalize LoRA fine-tune decision (Gemma-3 9B or prompt-only) | Medium | ~1 hour decision |
