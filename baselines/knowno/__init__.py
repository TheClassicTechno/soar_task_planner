"""
KnowNo Navigation Baseline

Implements the KnowNo (Ren et al., 2023) approach applied to outdoor robot
navigation uncertainty. KnowNo uses conformal prediction to determine when
a robot should act vs. ask for clarification — without retrieval augmentation.

Key difference from IntroPlan:
  IntroPlan: LLM sees retrieved similar examples (RAG) before predicting.
  KnowNo:    LLM scores options directly from the scenario alone (no RAG).

Both use identical conformal prediction calibration and the same evaluation
metrics. The comparison shows whether retrieval augmentation actually helps.

Reference:
  Allen Z. Ren et al., "Robots That Ask For Help: Uncertainty Alignment for
  Large Language Model Planners," CoRL 2023. arXiv:2307.01928
"""
