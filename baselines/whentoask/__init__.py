"""
When-to-Ask (UPS) Navigation Baseline

Implements the Uncertainty-Aware Policy Steering framework adapted to
text-based outdoor navigation under uncertainty.

Three resolution strategies (from Yuan et al., UPS, arXiv:2602.22474):
  EXECUTE   — prediction set = one real option (A/C/D), no "none" → act directly
  CLARIFY   — prediction set has multiple real options, no "none" → ask user
  INCAPABLE — prediction set includes "E" (none of the above) → escalate

Key differences from IntroPlan and KnowNo:
  1. Adds option E: "none of the above" as a K+1-th incapability signal
  2. Three-way strategy split instead of binary act/ask
  3. Bayesian intent factorization to reduce VLM overconfidence before CP
"""
