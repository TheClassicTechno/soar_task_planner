"""
Environmental Uncertainty Detection System

This package implements the autonomous environmental uncertainty pipeline
described in docs/methodology.md §4. It operates without a user instruction —
the robot detects uncertainty by perceiving the environment directly.

Pipeline:
  1. EnvironmentalUncertaintyDetector: run SAM3 (known regions) + SAM2
     (all regions); subtract to find unknown regions.
  2. TraversabilityMap: assign per-pixel traversability scores from
     terrain class labels and unknown-region flags.
  3. TrajectoryGenerator: generate candidate trajectories and score each
     against the traversability map.
  4. QuestionGenerator: convert the unknown region description into a
     natural-language question for the user.
  5. MapUpdater: apply user feedback to update the traversability map.
  6. EnvironmentalUncertaintyRunner: orchestrate steps 1–5 end-to-end.
"""
