"""Evaluation package for external benchmark runs (RCAEval RE2-OB).

`evaluator` is a faithful port of RCAEval's scoring semantics; it is the only
place in the repo that touches ground-truth labels. The engine path
(`rcaeval_adapter` -> P0/P1/P2 -> rank export) never sees them.
"""
