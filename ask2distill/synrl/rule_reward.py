"""Null rule-based reward for synthesizer RL (and for the code / tau2 OPD arms).

verl always runs a rule-based reward function alongside the reward model, keyed
on data_source, and dies with NotImplementedError on an unknown one. There is no
ground truth here -- a generated question has no correct answer to check -- so
this returns 0 and the entire learning signal comes from the reward worker's
rm_scores (divergence x compatibility). The code OPD arms use it for the same
reason: ttrl_math would run answer matching against ground_truth "N/A" on a
code solution, and the teacher supplies the whole signal.
"""


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> float:
    """Always 0.0 (logging only; ``rm_scores`` carry the reward)."""
    return 0.0
