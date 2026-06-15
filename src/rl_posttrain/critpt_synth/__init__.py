"""Synthetic CritPT-style code-answer data and verifier helpers."""

from rl_posttrain.critpt_synth.schema import SyntheticCritPTExample
from rl_posttrain.critpt_synth.verifier import verify_code_completion

__all__ = ["SyntheticCritPTExample", "verify_code_completion"]
