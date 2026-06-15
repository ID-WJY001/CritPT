from __future__ import annotations

import hashlib
import json
import random
from dataclasses import replace

from rl_posttrain.critpt_synth.schema import SyntheticCritPTExample
from rl_posttrain.critpt_synth.v15_hardmix import (
    generate_v15_hardmix_examples,
    summarize_v15_source,
    verify_v15_example,
)


def generate_v18_official_long_examples(
    size: int,
    seed: int,
    split: str,
) -> list[SyntheticCritPTExample]:
    """Wrap V15/V17 hard cases in longer official-like notebook prompts."""

    rng = random.Random(seed + 1800)
    examples = generate_v15_hardmix_examples(size, seed, split)
    return [_wrap_official_long(example, rng, idx) for idx, example in enumerate(examples)]


def verify_v18_example(example: SyntheticCritPTExample) -> tuple[bool, str]:
    return verify_v15_example(example)


def summarize_v18_source(example: SyntheticCritPTExample) -> str:
    source = example.metadata.get("v18_source")
    if isinstance(source, str):
        return source
    return summarize_v15_source(example)


def _wrap_official_long(
    example: SyntheticCritPTExample,
    rng: random.Random,
    idx: int,
) -> SyntheticCritPTExample:
    preamble = _official_like_preamble(example, rng)
    anti_shortcut = rng.choice(_ANTI_SHORTCUT_NOTES)
    prompt = f"""# Problem setup:
{preamble}

The benchmark cell below contains a compact synthetic sub-problem. Treat it as
the authoritative statement of the task. The surrounding physics prose is
context only: do not guess a famous formula, do not return placeholder zeros,
and do not add unused candidates.

# Main problem:
Solve the embedded task exactly. Return only the Python code block requested by
the embedded parsing template.

### Embedded benchmark cell:

{example.prompt}

Official-style guardrails:
- The embedded `Parsing template` is the contract.
- Keep `def answer(...)` complete and executable.
- Close the code block.
- If the answer is a list or tuple, include only entries that satisfy the stated
  filters; no near misses and no padding zeros.
- {anti_shortcut}
"""
    source = summarize_v15_source(example)
    payload = {
        "problem_id": example.problem_id,
        "idx": idx,
        "source": source,
        "preamble": preamble,
        "anti_shortcut": anti_shortcut,
    }
    wrapper_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return replace(
        example,
        problem_id=f"{example.problem_id}_v18long_{wrapper_hash}",
        prompt=prompt,
        metadata={
            **example.metadata,
            "generator_profile": "v18_official_long_hardmix",
            "v18_source": source,
            "v18_wrapper_hash": wrapper_hash,
            "uses_official_prompt": False,
            "official_overlap": "none",
            "prompt_style": "official_long_embedded_hardmix",
            "long_context_training": True,
            "anti_placeholder_zero": True,
            "anti_runaway": True,
            "no_think_target": True,
        },
    )


def _official_like_preamble(example: SyntheticCritPTExample, rng: random.Random) -> str:
    domain = str(example.metadata.get("domain", "quantum_field_theory"))
    family = example.family
    topic = _topic_for(domain, family)
    blocks = [_INTRO_BLOCKS[topic]]
    distractor_count = rng.choice([2, 3, 3, 4])
    blocks.extend(rng.sample(_DISTRACTOR_BLOCKS, k=distractor_count))
    blocks.append(
        "The final answer is checked by an automated parser, so the mathematical "
        "discussion above is not itself the output. The answer must be expressed "
        "through the exact Python function requested later in the prompt."
    )
    return "\n\n".join(block.strip() for block in blocks)


def _topic_for(domain: str, family: str) -> str:
    haystack = f"{domain} {family}".lower()
    if "operator" in haystack or "gauge" in haystack:
        return "operator"
    if "oam" in haystack or "angular" in haystack:
        return "oam"
    if "piecewise" in haystack or "lamet" in haystack:
        return "piecewise"
    if "recurrence" in haystack or "sequence" in haystack:
        return "recurrence"
    if "bns" in haystack or "interval" in haystack:
        return "interval"
    return "holography"


_INTRO_BLOCKS = {
    "operator": """
In a large-N gauge theory notebook, one often enumerates candidate single-trace
operators before imposing charge, spin, parity, and compactness filters. The
notation may look like a physical operator basis, but this benchmark asks for a
finite combinatorial object rather than a derivation of the full spectrum.
""",
    "oam": """
High-harmonic generation calculations often organize channels by orbital angular
momentum, helicity, and discrete selection rules. In the simplified benchmark
cell, the long selection-rule discussion is reduced to a finite table of allowed
channels that must be filtered exactly.
""",
    "piecewise": """
Piecewise kernels appear in LaMET and matching calculations when different
kinematic regions contribute different finite terms. The benchmark cell gives a
reduced symbolic version of such a kernel; the task is to keep the requested
regions and expressions in the specified order.
""",
    "recurrence": """
Generating functions and recurrence relations show up in spectral counting
problems. The simplified benchmark cell below gives all constants needed to
compute the requested finite prefix or closed form; no external theorem is
needed.
""",
    "interval": """
Binary neutron-star and detector sensitivity calculations frequently reduce to
intersections of allowed parameter windows. The benchmark cell gives a compact
interval arithmetic version of this idea and expects the exact surviving ranges.
""",
    "holography": """
Holographic anomaly problems can contain many tensor names and scheme-dependent
coefficients. In this benchmark variant, the physical language is context; the
actual answer is determined by the finite symbolic or numeric rules in the
embedded cell.
""",
}


_DISTRACTOR_BLOCKS = [
    """
For orientation, let P_{mu nu}, B_{mu nu}, and Omega_{mu nu} denote tensor-like
placeholders. Their continuum definitions are irrelevant here; only the finite
rules in the parsing cell are executable.
""",
    """
Some terms are written in an ordered basis. Unless the later task explicitly
asks for products, reversed duplicates, or over-threshold terms, they must be
excluded rather than padded into the answer.
""",
    """
Numerical constants in this preamble are illustrative: alpha=2, beta=5,
gamma=7, and Lambda=0.244. They are intentionally not the source of truth for
the embedded task.
""",
    """
The checker will not award credit for a generic physics essay. It evaluates the
return value of `answer()`, so the code must be syntactically valid and the
returned object must have the requested shape.
""",
    """
If several candidate channels have the same score, the later task gives a stable
tie-break rule. Use that rule literally rather than relying on physical
intuition or common ordering conventions.
""",
    """
The full notebook may mention integrals, projectors, kernels, or anomalous
dimensions. In this reduced task, expensive numerical integration is usually a
distraction unless it appears in the provided template.
""",
]


_ANTI_SHORTCUT_NOTES = [
    "Do not return an all-zero vector unless the embedded rules explicitly imply every entry is zero.",
    "Do not import scipy, sympy, or numpy unless the embedded template already requires it.",
    "Do not solve a different physical problem with similar notation.",
    "Do not use a long explanatory preface; the answer is only the code block.",
]
