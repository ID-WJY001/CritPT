from __future__ import annotations


FINAL_INSTRUCTION = (
    "/no_think\n"
    "Fill the template above. Do not output <think> tags, hidden reasoning, prose, or comments. "
    "Respond with exactly one Python code block containing only the complete answer() function. "
    "Keep the function compact: prefer directly returning final literal values after doing any calculation privately; "
    "do not copy long input tables into the answer unless absolutely necessary. "
    "Use concise Python, close the code block, and stop."
)


def render_prompt(problem_setup: str, main_problem: str, code_template: str) -> str:
    return (
        f"# Problem setup:\n{problem_setup.strip()}\n\n"
        f"# Main problem:\n{main_problem.strip()}\n\n"
        "### Parsing template:\n\n"
        f"```python\n{code_template.strip()}\n```\n\n"
        f"{FINAL_INSTRUCTION}"
    )
