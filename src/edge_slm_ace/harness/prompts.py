"""Prompt construction shared by adaptation and evaluation.

Belebele is scored by lm-evaluation-harness as a `multiple_choice` task whose
choices are the bare letters A-D, with the continuation joined to the context by
`target_delimiter` (a single space). Adaptation has to build byte-identical
prompts, or the playbook is tuned against one rendering and scored against
another -- the same arm-asymmetry problem in a new place.

The task template is reproduced here verbatim from
`lm_eval/tasks/belebele/_default_template_yaml`. `assert_matches_harness()`
checks it against the installed harness so an upstream edit fails a test rather
than silently splitting the two paths apart.
"""

from typing import Dict, List, Optional, Sequence

# Verbatim from the harness task. Keep in sync via assert_matches_harness().
BELEBELE_DOC_TO_TEXT = (
    "P: {{flores_passage}}\n"
    "Q: {{question.strip()}}\n"
    "A: {{mc_answer1}}\n"
    "B: {{mc_answer2}}\n"
    "C: {{mc_answer3}}\n"
    "D: {{mc_answer4}}\n"
    "Answer:"
)

CHOICE_LETTERS: List[str] = ["A", "B", "C", "D"]

# lm-eval joins context and continuation with TaskConfig.target_delimiter.
TARGET_DELIMITER = " "


def render_question(example: Dict) -> str:
    """
    Render one example exactly as the harness renders it.

    Args:
        example: A normalised Belebele example (see `data/belebele.py`).

    Returns:
        The context string, ending in "Answer:".
    """
    options = example["options"]
    return (
        f"P: {example['context']}\n"
        f"Q: {example['question'].strip()}\n"
        f"A: {options[0]}\n"
        f"B: {options[1]}\n"
        f"C: {options[2]}\n"
        f"D: {options[3]}\n"
        f"Answer:"
    )


def choice_continuations() -> List[str]:
    """The four continuations whose loglikelihood decides the answer."""
    return [f"{TARGET_DELIMITER}{letter}" for letter in CHOICE_LETTERS]


# The instruction block every non-baseline arm receives.
#
# `scaffold_control` gets this with no lessons; the ACE arms get it with lessons
# appended. That is what makes the two comparable: the only difference is
# content the playbook contributed.
SCAFFOLD_HEADER = (
    "You are answering multiple-choice reading comprehension questions.\n"
    "Read the passage, then choose the option best supported by it.\n"
    "Respond with a single letter: A, B, C, or D."
)

LESSON_HEADER = "\n\nStrategies learned from previous questions in this domain:"

LESSON_FOOTER = "\n\nApply these strategies when they are relevant to the passage."


def build_system_instruction(
    lessons: Optional[Sequence[str]] = None,
    include_scaffold: bool = True,
) -> str:
    """
    Render the prefix passed to `simple_evaluate(system_instruction=...)`.

    Args:
        lessons: Frozen playbook lesson texts, in retrieval order. None or empty
            produces the control arm's prefix.
        include_scaffold: False produces the bare `baseline` arm, which gets no
            prefix at all.

    Returns:
        The instruction text. Empty string for the baseline arm.

    Note:
        The prefix is *static* across the evaluation split. A frozen playbook
        cannot be retrieved per question through this interface, so every
        question sees the same lessons. That is a deliberate limitation of the
        loglik track and has to be stated as one: query-conditioned retrieval is
        measurable only in the generative track.
    """
    if not include_scaffold:
        return ""

    text = SCAFFOLD_HEADER
    if lessons:
        text += LESSON_HEADER
        for i, lesson in enumerate(lessons, 1):
            text += f"\n{i}. {lesson}"
        text += LESSON_FOOTER
    return text


def assert_matches_harness() -> None:
    """
    Verify this module still agrees with the installed harness task.

    Raises:
        AssertionError: If the template or delimiter has changed upstream, which
            would silently desynchronise adaptation from evaluation.
    """
    import dataclasses
    from pathlib import Path

    import yaml
    from lm_eval.api.task import TaskConfig
    import lm_eval.tasks

    template_path = Path(lm_eval.tasks.__file__).parent / "belebele" / "_default_template_yaml"
    config = yaml.safe_load(template_path.read_text(encoding="utf-8"))

    assert config["doc_to_text"] == BELEBELE_DOC_TO_TEXT, (
        "Belebele doc_to_text changed upstream; update BELEBELE_DOC_TO_TEXT "
        "and re-check render_question()."
    )
    assert (
        config["doc_to_choice"] == CHOICE_LETTERS
    ), f"Belebele doc_to_choice changed upstream: {config['doc_to_choice']}"
    assert config["output_type"] == "multiple_choice"

    defaults = {f.name: f.default for f in dataclasses.fields(TaskConfig)}
    assert (
        defaults["target_delimiter"] == TARGET_DELIMITER
    ), f"target_delimiter changed upstream: {defaults['target_delimiter']!r}"
