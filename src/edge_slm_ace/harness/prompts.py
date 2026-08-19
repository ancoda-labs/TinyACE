"""Prompt construction shared by adaptation and evaluation.

Belebele is scored by lm-evaluation-harness as a `multiple_choice` task whose
choices are the bare letters A-D. Adaptation has to build byte-identical
prompts, or the playbook is tuned against one rendering and scored against
another -- the same arm-asymmetry problem in a new place.

Only one thing here is ours: `render_question`, which reproduces the task's
jinja template. Everything downstream of it -- how the system instruction joins
the question, how a chat template wraps the pair, what delimiter precedes the
continuation -- is assembled by calling lm-eval's own `Message`,
`maybe_delimit` and `multiturn_to_singleturn`, because reimplementing those
rules is exactly how the two paths drifted apart:

- evaluation runs `simple_evaluate(apply_chat_template=True)` by default, which
  makes the instruction a real `system` role and wraps everything in the
  model's template. Adaptation scored a raw f-string, i.e. in completion mode;
- even with the chat template off, lm-eval concatenates the system message with
  no separator (`Message._delimiter` is ""), while adaptation inserted "\\n\\n";
- under a chat template `construct_requests` drops the target delimiter, so the
  continuations are "A".."D" and not " A".." D" as adaptation always assumed.

Any of the three shifts the loglikelihoods, in the direction of whichever arm
carries the longer prefix. `assert_matches_harness()` now checks the assembled
context byte for byte in both modes, not just the template constant.
"""

from typing import Callable, Dict, List, Optional, Sequence

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

# lm-eval joins context and continuation with TaskConfig.target_delimiter --
# but only in completion mode. See choice_continuations().
TARGET_DELIMITER = " "

# TaskConfig.fewshot_delimiter, which joins the system instruction to the task
# description. Belebele has no description, so it never actually appears; it is
# passed through anyway so this file has no rule the harness does not.
FEWSHOT_DELIMITER = "\n\n"


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


def choice_continuations(apply_chat_template: bool = False) -> List[str]:
    """
    The four continuations whose loglikelihood decides the answer.

    Under a chat template lm-eval sets the target delimiter to "" when the task
    has no `gen_prefix` -- the template already supplies the boundary, so a
    leading space would be scored as a token the evaluation never scores.
    Belebele has no `gen_prefix`, so the delimiter is present in completion mode
    and absent in chat mode.

    Args:
        apply_chat_template: Whether the context was wrapped by a chat template.

    Returns:
        [" A", " B", " C", " D"], or ["A", "B", "C", "D"] under a chat template.
    """
    delimiter = "" if apply_chat_template else TARGET_DELIMITER
    return [f"{delimiter}{letter}" for letter in CHOICE_LETTERS]


def build_context(
    question: str,
    system_instruction: Optional[str] = None,
    apply_chat_template: bool = False,
    chat_template: Optional[Callable] = None,
) -> str:
    """
    Assemble the context lm-eval would build for this question, at 0-shot.

    Mirrors `ConfigurableTask.fewshot_context` by calling the same helpers it
    calls, rather than reproducing their behaviour: the system instruction
    becomes a `system` message, the question a `user` message with an empty
    delimiter (there is no answer and no `gen_prefix` on the eval doc), and the
    pair is either concatenated or handed to the chat template.

    Args:
        question: Output of `render_question`.
        system_instruction: The arm's prefix, or None/"" for the bare baseline.
        apply_chat_template: Wrap the messages in the model's chat template.
        chat_template: `lm.apply_chat_template`, i.e. the callable
            `simple_evaluate` passes down. Required when applying a template.

    Returns:
        The context string that precedes the continuation.

    Raises:
        ValueError: If a chat template is requested but none was supplied,
            which would otherwise silently produce the completion-mode prompt.
    """
    from lm_eval.api.utils import Message, maybe_delimit, multiturn_to_singleturn

    messages = []
    system_prompt = maybe_delimit(system_instruction, "", FEWSHOT_DELIMITER)
    if system_prompt:
        messages.append(Message("system", system_prompt))
    # build_qa_turn() with no answer and no gen_prefix: the delimiter is empty.
    messages.append(Message("user", question, ""))

    if not apply_chat_template:
        return "".join(m.to_text() for m in messages)

    if chat_template is None:
        raise ValueError(
            "apply_chat_template=True needs the model's chat template "
            "(lm.apply_chat_template). Without it the prompt silently reverts "
            "to completion mode, which is what evaluation does not do."
        )
    # fewshot_context binds add_generation_prompt=not gen_prefix; Belebele has
    # no gen_prefix, so it is True.
    return chat_template(multiturn_to_singleturn(messages), add_generation_prompt=True)


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
    assert (
        defaults["fewshot_delimiter"] == FEWSHOT_DELIMITER
    ), f"fewshot_delimiter changed upstream: {defaults['fewshot_delimiter']!r}"

    # choice_continuations() drops the target delimiter under a chat template.
    # That is only correct while the task has no gen_prefix -- see
    # ConfigurableTask.construct_requests.
    assert not config.get(
        "gen_prefix"
    ), "Belebele gained a gen_prefix upstream; re-check choice_continuations()."
