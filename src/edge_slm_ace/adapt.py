"""Build a playbook on the adaptation split, then freeze it.

This is the only loop this project owns. Evaluation is a stock
lm-evaluation-harness run over a frozen artifact, so nothing here can influence
a reported number except through the lessons it writes.

Two properties make the loop honest:

- **Correctness is decided the same way evaluation decides it** -- the
  loglikelihood of the letters A-D, through the same `HFLM` and the same prompt
  rendering. Adaptation used to generate free text and parse it, which meant the
  playbook was tuned against one notion of correctness and reported against
  another.
- **It only ever sees the adaptation split.** The Reflector is shown gold
  answers, which is why the split it runs on must never be scored.
"""

import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from edge_slm_ace.core.ace_roles import (
    build_curator_prompt,
    choose_lessons_for_playbook,
    parse_curator_output,
    parse_reflector_output_to_lessons,
)
from edge_slm_ace.harness.prompts import CHOICE_LETTERS
from edge_slm_ace.harness.scorer import OptionScorer, option_margin
from edge_slm_ace.memory.playbook import Playbook

# Columns of the per-step adaptation log, in write order. Defined next to the
# only code that builds a row so a new field cannot leave a consumer behind.
ADAPT_LOG_FIELDS: List[str] = [
    "step",
    "item_id",
    "correct",
    "chosen_idx",
    "gold_idx",
    "gold_margin",
    "num_retrieved",
    "reflected",
    "lessons_proposed",
    "lessons_rejected_by_curator",
    "entries_added",
    "num_evictions",
    "num_entries",
    "total_tokens",
    "score_ms",
    "reflect_ms",
    "curate_ms",
]


def build_reflector_prompt(example: Dict, chosen_idx: int) -> str:
    """
    Ask the model why it chose wrongly, in terms of the passage.

    Belebele is reading comprehension, so a useful lesson is about how to read a
    passage -- not a formula. The prompt says so explicitly, because the generic
    "extract an actionable rule" phrasing produces arithmetic advice on a task
    with no arithmetic in it.

    Args:
        example: A normalised Belebele example.
        chosen_idx: The option the model's loglikelihoods preferred.

    Returns:
        The Reflector prompt.
    """
    options = example["options"]
    gold = example["gold_option_idx"]
    rendered = "\n".join(f"{L}: {t}" for L, t in zip(CHOICE_LETTERS, options))
    return f"""A reading-comprehension question was answered incorrectly.

Passage:
{example['context']}

Question: {example['question'].strip()}
{rendered}

Chosen: {CHOICE_LETTERS[chosen_idx]}. {options[chosen_idx]}
Correct: {CHOICE_LETTERS[gold]}. {options[gold]}

Write 1-2 short, reusable reading strategies that would have led to the correct
option on THIS question and would transfer to other passages.

Each strategy must:
- Describe how to read or compare the passage and the options
- Name the specific trap that was fallen into (for example: an option that is
  true in general but not stated in the passage; an option that reverses a
  cause; an option that answers a different question)
- Be usable without having seen this passage

Do NOT restate the answer to this question, and do NOT mention this passage's
subject matter. Write one strategy per line, starting with "-".
"""


def adapt_playbook(
    scorer: OptionScorer,
    examples: Sequence[Dict],
    playbook: Playbook,
    domain: str,
    generate: Optional[Callable[[str, int], str]] = None,
    top_k: int = 5,
    reflect_on_correct_every_n: int = 0,
    prune_every_n: int = 25,
    max_entries_per_domain: int = 32,
    use_curator: bool = True,
    max_new_tokens: int = 192,
    progress: bool = True,
) -> Dict:
    """
    Run one adaptation pass and return the log.

    The playbook is mutated in place; the caller saves it.

    Args:
        scorer: Decides correctness, identically to evaluation.
        examples: The adaptation split only. Never the evaluation split -- the
            Reflector is shown gold answers.
        playbook: Playbook to grow.
        domain: Domain to file lessons under. Use one domain per language to
            keep playbooks separate, or a shared one to study cross-lingual
            transfer.
        generate: `(prompt, max_new_tokens) -> text`. Defaults to the scorer's
            own model, so reflection and scoring share one checkpoint.
        top_k: Lessons retrieved per question.
        reflect_on_correct_every_n: 0 reflects only on errors. Reflecting on
            correct answers costs a generation each and mostly produces
            restatements of what already worked.
        prune_every_n: Prune every N steps.
        max_entries_per_domain: Cap after pruning.
        use_curator: Screen candidate lessons with one extra generation.
        max_new_tokens: Cap on reflection and curation length.
        progress: Print per-step progress.

    Returns:
        Dict with `log` (one row per step), `accuracy`, and counters.
    """
    if generate is None:
        generate = _default_generate(scorer)

    log: List[Dict] = []
    correct_count = 0

    for step, example in enumerate(examples, start=1):
        retrieved = playbook.get_top_k(
            domain, k=top_k, current_step=step, query=example["question"]
        )
        lesson_texts = [e.text for e in retrieved]

        start = time.time()
        chosen_idx, logprobs = scorer.score_one(example, lessons=lesson_texts)
        score_ms = (time.time() - start) * 1000

        gold_idx = example["gold_option_idx"]
        correct = chosen_idx == gold_idx
        correct_count += int(correct)

        should_reflect = (not correct) or (
            reflect_on_correct_every_n and step % reflect_on_correct_every_n == 0
        )

        proposed: List[str] = []
        curated: List[str] = []
        rejected = 0
        reflect_ms = curate_ms = 0.0

        if should_reflect:
            start = time.time()
            reflection = generate(build_reflector_prompt(example, chosen_idx), max_new_tokens)
            reflect_ms = (time.time() - start) * 1000

            proposed = choose_lessons_for_playbook(
                domain=domain,
                lessons=parse_reflector_output_to_lessons(reflection),
                existing_playbook=playbook,
            )
            curated = proposed

            if use_curator and proposed:
                start = time.time()
                verdict = generate(
                    build_curator_prompt(domain, proposed), max(64, max_new_tokens // 3)
                )
                curate_ms = (time.time() - start) * 1000
                flags = parse_curator_output(len(proposed), verdict)
                curated = [l for l, generic in zip(proposed, flags) if not generic]
                rejected = len(proposed) - len(curated)

        ids_before = {e.id for e in playbook.entries}
        for lesson in curated:
            playbook.add_entry(domain=domain, text=lesson, step=step)
        ids_after = {e.id for e in playbook.entries}
        entries_added = len(ids_after - ids_before)
        evictions = len(ids_before - ids_after)

        # Recency tracks what was shown; feedback tracks whether it helped.
        # There is no citation to attribute by -- the model emits no text at
        # scoring time -- so credit is uniform over what was retrieved, and the
        # success/failure terms should be read as weak per-lesson evidence.
        for entry in retrieved:
            playbook.mark_entry_used(entry.id, step)
            playbook.record_feedback(entry.id, helpful=correct)

        if prune_every_n and step % prune_every_n == 0:
            before = len(playbook.entries)
            playbook.prune(max_entries_per_domain=max_entries_per_domain, current_step=step)
            evictions += before - len(playbook.entries)

        log.append(
            {
                "step": step,
                "item_id": example["id"],
                "correct": int(correct),
                "chosen_idx": chosen_idx,
                "gold_idx": gold_idx,
                "gold_margin": option_margin(logprobs, gold_idx),
                "num_retrieved": len(retrieved),
                "reflected": int(bool(should_reflect)),
                "lessons_proposed": len(proposed),
                "lessons_rejected_by_curator": rejected,
                "entries_added": entries_added,
                "num_evictions": evictions,
                "num_entries": len(playbook.entries),
                "total_tokens": playbook.total_tokens,
                "score_ms": score_ms,
                "reflect_ms": reflect_ms,
                "curate_ms": curate_ms,
            }
        )

        if progress:
            print(
                f"[{step}/{len(examples)}] correct={int(correct)} "
                f"playbook={len(playbook.entries)} (+{entries_added})"
            )

    return {
        "log": log,
        "num_examples": len(examples),
        "accuracy": correct_count / len(examples) if examples else 0.0,
        "playbook_size": len(playbook.entries),
        "playbook_tokens": playbook.total_tokens,
        "lessons_rejected_by_curator": sum(r["lessons_rejected_by_curator"] for r in log),
        "total_reflect_ms": sum(r["reflect_ms"] for r in log),
        "total_curate_ms": sum(r["curate_ms"] for r in log),
    }


def _default_generate(scorer: OptionScorer) -> Callable[[str, int], str]:
    """Generation through the scorer's own model, so one checkpoint serves both."""

    def generate(prompt: str, max_new_tokens: int) -> str:
        from lm_eval.api.instance import Instance

        request = Instance(
            request_type="generate_until",
            doc={},
            arguments=(prompt, {"max_gen_toks": max_new_tokens, "until": ["\n\n\n"]}),
            idx=0,
        )
        return scorer.lm.generate_until([request])[0]

    return generate


def frozen_lessons(playbook: Playbook, domain: str, top_k: int = 5) -> List[str]:
    """
    The lessons a frozen evaluation will show, in rank order.

    Retrieval at evaluation time is not query-conditioned in the loglik track:
    `system_instruction` is one static prefix for the whole split. This selects
    the top entries by retention score alone, which is what that prefix holds.

    Args:
        playbook: The adapted playbook.
        domain: Domain to draw from.
        top_k: How many lessons the prefix carries.

    Returns:
        Lesson texts, best first.
    """
    entries = playbook.get_top_k(domain, k=top_k, current_step=0, query=None)
    return [e.text for e in entries]


def save_adaptation_log(log: Sequence[Dict], path: Path) -> None:
    """Write the per-step log as CSV, using ADAPT_LOG_FIELDS as the header."""
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ADAPT_LOG_FIELDS)
        writer.writeheader()
        writer.writerows(log)
