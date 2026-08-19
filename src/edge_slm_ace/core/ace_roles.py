"""Reflector and Curator: the two roles that still need a model to write text.

The Generator is gone. Under loglikelihood option scoring the model never
produces an answer to parse -- it ranks four fixed continuations -- so the
generator prompt and the citation machinery it carried have no role in the
primary track. parse_generator_output and extract_answer are kept because the
secondary generative track still has text to parse.
"""

import re
from typing import List, Optional, Tuple

from edge_slm_ace.memory.playbook import (
    _GENERIC_FLOOR,
    Playbook,
    compute_vagueness_score,
)

# Lines that end a section without starting one.
#
# Retained for the secondary generative track, where the model is asked to name
# the strategies it applied. That block arrives *after* the answer, and without
# a terminator the section parser folds it into the answer -- which corrupts the
# scored prediction in the arm that cites and leaves the arm that does not
# untouched.
SECTION_TERMINATORS = ["used strategies:", "used strategy:"]


def _drop_from_terminator(lines: List[str]) -> List[str]:
    """
    Truncate at the first terminator line.

    Args:
        lines: Non-empty, stripped output lines.

    Returns:
        Everything before the citation block. The whole list when there is
        none, so non-ACE output is unaffected.
    """
    for i, line in enumerate(lines):
        if any(line.lower().startswith(k) for k in SECTION_TERMINATORS):
            return lines[:i]
    return lines


def parse_generator_output(text: str) -> Tuple[str, Optional[str]]:
    """
    Parse the Generator's output to extract reasoning and answer.

    This parser is designed to be robust to various output formats:
    - Explicit "Reasoning:" and "Answer:" sections
    - Just an answer without structure
    - Malformed outputs with missing delimiters
    - Numeric answers embedded in text

    The Generator output ideally has the format:
        Reasoning:
        [reasoning text]

        Answer:
        [answer text]

    But we try to recover even when the format is different.

    Args:
        text: Raw output from the Generator model.

    Returns:
        Tuple of (answer, reasoning).
        - answer: The extracted answer (never None, at worst returns cleaned text)
        - reasoning: The extracted reasoning (may be None if not found)
    """
    if not text or not text.strip():
        return "", None

    text = text.strip()
    reasoning = None
    answer = None

    import re

    # Strategy 1: Section-based parsing (most reliable for structured output)
    reasoning_keywords = ["reasoning:", "step-by-step:", "steps:", "solution:"]
    answer_keywords = ["answer:", "final answer:", "result:", "therefore:"]

    lines = text.split("\n")
    current_section = None
    reasoning_lines = []
    answer_lines = []

    for line in lines:
        line_stripped = line.strip()
        line_lower = line_stripped.lower()

        # Check if this line starts a new section
        is_reasoning_header = any(line_lower.startswith(k) for k in reasoning_keywords)
        is_answer_header = any(line_lower.startswith(k) for k in answer_keywords)
        is_terminator = any(line_lower.startswith(k) for k in SECTION_TERMINATORS)

        if is_terminator:
            # Closes whatever section is open and contributes nothing. The
            # answer section is last in the Generator's response format, so
            # without this the citation block the ACE prompt asks for was
            # appended to the answer -- and only the ACE arm is asked for it,
            # so exact match was destroyed in one arm and intact in the other.
            current_section = None
        elif is_answer_header:
            current_section = "answer"
            # Extract text after the keyword
            for keyword in answer_keywords:
                if line_lower.startswith(keyword):
                    after_keyword = line_stripped[len(keyword) :].strip()
                    if after_keyword:
                        answer_lines.append(after_keyword)
                    break
        elif is_reasoning_header:
            current_section = "reasoning"
            for keyword in reasoning_keywords:
                if line_lower.startswith(keyword):
                    after_keyword = line_stripped[len(keyword) :].strip()
                    if after_keyword:
                        reasoning_lines.append(after_keyword)
                    break
        elif current_section == "reasoning" and line_stripped:
            reasoning_lines.append(line_stripped)
        elif current_section == "answer" and line_stripped:
            answer_lines.append(line_stripped)

    if reasoning_lines:
        reasoning = "\n".join(reasoning_lines).strip()

    if answer_lines:
        answer = "\n".join(answer_lines).strip()

    # Strategy 3: Fallback - extract last meaningful content
    if not answer:
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        # Drop the citation block and everything under it before scanning
        # backwards. When the model emits no "Answer:" header at all -- routine
        # for a 1.1B model -- the last line is the citation the prompt asked
        # for, and a reverse scan would return "1, 3" as the answer.
        lines = _drop_from_terminator(lines)

        if lines:
            # Try to find the last line that looks like an answer
            # (contains a number, is short, or is after "therefore"/"so"/"thus")
            for i in range(len(lines) - 1, -1, -1):
                line = lines[i]
                line_lower = line.lower()

                # Skip lines that are clearly just labels
                if line_lower in ["reasoning:", "answer:", "steps:", "solution:"]:
                    continue

                # Found a content line
                # If it starts with a transition word, take what follows
                for prefix in ["therefore,", "so,", "thus,", "hence,"]:
                    if line_lower.startswith(prefix):
                        answer = line[len(prefix) :].strip()
                        break

                if not answer:
                    answer = line
                break

    # Strategy 4: Last resort - just clean and return the text
    if not answer:
        # Remove common prefixes and return
        answer = text.strip()
        # Try to extract just the final value if text ends with a number
        final_number = re.search(r"(\$?[\d,]+\.?\d*%?)\s*$", answer)
        if final_number:
            answer = final_number.group(1)

    # Clean up the answer
    if answer:
        # Remove trailing punctuation except for % and .
        answer = answer.rstrip(".,;:!?")
        if answer.endswith(".") or answer.endswith("%"):
            pass  # Keep these
        answer = answer.strip()

    return answer or "", reasoning


def extract_answer(raw_output: str) -> Tuple[str, Optional[str]]:
    """
    Extract the final answer from a model generation.

    Every arm must go through this one function. Previously the ACE arm ran
    parse_generator_output while the baseline arm scored the raw generation,
    so a difference attributed to the playbook could just as easily have been
    a difference in parsing. That asymmetry was most damaging for small
    models: parse_generator_output falls through to "return the whole cleaned
    text" when no `Answer:` header is emitted, which is exactly what a 1.1B
    model does, so the ACE arm was scored on a full reasoning dump while the
    baseline arm was scored on a short answer.

    Args:
        raw_output: The raw generation.

    Returns:
        Tuple of (answer, reasoning). `answer` falls back to the stripped raw
        output rather than the empty string.
    """
    answer, reasoning = parse_generator_output(raw_output)
    if not answer:
        answer = (raw_output or "").strip()
    return answer, reasoning


# Lines that introduce a list rather than being one of its items.
_PREAMBLE_RE = re.compile(
    r"^(here (are|is)|the following|below are|two|strategies|lessons|reading strategies)\b",
    re.IGNORECASE,
)

# Leading bullet or list numbering.
_BULLET_RE = re.compile(r"^\s*(?:[-•*]+|\d+[.)])\s*")


def _is_degenerate(text: str, min_unique_ratio: float = 0.4) -> bool:
    """
    True for output that repeats one token instead of saying something.

    A small or randomly-initialised model answers the Reflector with runs like
    "factors factors factors ...". That is 200 words, passes every length check,
    contains no generic phrase and therefore scores vagueness 0.0 -- the most
    specific-looking entry in the playbook. The CI smoke test produces exactly
    this, which is how it was found.
    """
    words = [w.lower() for w in text.split()]
    if len(words) < 6:
        return False
    return len(set(words)) / len(words) < min_unique_ratio


def parse_reflector_output_to_lessons(
    text: str,
    min_words: int = 4,
    max_words: int = 60,
) -> List[str]:
    """
    Parse the Reflector's output into a list of lesson strings.

    The prompt asks for one strategy per line starting with "-", so when the
    model complies, only those lines are lessons. The previous version also
    accepted any line longer than ten characters *in addition* to the bullets,
    which meant the sentence introducing the list -- "Here are two reading
    strategies that would have helped:" -- was stored as a strategy alongside
    them. Free-form lines are a fallback for output with no bullets at all, not
    a supplement to output that has them.

    Args:
        text: Raw output from the Reflector model.
        min_words: Shorter candidates are fragments, not strategies.
        max_words: Longer ones are the model restating the passage.

    Returns:
        Lesson strings, cleaned and filtered.
    """
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]

    bulleted = [_BULLET_RE.sub("", line) for line in lines if _BULLET_RE.match(line)]
    candidates = bulleted if bulleted else lines

    lessons = []
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        # A line ending in a colon introduces what follows; it is not itself a
        # strategy. Same for the stock lead-ins.
        if candidate.endswith(":") or _PREAMBLE_RE.match(candidate):
            continue
        if not min_words <= len(candidate.split()) <= max_words:
            continue
        if _is_degenerate(candidate):
            continue
        lessons.append(candidate)

    return lessons


def build_curator_prompt(
    domain: str,
    lessons: List[str],
) -> str:
    """
    Build a prompt for the Curator role (model that marks generic rules).

    The Curator marks obviously generic rules as is_generic=True so they are
    rejected before they reach the playbook.

    This prompt used to demand "concrete formulas, equations, procedures" and
    illustrate them with compound interest and percentage conversion -- carried
    over unchanged from the arithmetic task this project used to run. On
    Belebele it rejected almost every well-formed reading strategy, including
    the ones `build_reflector_prompt` explicitly asks for, so the Curator arm
    and the no-Curator arm differed mainly in whether the playbook was allowed
    to have entries at all.

    Args:
        domain: Domain name.
        lessons: List of lesson strings to evaluate.

    Returns:
        Formatted prompt string for curator evaluation.
    """
    lessons_text = ""
    for i, lesson in enumerate(lessons, 1):
        lessons_text += f"{i}. {lesson}\n"

    prompt = f"""You are a Curator screening reading strategies proposed for the
{domain} playbook. The task is multiple-choice reading comprehension: a passage,
a question about it, and four options.

Mark a strategy GENERIC when it is advice about attitude or effort, and could
have been written without seeing any question. Mark it SPECIFIC when it names
something a reader can actually do to the passage or the options -- a comparison
to make, a kind of wrong option to recognise, a test to apply to a candidate
answer.

Strategies to evaluate:
{lessons_text}

Examples of GENERIC (mark is_generic=True):
- "Think carefully about the question."
- "Pay attention to details."
- "Read the passage thoroughly before answering."
- "Use the context provided." (without saying how)
- "Consider all the options." (without saying what to compare)

Examples of SPECIFIC (mark is_generic=False):
- "Reject an option that is true in general but is not stated in the passage."
- "When two options paraphrase the same sentence, prefer the one that keeps the
  direction of the cause, not the one that reverses it."
- "If an option answers a different question than the one asked, eliminate it
  even when the passage supports it."

A strategy does not need a formula, an equation or a number. This is reading
comprehension: naming the trap and the comparison to make is what specific
means here.

For each strategy, respond with:
Lesson [number]: is_generic=[True/False]
"""

    return prompt


def parse_curator_output(lesson_count: int, text: str) -> List[bool]:
    """
    Parse the Curator's output to extract is_generic flags for each lesson.

    Args:
        lesson_count: Number of lessons being evaluated.
        text: Raw output from the Curator model.

    Returns:
        List of boolean values indicating is_generic status for each lesson (True = generic).
        Defaults to False (not generic) if parsing fails for a lesson.
    """
    is_generic_flags = [False] * lesson_count  # Default to not generic
    lines = text.strip().split("\n")

    for line in lines:
        line = line.strip()
        # Look for pattern: "Lesson 1: is_generic=True" or "Lesson 1: is_generic=False"
        match = re.search(
            r"Lesson\s+(\d+)\s*:\s*is_generic\s*=\s*(True|False)", line, re.IGNORECASE
        )
        if match:
            lesson_num = int(match.group(1))
            is_generic = match.group(2).lower() == "true"
            if 1 <= lesson_num <= lesson_count:
                is_generic_flags[lesson_num - 1] = is_generic  # Convert to 0-based index

    return is_generic_flags


def choose_lessons_for_playbook(
    domain: str,
    lessons: List[str],
    existing_playbook: Playbook,
    min_length: int = 15,
) -> List[str]:
    """
    Filter candidate lessons before offering them to the playbook.

    Only length and genericness filtering happens here, and genericness means
    the same thing it means to the retention score. Deduplication is the
    playbook's job: it can compare a candidate against the incumbent and keep
    whichever is more specific, whereas dropping the candidate here would
    always preserve whichever lesson happened to arrive first.

    Args:
        domain: Domain name.
        lessons: List of candidate lesson strings.
        existing_playbook: Current playbook (retained for signature
            compatibility; duplicate resolution now lives in Playbook).
        min_length: Minimum character length for a lesson to be kept.

    Returns:
        Filtered list of lessons suitable for the playbook.
    """
    filtered = []

    for lesson in lessons:
        lesson = lesson.strip()

        if len(lesson) < min_length:
            continue

        # One definition of generic, the one delta is computed from. This used
        # to carry a second, shorter list of its own that disagreed with
        # GENERIC_PHRASES -- it included the bare words "consider" and
        # "remember", which appear in perfectly concrete reading strategies --
        # behind a `len(lesson.split()) < 5` guard that a 15-character minimum
        # meant almost never fired. The filter was very nearly dead code that
        # would have been wrong had it run.
        if compute_vagueness_score(lesson) >= _GENERIC_FLOOR:
            continue

        # Defer duplicate handling to the playbook, which resolves an overlap
        # by keeping the more specific text rather than always keeping the
        # incumbent. Dropping the lesson here would deny it that chance.
        filtered.append(lesson)

    return filtered
