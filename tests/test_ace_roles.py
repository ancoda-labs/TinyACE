"""Tests for ACE role-based prompts and parsing."""

import pytest

from edge_slm_ace.core.ace_roles import (
    parse_curator_output,
    choose_lessons_for_playbook,
    build_curator_prompt,
    parse_reflector_output_to_lessons,
)
from edge_slm_ace.memory.playbook import Playbook


class TestParseReflectorOutput:
    """Tests for parse_reflector_output_to_lessons."""

    def test_bullet_points(self):
        """Test parsing bullet-pointed lessons."""
        text = """
- Calculate revenue before expenses
- Always subtract costs from income
- Remember to apply tax rate at the end
"""
        lessons = parse_reflector_output_to_lessons(text)

        assert len(lessons) >= 3
        assert any("revenue" in l.lower() for l in lessons)

    def test_numbered_list(self):
        """Test that non-bullet text is also captured."""
        text = """
1. First calculate the base amount
2. Then apply the percentage
3. Finally round to nearest integer
"""
        lessons = parse_reflector_output_to_lessons(text)

        # Should capture at least some lessons
        assert len(lessons) > 0

    def test_mixed_format(self):
        """Test parsing mixed format lessons."""
        text = """
Here are the lessons learned:
- Use the formula: profit = revenue - expenses
• For percentages, divide by 100 first
Also remember to check units
"""
        lessons = parse_reflector_output_to_lessons(text)

        assert len(lessons) >= 2


class TestChooseLessonsForPlaybook:
    """Tests for lesson filtering and deduplication."""

    def test_filters_short_lessons(self):
        """Test that very short lessons are filtered."""
        lessons = [
            "Be careful",
            "For revenue calculations, always subtract expenses before tax",
            "Check",
        ]
        playbook = Playbook()

        filtered = choose_lessons_for_playbook("finance", lessons, playbook, min_length=15)

        # Only the long lesson should remain
        assert len(filtered) == 1
        assert "revenue" in filtered[0].lower()

    def test_duplicates_do_not_grow_the_playbook(self):
        """Duplicate resolution now happens in Playbook.add_entry.

        This asserts the invariant that matters end to end -- a repeated
        lesson must not create a second entry -- rather than which layer
        performs the check.
        """
        playbook = Playbook()
        playbook.add_entry("finance", "Calculate revenue before expenses", step=1)

        lessons = [
            "Calculate revenue before expenses",  # Exact duplicate
            "For tax calculations, apply rate to pre-tax amount",  # New
        ]

        for lesson in choose_lessons_for_playbook("finance", lessons, playbook):
            playbook.add_entry("finance", lesson, step=2)

        assert (
            len(playbook.entries) == 2
        ), f"Expected the duplicate to merge, got {[e.text for e in playbook.entries]}"
        assert any("tax" in e.text.lower() for e in playbook.entries)

    def test_filters_generic_advice(self):
        """Test that generic advice is filtered."""
        lessons = [
            "Think carefully about the problem",  # Generic
            "For percentage calculation: divide by 100, then multiply",  # Specific
            "Pay attention to details",  # Generic
        ]
        playbook = Playbook()

        filtered = choose_lessons_for_playbook("finance", lessons, playbook)

        # Only specific lesson should remain
        assert any("percentage" in l.lower() for l in filtered)


# Run basic tests when executing this file directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestCuratorScreensReadingStrategies:
    """
    The Curator prompt was never retargeted from the arithmetic task.

    It demanded "concrete formulas, equations, procedures", illustrated
    SPECIFIC with compound interest and percentage conversion, and closed with
    "if a lesson doesn't contain specific formulas, procedures, or concrete
    steps, mark it as generic". Belebele has no arithmetic in it, so on this
    task it rejected the strategies `build_reflector_prompt` explicitly asks
    for -- and `use_curator` is on by default, which left the Curator arm and
    the no-Curator arm differing mainly in whether the playbook could have
    entries at all.
    """

    def test_the_prompt_is_about_passages_not_formulas(self):
        prompt = build_curator_prompt("belebele_ne", ["Some strategy"]).lower()
        assert "passage" in prompt
        assert "option" in prompt
        assert "does not need a formula" in prompt

    def test_the_prompt_does_not_demand_arithmetic(self):
        prompt = build_curator_prompt("belebele_ne", ["Some strategy"]).lower()
        for demand in ("compound interest", "percentage to decimal", "p(1 + r/n)"):
            assert demand not in prompt

    def test_a_reading_strategy_survives_the_pre_filter(self):
        """choose_lessons_for_playbook screens before the Curator ever sees it."""
        lessons = [
            "Think carefully about the question and the options",
            "Reject an option that is true in general but is not stated in the passage",
            "Pay attention to details",
        ]
        kept = choose_lessons_for_playbook("belebele_ne", lessons, Playbook())

        assert kept == ["Reject an option that is true in general but is not stated in the passage"]

    def test_generic_means_one_thing(self):
        """The filter used its own shorter list, which disagreed with delta."""
        from edge_slm_ace.memory.playbook import compute_vagueness_score

        for lesson in ("Consider what the passage states about the claim",):
            assert compute_vagueness_score(lesson) < 0.6
            assert choose_lessons_for_playbook("d", [lesson], Playbook()) == [lesson]


class TestCuratorParseFailureKeepsTheLesson:
    """What an unreadable Curator verdict decides, and that it is deliberate.

    `parse_curator_output` seeds its flags with False, so a verdict it cannot
    parse keeps every lesson -- the Curator goes quiet rather than emptying the
    playbook. Seeding True is a one-character change with the opposite effect
    and no test objected to it, which `make mutants` found.

    It is not a hypothetical input: a small or randomly-initialised model
    answers the Curator with prose that matches nothing, and the CI smoke test
    on tiny-gpt2 produces exactly that.
    """

    def test_unparseable_output_keeps_every_lesson(self):
        assert parse_curator_output(3, "blah blah nonsense") == [False, False, False]

    def test_empty_output_keeps_every_lesson(self):
        assert parse_curator_output(2, "") == [False, False]

    def test_a_partial_verdict_only_decides_the_lessons_it_names(self):
        """One parsed line must not imply anything about the others."""
        assert parse_curator_output(3, "Lesson 2: is_generic=True") == [False, True, False]

    def test_an_out_of_range_lesson_number_is_ignored(self):
        assert parse_curator_output(2, "Lesson 7: is_generic=True") == [False, False]
        assert parse_curator_output(2, "Lesson 0: is_generic=True") == [False, False]

    def test_the_verdict_is_read_the_right_way_round(self):
        assert parse_curator_output(1, "Lesson 1: is_generic=True") == [True]
        assert parse_curator_output(1, "Lesson 1: is_generic=False") == [False]


class TestReflectorParserRejectsNonLessons:
    """
    Anything longer than ten characters became a lesson, bullets or not.

    Two things got in that way: the sentence introducing the list, and
    degenerate repetition from a weak model. The second is the worse of the
    two -- a 200-word run of one token contains no generic phrase, so it scores
    vagueness 0.0 and ranks as the most specific entry in the playbook. The CI
    smoke test produces exactly that, which is where this came from.
    """

    def test_the_line_introducing_the_list_is_not_a_lesson(self):
        text = (
            "Here are two reading strategies that would have helped:\n"
            "- Reject an option that is true in general but not stated in the passage\n"
            "- Check whether the option answers the question that was actually asked\n"
        )
        lessons = parse_reflector_output_to_lessons(text)

        assert len(lessons) == 2
        assert not any(l.lower().startswith("here are") for l in lessons)

    def test_a_header_line_is_not_a_lesson(self):
        text = "Strategies:\n- Compare each option against the wording of the passage\n"
        assert parse_reflector_output_to_lessons(text) == [
            "Compare each option against the wording of the passage"
        ]

    def test_degenerate_repetition_is_rejected(self):
        assert parse_reflector_output_to_lessons("- " + "factors " * 40) == []

    def test_free_text_is_still_read_when_there_are_no_bullets(self):
        text = "Check whether the option is actually stated in the passage before picking it"
        assert parse_reflector_output_to_lessons(text) == [text]

    def test_a_fragment_is_not_a_lesson(self):
        assert parse_reflector_output_to_lessons("- yes\n- ok then") == []

    def test_numbering_is_stripped(self):
        text = "1. Compare each option against what the passage explicitly states\n"
        assert parse_reflector_output_to_lessons(text) == [
            "Compare each option against what the passage explicitly states"
        ]
