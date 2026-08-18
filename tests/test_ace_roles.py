"""Tests for ACE role-based prompts and parsing."""

import pytest

from edge_slm_ace.core.ace_roles import (
    build_generator_prompt,
    build_self_refine_critique_prompt,
    build_self_refine_rewrite_prompt,
    parse_used_strategies,
)
from edge_slm_ace.memory.playbook import Playbook

from edge_slm_ace.core.ace_roles import (
    parse_generator_output,
    parse_reflector_output_to_lessons,
    choose_lessons_for_playbook,
)
from edge_slm_ace.memory.playbook import Playbook


class TestParseGeneratorOutput:
    """Tests for parse_generator_output robustness."""
    
    def test_standard_format(self):
        """Test parsing standard Reasoning: / Answer: format."""
        text = """Reasoning:
First, I calculate the total revenue which is $100,000.
Then I subtract expenses of $30,000.

Answer:
70000"""
        answer, reasoning = parse_generator_output(text)
        
        assert answer == "70000"
        assert reasoning is not None
        assert "revenue" in reasoning.lower()
    
    def test_answer_only(self):
        """Test parsing when there's only an answer."""
        text = "Answer: 42"
        answer, reasoning = parse_generator_output(text)
        
        assert answer == "42"
        assert reasoning is None
    
    def test_no_structure(self):
        """Test parsing unstructured text."""
        text = "The result is 12345."
        answer, reasoning = parse_generator_output(text)
        
        # Should extract something reasonable
        assert answer is not None
        assert len(answer) > 0
    
    def test_numeric_answer_extraction(self):
        """Test extracting numeric answers from text."""
        # Test with explicit answer marker
        text = """Let me calculate...
Revenue = $100,000
Expenses = $30,000
Profit = $100,000 - $30,000 = $70,000

Answer: $70,000"""
        
        answer, reasoning = parse_generator_output(text)
        
        # Should extract the answer after the marker
        assert answer is not None
        assert "70" in answer or "70000" in answer
    
    def test_with_therefore(self):
        """Test parsing with 'therefore' transition."""
        text = """I need to calculate the profit.
Revenue is 100 and expenses are 40.
Therefore, profit = 60."""
        
        answer, reasoning = parse_generator_output(text)
        
        assert "60" in answer
    
    def test_final_answer_marker(self):
        """Test 'Final Answer:' format."""
        text = """Step 1: Calculate revenue
Step 2: Subtract expenses
Final Answer: 25500"""
        
        answer, reasoning = parse_generator_output(text)
        
        assert answer == "25500"
    
    def test_result_marker(self):
        """Test 'Result:' format."""
        text = """Calculation complete.
Result: 42%"""
        
        answer, reasoning = parse_generator_output(text)
        
        assert "42" in answer
    
    def test_empty_input(self):
        """Test handling of empty input."""
        assert parse_generator_output("") == ("", None)
        assert parse_generator_output("   ") == ("", None)
    
    def test_malformed_but_has_answer(self):
        """Test handling of malformed output that still contains an answer."""
        text = """This is some gibberish output
that doesn't follow any format
answer 12345
more gibberish"""
        
        answer, reasoning = parse_generator_output(text)
        
        # Should still extract something
        assert answer is not None
        assert len(answer) > 0
    
    def test_answer_with_currency(self):
        """Test parsing answers with currency symbols."""
        text = "Answer: $1,234.56"
        answer, reasoning = parse_generator_output(text)
        
        assert "$1,234.56" in answer or "1234.56" in answer
    
    def test_answer_with_percentage(self):
        """Test parsing answers with percentage."""
        text = "Answer: 45%"
        answer, reasoning = parse_generator_output(text)
        
        assert "45" in answer


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
        lessons = ["Be careful", "For revenue calculations, always subtract expenses before tax", "Check"]
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

        assert len(playbook.entries) == 2, \
            f"Expected the duplicate to merge, got {[e.text for e in playbook.entries]}"
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


class TestSelfRefinePrompts:
    """The scored prediction must never be generated from the label."""

    def test_critique_prompt_excludes_ground_truth_by_default(self):
        prompt = build_self_refine_critique_prompt(
            domain="science",
            question="What gas do plants absorb?",
            context=None,
            initial_answer="oxygen",
        )
        assert "carbon dioxide" not in prompt
        assert "Correct Answer" not in prompt

    def test_rewrite_prompt_excludes_ground_truth_by_default(self):
        prompt = build_self_refine_rewrite_prompt(
            domain="science",
            question="What gas do plants absorb?",
            context=None,
            initial_answer="oxygen",
            critique="I confused absorption with release.",
        )
        assert "carbon dioxide" not in prompt
        assert "Correct Answer" not in prompt

    def test_oracle_mode_is_opt_in_and_explicit(self):
        prompt = build_self_refine_rewrite_prompt(
            domain="science",
            question="What gas do plants absorb?",
            context=None,
            initial_answer="oxygen",
            critique="I confused absorption with release.",
            ground_truth="carbon dioxide",
        )
        assert "Correct Answer: carbon dioxide" in prompt


class TestParseUsedStrategies:
    """Credit must go to the lessons the model says it applied."""

    def test_parses_a_citation_list(self):
        text = "Reasoning: ...\nAnswer: B\nUsed strategies: 1, 3"
        assert parse_used_strategies(text, 5) == [0, 2]

    def test_singular_wording_is_accepted(self):
        assert parse_used_strategies("Used strategy: 2", 5) == [1]

    def test_explicit_none_is_an_empty_credit_set(self):
        """Distinct from 'no citation': the model said nothing helped."""
        assert parse_used_strategies("Used strategies: none", 5) == []

    def test_missing_citation_returns_none(self):
        """None means 'no attribution available', not 'nothing was used'."""
        assert parse_used_strategies("Answer: B", 5) is None

    def test_out_of_range_numbers_are_rejected(self):
        assert parse_used_strategies("Used strategies: 9", 5) is None

    def test_duplicates_are_collapsed(self):
        assert parse_used_strategies("Used strategies: 2, 2, 3", 5) == [1, 2]

    def test_only_the_citation_line_is_read(self):
        text = "Used strategies: 1\nThen I checked equation 4 again."
        assert parse_used_strategies(text, 5) == [0]

    def test_no_strategies_offered_means_nothing_to_cite(self):
        assert parse_used_strategies("Used strategies: 1", 0) is None


class TestGeneratorCitationPrompt:
    def test_prompt_requests_citations_when_strategies_exist(self):
        playbook = Playbook()
        playbook.add_entry("science", "For wind questions, apply the Coriolis effect.", step=1)

        prompt = build_generator_prompt("science", playbook, "Why does wind curve?")

        assert "Used strategies:" in prompt

    def test_prompt_omits_citation_block_when_playbook_is_empty(self):
        """The control arm shows no strategies, so it must not ask for them."""
        prompt = build_generator_prompt("science", Playbook(), "Why does wind curve?")

        assert "Used strategies:" not in prompt
