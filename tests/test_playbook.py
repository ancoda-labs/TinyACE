"""Tests for playbook functionality."""

import tempfile
from pathlib import Path

import pytest

from edge_slm_ace.memory.playbook import (
    Playbook,
    PlaybookEntry,
    ScoringParams,
    compute_vagueness_score,
)


class TestVaguenessScore:
    """Tests for the vagueness scoring function."""

    def test_generic_phrase_is_vague(self):
        """Generic phrases should have high vagueness scores."""
        vague_texts = [
            "Think carefully about the problem",
            "Pay attention to details",
            "Remember to check your work",
            "Be thorough",
        ]
        for text in vague_texts:
            score = compute_vagueness_score(text)
            assert score > 0.3, f"'{text}' should be vague (score={score})"

    def test_specific_lesson_is_not_vague(self):
        """Specific lessons should have low vagueness scores."""
        specific_texts = [
            "For percentage calculations: divide by 100, then multiply by the base amount",
            "When calculating profit margin: (revenue - expenses) / revenue * 100",
            "If the question asks for net profit after tax, apply the tax rate to pre-tax profit",
        ]
        for text in specific_texts:
            score = compute_vagueness_score(text)
            assert score < 0.4, f"'{text}' should be specific (score={score})"

    def test_short_text_is_vague(self):
        """Very short text should be considered vague."""
        short_texts = ["Be good", "Try harder", "Think"]
        for text in short_texts:
            score = compute_vagueness_score(text)
            assert score >= 0.5, f"'{text}' is too short, should be vague (score={score})"


class TestPlaybookEntry:
    """Tests for PlaybookEntry scoring."""

    def test_entry_creation(self):
        """Test basic entry creation."""
        entry = PlaybookEntry(
            id="1",
            domain="finance",
            text="Calculate profit by subtracting expenses from revenue",
        )
        assert entry.id == "1"
        assert entry.domain == "finance"
        assert entry.success_count == 0
        assert entry.failure_count == 0
        assert entry.token_count > 0  # Auto-computed
        assert 0 <= entry.vagueness_score <= 1  # Auto-computed

    def test_score_formula_success_term(self):
        """Test that success increases score."""
        entry = PlaybookEntry(id="1", domain="test", text="Test lesson with some content")

        # Initial score (no uses)
        initial_score = entry.score(current_step=10)

        # Add successes
        entry.success_count = 5
        entry.failure_count = 0
        entry.last_used_at = 10

        score_with_success = entry.score(current_step=10)

        # Score should increase with successes
        assert score_with_success > initial_score

    def test_score_formula_failure_term(self):
        """Test that failures decrease score."""
        entry = PlaybookEntry(id="1", domain="test", text="Test lesson with some content")
        entry.last_used_at = 10

        # Add only failures
        entry.success_count = 0
        entry.failure_count = 5

        score_with_failures = entry.score(current_step=10)

        # Entry with only failures should have negative or low score
        assert score_with_failures < 0.5

    def test_score_formula_recency_decay(self):
        """Test that older entries have lower recency bonus."""
        entry = PlaybookEntry(id="1", domain="test", text="Test lesson with some content")
        entry.success_count = 3
        entry.failure_count = 1
        entry.last_used_at = 5

        # Score at step 5 (just used)
        score_fresh = entry.score(current_step=5)

        # Score at step 100 (old)
        score_old = entry.score(current_step=100)

        # Fresh entry should have higher score due to recency
        assert score_fresh > score_old

    def test_score_formula_vagueness_penalty(self):
        """Test that vague entries have lower scores."""
        specific_entry = PlaybookEntry(
            id="1",
            domain="test",
            text="For percentage: divide by 100, multiply by base amount = result",
        )

        vague_entry = PlaybookEntry(
            id="2",
            domain="test",
            text="Think carefully",
        )

        # Both at same step, no usage history
        assert specific_entry.score() > vague_entry.score()

    def test_score_with_custom_params(self):
        """Test scoring with custom hyperparameters."""
        entry = PlaybookEntry(id="1", domain="test", text="Test lesson here")
        entry.success_count = 5
        entry.failure_count = 2
        entry.last_used_at = 10

        # Default params
        default_score = entry.score(current_step=10)

        # High alpha (emphasize success)
        high_alpha = ScoringParams(alpha=2.0, beta=0.1)
        high_alpha_score = entry.score(current_step=10, params=high_alpha)

        assert high_alpha_score > default_score


class TestPlaybook:
    """Tests for Playbook class."""

    def test_add_and_save(self):
        """Test adding entries and saving playbook."""
        playbook = Playbook()

        entry1 = playbook.add_entry("finance", "Calculate revenue before tax", step=1)
        entry2 = playbook.add_entry("finance", "Subtract expenses from revenue", step=2)
        entry3 = playbook.add_entry("medical", "Check blood pressure first", step=1)

        assert len(playbook.entries) == 3
        assert entry1.domain == "finance"
        assert entry2.domain == "finance"
        assert entry3.domain == "medical"

        # Save and load
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            temp_path = Path(f.name)

        try:
            playbook.save(temp_path)
            assert temp_path.exists()

            loaded = Playbook.load(temp_path)
            assert len(loaded.entries) == 3
            assert loaded.entries[0].text == "Calculate revenue before tax"
        finally:
            temp_path.unlink()

    def test_deduplication(self):
        """Test that duplicate entries are not added."""
        playbook = Playbook()

        entry1 = playbook.add_entry("finance", "Calculate revenue", step=1)
        entry2 = playbook.add_entry("finance", "Calculate revenue", step=2)  # Duplicate

        assert len(playbook.entries) == 1
        assert entry1.id == entry2.id  # Should return same entry

    def test_get_top_k(self):
        """Test getting top-k entries for a domain."""
        playbook = Playbook()

        # Add entries with different success rates
        for i in range(5):
            entry = playbook.add_entry("finance", f"Strategy number {i} with details", step=i)
            entry.success_count = 5 - i  # Decreasing success
            entry.failure_count = i
            entry.last_used_at = i

        top_3 = playbook.get_top_k("finance", k=3, current_step=10)
        assert len(top_3) == 3

        # Should be sorted by score (descending)
        scores = [e.score(current_step=10) for e in top_3]
        assert scores == sorted(scores, reverse=True)

    def test_get_top_entries_for_budget(self):
        """Test token-budgeted entry retrieval."""
        playbook = Playbook()

        # Add entries of different sizes
        playbook.add_entry("finance", "Short", step=1)  # ~2 tokens
        playbook.add_entry("finance", "Medium length lesson with some details", step=2)  # ~9 tokens
        playbook.add_entry(
            "finance",
            "A very long and detailed lesson with many words and specific instructions about financial calculations",
            step=3,
        )  # ~20 tokens

        # Get entries within small budget
        entries = playbook.get_top_entries_for_budget("finance", token_budget=15, current_step=5)

        # Should not exceed budget
        total_tokens = sum(e.token_count for e in entries)
        assert total_tokens <= 15

    def test_record_feedback_only_for_used_entries(self):
        """Test that feedback is only recorded for entries that were used."""
        playbook = Playbook()

        entry = playbook.add_entry("finance", "Test strategy with content", step=1)

        # New entry should have no feedback
        assert entry.success_count == 0
        assert entry.failure_count == 0

        # Record feedback (simulating actual use)
        playbook.record_feedback(entry.id, helpful=True)
        assert entry.success_count == 1
        assert entry.failure_count == 0

        playbook.record_feedback(entry.id, helpful=False)
        assert entry.success_count == 1
        assert entry.failure_count == 1

    def test_token_budget_eviction(self):
        """Test that low-score entries are evicted when over budget."""
        # Create playbook with small token budget
        playbook = Playbook(token_budget=50)

        # Add entries until we hit the budget
        entries = []
        for i in range(5):
            entry = playbook.add_entry(
                "finance",
                f"Lesson {i}: This is a detailed lesson with specific content about topic {i}",
                step=i,
                enforce_budget=True,
            )
            entries.append(entry)

            # Give some entries better scores
            if i < 2:
                entry.success_count = 5
            else:
                entry.failure_count = 3

        # Add one more entry that should trigger eviction
        playbook.add_entry(
            "finance",
            "New lesson: Another detailed lesson that should trigger eviction of low-score entries",
            step=10,
            enforce_budget=True,
        )

        # Total tokens should be within budget (or close)
        domain_tokens = playbook.get_domain_tokens("finance")
        assert domain_tokens <= playbook.token_budget + 50  # Some tolerance

    def test_prune_keeps_top_entries(self):
        """Test that pruning keeps highest-scoring entries."""
        playbook = Playbook()

        # Add entries with different scores
        for i in range(10):
            entry = playbook.add_entry("finance", f"Strategy {i} with some content", step=i)
            entry.success_count = 10 - i  # Higher index = lower score
            entry.last_used_at = i

        assert len(playbook.entries) == 10

        # Prune to top 3
        removed = playbook.prune(max_entries_per_domain=3, current_step=15)

        assert len(playbook.entries) == 3
        assert removed == 7

        # Remaining entries should have highest success counts
        success_counts = [e.success_count for e in playbook.entries]
        assert all(count >= 8 for count in success_counts)

    def test_get_stats(self):
        """Test playbook statistics."""
        playbook = Playbook()

        # Add some entries
        for i in range(3):
            entry = playbook.add_entry("finance", f"Lesson {i} about finance topics", step=i)
            entry.success_count = 2
            entry.failure_count = 1

        stats = playbook.get_stats("finance")

        assert stats["num_entries"] == 3
        assert stats["total_tokens"] > 0
        assert stats["avg_success_rate"] > 0
        assert "finance" in stats["domains"]


class TestPlaybookLegacyCompatibility:
    """Tests for backward compatibility with legacy playbook format."""

    def test_load_legacy_entry(self):
        """Test loading entries from legacy format."""
        legacy_dict = {
            "id": "1",
            "domain": "finance",
            "text": "Test lesson",
            "helpful_count": 5,
            "harmful_count": 2,
            "created_at": 1234567890.0,
            "last_seen_step": 10,
            "is_generic": False,
        }

        entry = PlaybookEntry.from_dict(legacy_dict)

        assert entry.id == "1"
        assert entry.success_count == 5  # Migrated from helpful_count
        assert entry.failure_count == 2  # Migrated from harmful_count

    def test_save_includes_legacy_fields(self):
        """Test that saved entries include legacy fields."""
        entry = PlaybookEntry(
            id="1",
            domain="finance",
            text="Test lesson",
        )
        entry.success_count = 3
        entry.failure_count = 1

        entry_dict = entry.to_dict()

        # Should include both new and legacy fields
        assert "success_count" in entry_dict
        assert "failure_count" in entry_dict
        assert "helpful_count" in entry_dict
        assert "harmful_count" in entry_dict


# Legacy test names for backward compatibility
def test_playbook_add_and_save():
    """Test adding entries and saving playbook."""
    TestPlaybook().test_add_and_save()


def test_playbook_prune():
    """Test pruning playbook to top entries."""
    TestPlaybook().test_prune_keeps_top_entries()


def test_playbook_get_top_k():
    """Test getting top-k entries for a domain."""
    TestPlaybook().test_get_top_k()


def test_playbook_record_feedback():
    """Test recording feedback for entries."""
    TestPlaybook().test_record_feedback_only_for_used_entries()


class TestAblationFlags:
    """Tests for retention scoring ablation flags."""

    def test_disable_vagueness_penalty(self):
        """When disable_vagueness_penalty=True, vagueness term should be zero."""
        entry = PlaybookEntry(
            id="1",
            domain="test",
            text="Think carefully",  # Vague text
            success_count=1,
            failure_count=0,
            last_used_at=0,
        )

        # With vagueness penalty enabled (default)
        params_default = ScoringParams()
        score_with_penalty = entry.score(current_step=1, params=params_default)

        # With vagueness penalty disabled
        params_no_vagueness = ScoringParams(disable_vagueness_penalty=True)
        score_without_penalty = entry.score(current_step=1, params=params_no_vagueness)

        # Score without penalty should be higher (less negative)
        assert (
            score_without_penalty > score_with_penalty
        ), "Disabling vagueness penalty should increase score"

    def test_disable_recency_decay(self):
        """When disable_recency_decay=True, recency term should be zero."""
        entry = PlaybookEntry(
            id="1",
            domain="test",
            text="Test lesson",
            success_count=1,
            failure_count=0,
            last_used_at=0,  # Used at step 0
        )

        # With recency enabled (default), recent entries get bonus
        params_default = ScoringParams()
        score_recent = entry.score(current_step=1, params=params_default)  # Recent

        # With recency disabled
        params_no_recency = ScoringParams(disable_recency_decay=True)
        score_no_recency = entry.score(current_step=1, params=params_no_recency)

        # Score without recency should be lower (no recency bonus)
        assert score_no_recency < score_recent, "Disabling recency should remove recency bonus"

    def test_disable_failure_penalty(self):
        """When disable_failure_penalty=True, failure term should be zero."""
        entry = PlaybookEntry(
            id="1",
            domain="test",
            text="Test lesson",
            success_count=1,
            failure_count=2,  # Has failures
            last_used_at=0,
        )

        # With failure penalty enabled (default)
        params_default = ScoringParams()
        score_with_penalty = entry.score(current_step=1, params=params_default)

        # With failure penalty disabled
        params_no_failure = ScoringParams(disable_failure_penalty=True)
        score_without_penalty = entry.score(current_step=1, params=params_no_failure)

        # Score without penalty should be higher (no failure penalty)
        assert (
            score_without_penalty > score_with_penalty
        ), "Disabling failure penalty should increase score"

    def test_fifo_memory_evicts_oldest_first(self):
        """When fifo_memory=True, the oldest entry must be evicted first."""
        import time

        now = time.time()
        entry1 = PlaybookEntry(
            id="1",
            domain="test",
            text="First entry",
            created_at=now - 10,  # Older
        )
        entry2 = PlaybookEntry(
            id="2",
            domain="test",
            text="Second entry",
            created_at=now,  # Newer
        )

        params_fifo = ScoringParams(fifo_memory=True)

        key1 = entry1.eviction_key(current_step=1, params=params_fifo)
        key2 = entry2.eviction_key(current_step=1, params=params_fifo)

        # Eviction sorts ascending and drops the front, so first-in must
        # have the lower key.
        assert key1 < key2, "In FIFO mode, the older entry must be evicted first"

    def test_fifo_memory_does_not_change_retrieval(self):
        """FIFO is an eviction policy; retrieval ranking stays score-based."""
        import time

        now = time.time()
        strong = PlaybookEntry(
            id="1",
            domain="test",
            text="Use formula F = m * a for force.",
            success_count=9,
            failure_count=0,
            created_at=now - 10,
        )
        weak = PlaybookEntry(
            id="2",
            domain="test",
            text="Use formula F = m * a for force.",
            success_count=0,
            failure_count=9,
            created_at=now,
        )

        params_fifo = ScoringParams(fifo_memory=True)

        assert strong.retrieval_key(current_step=1, params=params_fifo) > weak.retrieval_key(
            current_step=1, params=params_fifo
        ), "FIFO must not degrade retrieval ranking to insertion order"

    def test_fifo_playbook_evicts_oldest_entry(self):
        """End-to-end: adding over budget under FIFO drops the oldest entry."""
        playbook = Playbook(
            token_budget=25,
            store_token_capacity=25,
            scoring_params=ScoringParams(fifo_memory=True),
        )
        first = playbook.add_entry("test", "First lesson about acceleration " * 2, step=1)
        second = playbook.add_entry("test", "Second lesson about velocity " * 2, step=2)
        playbook.add_entry("test", "Third lesson about momentum " * 2, step=3)

        remaining = {e.id for e in playbook.entries}
        assert first.id not in remaining, "Oldest entry should have been evicted first"
        assert second.id in remaining, "Newer entries should be retained under FIFO"


class TestDuplicateResolution:
    """A vague incumbent must not block a more specific lesson."""

    VAGUE = "Check the units."
    SPECIFIC = (
        "For density problems, check the units: convert g/cm3 to kg/m3 by " "multiplying by 1000."
    )

    def test_specific_lesson_replaces_vague_one(self):
        playbook = Playbook()
        playbook.add_entry("science", self.VAGUE, step=1)
        playbook.add_entry("science", self.SPECIFIC, step=2)

        assert len(playbook.entries) == 1, "Overlapping lessons should still merge"
        assert (
            playbook.entries[0].text == self.SPECIFIC
        ), "The more specific phrasing should have won"

    def test_vague_lesson_does_not_replace_specific_one(self):
        playbook = Playbook()
        playbook.add_entry("science", self.SPECIFIC, step=1)
        playbook.add_entry("science", self.VAGUE, step=2)

        assert len(playbook.entries) == 1
        assert playbook.entries[0].text == self.SPECIFIC

    def test_merging_preserves_feedback_history(self):
        """Adopting better text must not discard accumulated statistics."""
        playbook = Playbook()
        entry = playbook.add_entry("science", self.VAGUE, step=1)
        playbook.record_feedback(entry.id, helpful=True)
        playbook.record_feedback(entry.id, helpful=True)

        playbook.add_entry("science", self.SPECIFIC, step=2)

        assert playbook.entries[0].success_count == 2
        assert playbook.entries[0].id == entry.id

    def test_token_count_follows_the_adopted_text(self):
        playbook = Playbook()
        playbook.add_entry("science", self.VAGUE, step=1)
        playbook.add_entry("science", self.SPECIFIC, step=2)

        entry = playbook.entries[0]
        assert entry.token_count == entry._estimate_tokens()


class TestIdAllocation:
    def test_non_numeric_ids_do_not_cause_collisions(self):
        """A legacy playbook with string ids must not remint id '1'."""
        playbook = Playbook(
            entries=[
                PlaybookEntry(id="legacy-a", domain="science", text="Lesson one here."),
                PlaybookEntry(id="1", domain="science", text="Lesson two here."),
            ]
        )
        new_entry = playbook.add_entry("science", "A completely different lesson.", step=1)

        ids = [e.id for e in playbook.entries]
        assert len(ids) == len(set(ids)), f"Duplicate ids minted: {ids}"
        assert new_entry.id != "1"

    def test_feedback_reaches_the_intended_entry(self):
        playbook = Playbook(
            entries=[
                PlaybookEntry(id="legacy-a", domain="science", text="Lesson one here."),
                PlaybookEntry(id="1", domain="science", text="Lesson two here."),
            ]
        )
        new_entry = playbook.add_entry("science", "A completely different lesson.", step=1)
        playbook.record_feedback(new_entry.id, helpful=True)

        assert new_entry.success_count == 1
        assert sum(e.success_count for e in playbook.entries) == 1


class TestEntrySerialisation:
    def test_from_dict_does_not_mutate_input(self):
        payload = {"id": "1", "domain": "science", "text": "Some lesson text here."}
        snapshot = dict(payload)

        PlaybookEntry.from_dict(payload)

        assert payload == snapshot, "from_dict must not rewrite the caller's dict"

    def test_from_dict_tolerates_unknown_fields(self):
        entry = PlaybookEntry.from_dict(
            {
                "id": "1",
                "domain": "science",
                "text": "Some lesson text here.",
                "field_from_a_newer_version": 123,
            }
        )
        assert entry.id == "1"

    def test_round_trip_preserves_computed_fields(self):
        original = PlaybookEntry(id="1", domain="science", text="Think carefully.")
        restored = PlaybookEntry.from_dict(original.to_dict())

        assert restored.vagueness_score == original.vagueness_score
        assert restored.token_count == original.token_count


class TestStoreCapacityVsPromptBudget:
    """Keeping and showing are different budgets, or ranking cannot select."""

    def _entry_text(self, i):
        return f"Lesson number {i} explains how to convert unit {i} into unit {i + 1}."

    def test_store_capacity_defaults_larger_than_prompt_budget(self):
        playbook = Playbook(token_budget=256)
        assert playbook.store_token_capacity > playbook.token_budget

    def test_explicit_capacity_is_respected(self):
        playbook = Playbook(token_budget=256, store_token_capacity=300)
        assert playbook.store_token_capacity == 300

    def test_retrieval_actually_selects_a_subset(self):
        """With a larger store than budget, some entries must be left out."""
        playbook = Playbook(token_budget=40, store_token_capacity=400)
        for i in range(12):
            playbook.add_entry("science", self._entry_text(i), step=i + 1)

        stored = playbook.get_domain_tokens("science")
        retrieved = playbook.get_top_entries_for_budget("science", token_budget=40)
        retrieved_tokens = sum(e.token_count for e in retrieved)

        assert stored > 40, "The store should hold more than one promptful"
        assert retrieved_tokens <= 40
        assert len(retrieved) < len(
            playbook.entries
        ), "Retrieval returned everything, so ranking selected nothing"

    def test_no_capacity_means_unbounded(self):
        playbook = Playbook()
        assert playbook.store_token_capacity is None
        for i in range(20):
            playbook.add_entry("science", self._entry_text(i), step=i + 1)
        assert len(playbook.entries) == 20


class TestVaguenessHeuristicTable:
    """
    Hand-labelled lessons, because the previous tests only exercised extremes.

    delta weights this term in the retention score and has a dedicated ablation
    arm, so the threshold has to survive ordinary prose. It did not: the
    formula check tested for the bare characters "=+-*/%^", so any hyphenated
    word or slash counted as a formula and earned a specificity credit.
    "Think carefully about the well-known question" -- a phrase from
    GENERIC_PHRASES, and the Reflector prompt's own first example of a bad
    lesson -- scored 0.25 against an is_generic threshold of 0.5.
    """

    GENERIC = [
        "Think carefully about the well-known question",
        "Pay attention to details",
        "Check your work",
        "Remember to check your work and/or ask again",
        "Be thorough in your analysis of the state-of-the-art",
        "Take your time and double check",
    ]

    SPECIFIC = [
        "For density problems, convert g/cm3 to kg/m3 by multiplying by 1000",
        "When calculating profit margin: (revenue - expenses) / revenue * 100",
        "For percentage calculations: divide by 100, then multiply by the base amount",
        "If the question asks for net profit after tax, apply the tax rate to pre-tax profit",
        "Photosynthesis converts CO2 and H2O into glucose using light energy",
    ]

    @pytest.mark.parametrize("text", GENERIC)
    def test_generic_advice_is_flagged(self, text):
        score = compute_vagueness_score(text)
        assert score > 0.5, f"'{text}' should be generic (score={score})"

    @pytest.mark.parametrize("text", SPECIFIC)
    def test_actionable_rules_are_not_flagged(self, text):
        score = compute_vagueness_score(text)
        assert score <= 0.5, f"'{text}' should be specific (score={score})"

    def test_a_hyphen_alone_is_not_a_formula(self):
        """The regression: prose credited as specific because it contains '-'."""
        hyphenated = compute_vagueness_score("Be careful with the well-known cases")
        plain = compute_vagueness_score("Be careful with the well known cases")
        assert hyphenated == plain

    def test_a_generic_phrase_with_a_real_procedure_is_still_specific(self):
        """The floor must not punish a concrete lesson for one stock phrase."""
        score = compute_vagueness_score(
            "For density problems, make sure to convert g/cm3 to kg/m3 by multiplying by 1000"
        )
        assert score <= 0.5


class TestPruneRequiresAStep:
    """
    prune() defaulted current_step to 0, and the runner relied on that default.
    Every entry's age was then 0, so every entry received the identical recency
    bonus gamma -- a constant, which cannot change a ranking. Pruning ignored
    recency entirely, which is half of what the no-recency ablation is meant to
    be testing.
    """

    def test_step_is_required(self):
        playbook = Playbook()
        playbook.add_entry("science", "For wind questions, apply the Coriolis effect.", step=1)
        with pytest.raises(TypeError):
            playbook.prune(max_entries_per_domain=1)

    def test_recency_decides_between_otherwise_equal_entries(self):
        """With no feedback and equal vagueness, the recently used one stays."""
        playbook = Playbook()
        stale = playbook.add_entry("science", "Apply Coriolis deflection to wind direction.", 1)
        fresh = playbook.add_entry("science", "Apply Bernoulli to pressure differences.", 1)
        stale.last_used_at = 1
        fresh.last_used_at = 40

        playbook.prune(max_entries_per_domain=1, current_step=40)

        assert [e.id for e in playbook.entries] == [fresh.id]


class TestVaguenessOnReadingComprehension:
    """
    The lexicon was arithmetic vocabulary on a reading-comprehension task.

    A well-formed reading strategy -- the kind `build_reflector_prompt` asks
    for by name -- could not earn a specificity credit, so delta penalised it
    for being exactly what it was told to be. And the terms were matched as
    substrings, so "if" fired on *verify*, *specific* and *clarify*, "add" on
    *address*, "then" on *strengthen*: a large share of the credits the term
    awarded were accidents of spelling.
    """

    STRATEGIES = [
        "Reject an option that is true in general but is not stated in the passage",
        "Compare each option against what the passage explicitly states",
        "When two options paraphrase the same sentence, keep the one that does not "
        "reverse the direction of the cause",
        "If an option answers a different question than the one asked, eliminate it",
    ]

    @pytest.mark.parametrize("text", STRATEGIES)
    def test_a_reading_strategy_is_specific(self, text):
        score = compute_vagueness_score(text)
        assert score <= 0.5, f"'{text}' should be specific (score={score})"

    def test_arithmetic_lessons_are_still_specific(self):
        """Numbers and applied operators carry that signal, not a word list."""
        assert (
            compute_vagueness_score(
                "For percentage calculations: divide by 100, then multiply by the base"
            )
            <= 0.5
        )

    @pytest.mark.parametrize(
        "word", ["verify", "specific", "different", "clarify", "address", "strengthen"]
    )
    def test_a_term_inside_another_word_is_not_a_credit(self, word):
        """The substring bug: these words contain "if", "add" or "then"."""
        generic = f"Pay attention and {word} things"
        assert (
            compute_vagueness_score(generic) > 0.5
        ), f"'{word}' must not earn a specificity credit"
