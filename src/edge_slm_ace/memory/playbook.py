"""ACE-style playbook for storing and managing domain-specific strategies.

This module implements the TinyACE working memory system with:
- Retention scoring based on success/failure rates, recency, and vagueness
- Token-budgeted eviction (strategic forgetting)
- Support for both ace_full (top-k) and ace_working_memory modes
"""

import json
import math
import re
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from edge_slm_ace.memory.relevance import LessonRelevance, blend

# Default hyperparameters for retention scoring
# These match the formal equation:
# S(l_i, t) = α·(N_succ/N_used+ε) - β·(N_fail/N_used+ε) + γ·exp(-λ·(t-t_last)) - δ·V(l_i)
DEFAULT_ALPHA = 1.0  # Weight for success ratio
DEFAULT_BETA = 0.5  # Weight for failure ratio (penalty)
DEFAULT_GAMMA = 0.3  # Weight for recency bonus
DEFAULT_DELTA = 0.4  # Weight for vagueness penalty
DEFAULT_LAMBDA = 0.05  # Decay rate for recency (smaller = slower decay)
DEFAULT_EPSILON = 1.0  # Smoothing constant to avoid division by zero

# How much larger the store is than the prompt budget, when not set explicitly.
# Must exceed 1 or retrieval has nothing to choose between.
DEFAULT_STORE_CAPACITY_MULTIPLIER = 4


@dataclass
class ScoringParams:
    """Hyperparameters for the retention scoring formula."""

    alpha: float = DEFAULT_ALPHA  # Success ratio weight
    beta: float = DEFAULT_BETA  # Failure ratio weight (penalty)
    gamma: float = DEFAULT_GAMMA  # Recency bonus weight
    delta: float = DEFAULT_DELTA  # Vagueness penalty weight
    lambda_decay: float = DEFAULT_LAMBDA  # Recency decay rate
    epsilon: float = DEFAULT_EPSILON  # Smoothing constant
    # Ablation flags
    disable_vagueness_penalty: bool = False  # If True, set δ=0 (ignore vagueness term)
    disable_recency_decay: bool = False  # If True, set γ=0 (ignore recency term)
    disable_failure_penalty: bool = False  # If True, set β=0 (ignore failure term)
    fifo_memory: bool = False  # If True, evict oldest-first instead of lowest-score
    # Weight on question-lesson relevance when ranking for retrieval. 0.0
    # reproduces the original domain-only behaviour, where every question in a
    # run received the identical lesson list.
    relevance_weight: float = 0.5


# Generic phrases that indicate vague/unhelpful lessons
GENERIC_PHRASES = [
    "think carefully",
    "think step by step",
    "consider all perspectives",
    "be thorough",
    "pay attention",
    "make sure",
    "remember to",
    "check your work",
    "read carefully",
    "double check",
    "be careful",
    "take your time",
]


def _normalize_for_comparison(text: str) -> str:
    """
    Normalise a lesson for duplicate detection.

    Lowercases, strips punctuation and collapses whitespace. Raw containment
    was punctuation-sensitive, so "check the units." did not match "check the
    units:" and near-identical lessons accumulated as separate entries.
    """
    stripped = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    return re.sub(r"\s+", " ", stripped).strip()


# An operator counts as a formula only when it is applied to a number, or when
# it is an equals sign. Testing for the bare characters credited any hyphenated
# word ("well-known") and any slash ("and/or") as evidence of a formula, so
# "Think carefully about the well-known question" -- a phrase that is literally
# in GENERIC_PHRASES, and that the Reflector prompt gives as its first example
# of a bad lesson -- scored 0.25 against an is_generic threshold of 0.5.
_FORMULA_RE = re.compile(r"\d\s*[-+*/^=%]|[-+*/^=%]\s*\d|=")

# Terms that indicate a lesson names a procedure rather than an attitude.
#
# Two things were wrong with the previous list. It was arithmetic vocabulary
# (formula, multiply, percentage) on a reading-comprehension task, so the
# lessons the Reflector is asked for could never earn a specificity credit and
# were penalised by delta for being what they were told to be. And it was
# matched as a substring, so "if" fired on *verify*, *specific*, *different* and
# *clarify*, "add" on *address*, and "then" on *strengthen*. Roughly half the
# credits it awarded were accidents of spelling.
#
# The list is now reading-comprehension procedure plus the conditional markers
# that name a case, and it is matched on word boundaries. The arithmetic signal
# has not been lost: numbers and applied operators are scored separately, which
# is what actually identifies a quantitative lesson.
_SPECIFIC_TERMS = (
    "passage",
    "option",
    "options",
    "stated",
    "states",
    "explicit",
    "explicitly",
    "contradicts",
    "paraphrase",
    "paraphrases",
    "eliminate",
    "quote",
    "restates",
    "reverses",
    "distractor",
    "claim",
    "wording",
    "sentence",
    "if",
    "when",
    "then",
    "unless",
)

_SPECIFIC_RE = re.compile(
    r"\b(?:" + "|".join(sorted(_SPECIFIC_TERMS, key=len, reverse=True)) + r")\b"
)

# Floor for a lesson that contains a generic phrase and no specificity signal
# at all. Above the is_generic threshold, so specificity credits cannot excuse
# a lesson that is nothing but generic advice.
_GENERIC_FLOOR = 0.6


def compute_vagueness_score(text: str) -> float:
    """
    Compute a vagueness/genericness score for a lesson.

    Returns a score in [0, 1] where:
    - 0.0 = specific, actionable lesson
    - 1.0 = very vague/generic lesson

    Heuristics, in the order they are applied:
    - Very short text is generic.
    - A phrase from GENERIC_PHRASES is strong evidence of genericness.
    - Numbers, formulas applied to numbers, and procedural terms are evidence
      against it, and reduce the score -- but they cannot pull a lesson that is
      *only* generic advice below `_GENERIC_FLOOR`. Procedural terms are matched
      on word boundaries; as substrings, "if" fired on *verify* and *specific*.

    delta weights this term in the retention score and has its own ablation
    arm, so a threshold that a hyphen could flip was not measuring what the
    equation claims.

    Args:
        text: The lesson text to score.

    Returns:
        Vagueness score between 0 and 1.
    """
    text_lower = text.lower().strip()
    word_count = len(text.split())

    score = 0.0

    # Very short text is likely generic
    if word_count < 5:
        score += 0.5
    elif word_count < 10:
        score += 0.2

    # Check for generic phrases
    generic_count = sum(1 for phrase in GENERIC_PHRASES if phrase in text_lower)
    if generic_count > 0:
        # More generic phrases = higher vagueness
        score += min(0.5, generic_count * 0.25)

    # Check for specificity indicators (formulas, numbers, specific terms)
    has_numbers = any(c.isdigit() for c in text)
    has_formula = bool(_FORMULA_RE.search(text))
    has_specific_terms = bool(_SPECIFIC_RE.search(text_lower))

    # Reduce score for specific content
    if has_numbers:
        score -= 0.15
    if has_formula:
        score -= 0.15
    if has_specific_terms:
        score -= 0.1

    # A lesson carrying a generic phrase and nothing concrete stays generic,
    # whatever the length term contributed.
    #
    # A bare term does not count as concrete on its own. "Think carefully about
    # the question and the options" names an option and is still nothing but
    # advice; one topic noun was enough to lift it off the floor. Numbers and
    # applied operators are content by themselves; a procedural term counts only
    # in a lesson long enough to have said something with it.
    concrete = has_numbers or has_formula or (has_specific_terms and word_count >= 10)
    if generic_count and not concrete:
        score = max(score, _GENERIC_FLOOR)

    # Clamp to [0, 1]
    return max(0.0, min(1.0, score))


@dataclass
class PlaybookEntry:
    """A single entry in the ACE playbook.

    Attributes:
        id: Unique identifier for this entry.
        domain: Domain/task this lesson applies to (e.g., "finance", "medical").
        text: The lesson text.
        success_count: Number of times this entry was used and the answer was correct.
        failure_count: Number of times this entry was used and the answer was incorrect.
        created_at: Timestamp when this entry was created.
        last_used_at: Step counter when this entry was last used in a prompt.
        token_count: Estimated token count for this entry's text.
        vagueness_score: Cached vagueness score (0=specific, 1=vague).
    """

    id: str
    domain: str
    text: str
    success_count: int = 0
    failure_count: int = 0
    created_at: float = 0.0
    last_used_at: int = 0
    token_count: Optional[int] = None
    vagueness_score: Optional[float] = None
    # Legacy fields for backward compatibility
    helpful_count: int = 0
    harmful_count: int = 0
    last_seen_step: int = 0
    is_generic: bool = False

    def __post_init__(self):
        """Initialize computed fields."""
        if self.created_at == 0.0:
            self.created_at = datetime.now().timestamp()

        # None means "not computed yet". Using 0 as the sentinel meant a
        # genuinely non-vague entry (score 0.0) was silently rescored on every
        # load, and a one-token entry could never cache its count.
        if self.token_count is None:
            self.token_count = self._estimate_tokens()

        if self.vagueness_score is None:
            self.vagueness_score = compute_vagueness_score(self.text)

        # Update legacy field
        self.is_generic = self.vagueness_score > 0.5

    def _estimate_tokens(self, tokens_per_word: float = 1.3) -> int:
        """
        Approximate the token count from the word count.

        Only a fallback. `words * 1.3` disagrees with the real tokenizer by a
        wide margin on technical text, so a "256-token budget" measured this
        way is not 256 tokens. Call `recount_tokens()` with the run's
        tokenizer to make the budget mean what it says.
        """
        word_count = len(self.text.split())
        return max(1, int(word_count * tokens_per_word))

    def recount_tokens(self, tokenizer) -> int:
        """
        Recompute this entry's token count with a real tokenizer.

        Args:
            tokenizer: A HuggingFace tokenizer.

        Returns:
            The updated token count.
        """
        try:
            self.token_count = max(1, len(tokenizer.encode(self.text, add_special_tokens=False)))
        except Exception:
            self.token_count = self._estimate_tokens()
        return self.token_count

    def total_uses(self) -> int:
        """Return total number of times this entry was used."""
        return self.success_count + self.failure_count

    def score(
        self,
        current_step: int = 0,
        params: Optional[ScoringParams] = None,
    ) -> float:
        """
        Compute the retention score for this entry.

        Formula:
        S(l_i, t) = α·(N_succ/(N_used+ε)) - β·(N_fail/(N_used+ε))
                  + γ·exp(-λ·(t - t_last)) - δ·V(l_i)

        Ablation flags can disable individual terms:
        - disable_vagueness_penalty: Set δ=0
        - disable_recency_decay: Set γ=0
        - disable_failure_penalty: Set β=0

        Note: `fifo_memory` deliberately has no effect here. It is an
        *eviction* policy, not a retention score -- see `eviction_key()`.
        Retrieval ranking always uses this score so that the FIFO ablation
        isolates eviction order as the single changed variable.

        Args:
            current_step: Current step counter for recency calculation.
            params: Scoring hyperparameters (uses defaults if None).

        Returns:
            Retention score (higher = better, should be retained).
        """
        if params is None:
            params = ScoringParams()

        n_used = self.total_uses()

        # Success ratio term: α · (N_succ / (N_used + ε))
        success_term = params.alpha * (self.success_count / (n_used + params.epsilon))

        # Failure ratio term: β · (N_fail / (N_used + ε))
        # Disabled if disable_failure_penalty is True
        if params.disable_failure_penalty:
            failure_term = 0.0
        else:
            failure_term = params.beta * (self.failure_count / (n_used + params.epsilon))

        # Recency term: γ · exp(-λ · (t - t_last))
        # Disabled if disable_recency_decay is True
        if params.disable_recency_decay:
            recency_term = 0.0
        else:
            # max() already handles a step behind an entry's last use; the
            # old `if current_step > 0` guard added nothing numerically and
            # only made a missing step look intentional.
            age = max(0, current_step - self.last_used_at)
            recency_term = params.gamma * math.exp(-params.lambda_decay * age)

        # Vagueness penalty: δ · V(l_i)
        # Disabled if disable_vagueness_penalty is True
        if params.disable_vagueness_penalty:
            vagueness_term = 0.0
        else:
            vagueness_term = params.delta * self.vagueness_score

        # Final score
        return success_term - failure_term + recency_term - vagueness_term

    def retrieval_key(
        self,
        current_step: int = 0,
        params: Optional[ScoringParams] = None,
    ) -> float:
        """
        Ranking key for deciding which entries to show the Generator.

        Higher = better. Always the retention score, including under the
        FIFO ablation, so that ablation changes eviction only.
        """
        return self.score(current_step, params)

    def eviction_key(
        self,
        current_step: int = 0,
        params: Optional[ScoringParams] = None,
    ) -> float:
        """
        Ordering key for deciding which entries to drop. Lower = evicted first.

        Under `fifo_memory` this is the creation timestamp, so the oldest
        entry (smallest timestamp, hence lowest key) is evicted first --
        which is what first-in-first-out means. Otherwise it is the
        retention score, so the weakest entry is evicted first.
        """
        if params is None:
            params = ScoringParams()
        if params.fifo_memory:
            return self.created_at
        return self.score(current_step, params)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PlaybookEntry":
        """
        Create from a dictionary, tolerating entries written by older runs.

        Copies before filling defaults; the previous version mutated the
        caller's dict in place, so loading a playbook silently rewrote the
        JSON objects the caller still held.
        """
        data = dict(d)

        defaults = {
            "success_count": data.get("helpful_count", 0),
            "failure_count": data.get("harmful_count", 0),
            "token_count": None,
            "vagueness_score": None,
            "helpful_count": 0,
            "harmful_count": 0,
            "last_seen_step": data.get("last_used_at", 0),
            "is_generic": False,
        }

        for key, default_val in defaults.items():
            data.setdefault(key, default_val)

        # Drop unknown keys rather than raising, so a playbook written by a
        # newer version stays loadable.
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


class Playbook:
    """ACE-style playbook storing domain-specific strategies.

    Supports two modes:
    - ace_full: Top-k entries per query (unbounded playbook size)
    - ace_working_memory: Token-budgeted entries (limited context window)

    The playbook implements strategic forgetting via:
    - Retention scoring (success rate, recency, vagueness)
    - Token-budget eviction (remove lowest-score entries when over budget)
    """

    def __init__(
        self,
        entries: Optional[List[PlaybookEntry]] = None,
        token_budget: Optional[int] = None,
        scoring_params: Optional[ScoringParams] = None,
        tokenizer=None,
        store_token_capacity: Optional[int] = None,
    ):
        """
        Initialize playbook.

        Args:
            entries: Initial entries (optional).
            token_budget: Maximum total tokens for working memory mode (None = unlimited).
            scoring_params: Hyperparameters for retention scoring.
            tokenizer: Optional HuggingFace tokenizer. When supplied, entry
                token counts come from it rather than from the words * 1.3
                heuristic, so the token budget is denominated in the same
                units as the prompt it is meant to bound.
            store_token_capacity: How many tokens of lessons to KEEP. Distinct
                from token_budget, which is how many to SHOW.

                These were previously the same number, which made retrieval a
                no-op: eviction held the store at or below the budget, and
                retrieval then filled up to that same budget, so every
                surviving entry was always retrieved and ranking never
                selected anything. WM-256 versus WM-512 compared how many
                lessons survived, not which were chosen.

                Defaults to DEFAULT_STORE_CAPACITY_MULTIPLIER x token_budget so
                that selection actually selects.
        """
        self.entries: List[PlaybookEntry] = entries or []
        self.token_budget = token_budget
        if store_token_capacity is not None:
            self.store_token_capacity = store_token_capacity
        elif token_budget is not None:
            self.store_token_capacity = DEFAULT_STORE_CAPACITY_MULTIPLIER * token_budget
        else:
            self.store_token_capacity = None
        self.scoring_params = scoring_params or ScoringParams()
        self.tokenizer = tokenizer
        if tokenizer is not None:
            for entry in self.entries:
                entry.recount_tokens(tokenizer)
        self._next_id = self._compute_next_id()

    def _compute_next_id(self) -> int:
        """Return an id one past the highest numeric id currently in use."""
        max_id = 0
        for entry in self.entries:
            try:
                max_id = max(max_id, int(entry.id))
            except (ValueError, TypeError):
                # Non-numeric id: skip it for the numeric maximum, but the
                # uniqueness check in add_entry still guards against reuse.
                continue
        return max_id + 1

    @property
    def total_tokens(self) -> int:
        """Return total token count across all entries."""
        return sum(e.token_count for e in self.entries)

    def get_domain_tokens(self, domain: str) -> int:
        """Return total token count for a specific domain."""
        return sum(e.token_count for e in self.entries if e.domain == domain)

    @classmethod
    def load(cls, path: Path, token_budget: Optional[int] = None, tokenizer=None) -> "Playbook":
        """
        Load playbook from a JSONL file.

        Args:
            path: Path to JSONL file.
            token_budget: Token budget for working memory mode.
            tokenizer: Optional tokenizer for exact token counts.

        Returns:
            Playbook instance.
        """
        entries = []
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entry_dict = json.loads(line)
                            entries.append(PlaybookEntry.from_dict(entry_dict))
                        except (json.JSONDecodeError, TypeError) as e:
                            # Skip malformed entries
                            print(f"Warning: Skipping malformed playbook entry: {e}")
                            continue

        playbook = cls(entries, token_budget=token_budget, tokenizer=tokenizer)

        # Set next_id past every existing id. Non-numeric ids used to be
        # skipped entirely, so a playbook containing any of them reset the
        # counter to 1 and minted duplicates on the next add. record_feedback
        # returns on its first match, so the duplicate silently received the
        # feedback meant for the original.
        playbook._next_id = playbook._compute_next_id()

        return playbook

    def save(self, path: Path) -> None:
        """
        Save playbook to a JSONL file.

        Args:
            path: Path to save JSONL file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for entry in self.entries:
                f.write(json.dumps(entry.to_dict()) + "\n")

    def _rank_for_retrieval(
        self,
        domain: str,
        current_step: int,
        query: Optional[str] = None,
    ) -> List[PlaybookEntry]:
        """
        Order a domain's entries by how worth showing they are.

        Combines retention score with relevance to `query`. Without a query
        this is retention-only, which is what the original code did for every
        question in a run -- all sciq_* tasks share one domain, so the same
        lesson list was returned regardless of what was being asked.

        Args:
            domain: Domain to rank within.
            current_step: Step counter, for the recency term.
            query: The question being answered, if available.

        Returns:
            Entries ordered best-first.
        """
        domain_entries = [e for e in self.entries if e.domain == domain]
        if not domain_entries:
            return []

        retention = [e.retrieval_key(current_step, self.scoring_params) for e in domain_entries]

        relevance = None
        weight = getattr(self.scoring_params, "relevance_weight", 0.0)
        if query and weight > 0:
            try:
                relevance = LessonRelevance.get_instance().score(
                    query, [e.text for e in domain_entries]
                )
            except Exception:
                relevance = None  # fall back to retention-only ranking

        keys = blend(retention, relevance, weight)
        return [
            e for _, e in sorted(zip(keys, domain_entries), key=lambda pair: pair[0], reverse=True)
        ]

    def get_top_k(
        self,
        domain: str,
        k: int = 5,
        current_step: int = 0,
        query: Optional[str] = None,
    ) -> List[PlaybookEntry]:
        """
        Get the top-k entries for a domain.

        Args:
            domain: Domain name (e.g., "finance", "medical").
            k: Number of entries to return.
            current_step: Current step counter for recency calculation.
            query: The question being answered, used to rank by relevance.

        Returns:
            List of top-k PlaybookEntry objects.
        """
        return self._rank_for_retrieval(domain, current_step, query)[:k]

    def get_top_entries_for_budget(
        self,
        domain: str,
        token_budget: int,
        current_step: int = 0,
        query: Optional[str] = None,
    ) -> List[PlaybookEntry]:
        """
        Get top entries for a domain that fit within a token budget.

        Entries are sorted by retention score (descending) and greedily included
        until the estimated total token count <= token_budget.

        Args:
            domain: Domain name (e.g., "finance", "medical").
            token_budget: Maximum number of tokens allowed.
            current_step: Current step counter for recency calculation.
            query: The question being answered, used to rank by relevance.

        Returns:
            List of PlaybookEntry objects that fit within the budget.
        """
        domain_entries = self._rank_for_retrieval(domain, current_step, query)

        selected_entries = []
        total_tokens = 0

        for entry in domain_entries:
            if total_tokens + entry.token_count <= token_budget:
                selected_entries.append(entry)
                total_tokens += entry.token_count
            else:
                break

        return selected_entries

    def _find_duplicate(self, domain: str, text: str) -> Optional[PlaybookEntry]:
        """
        Find an existing entry that duplicates `text`.

        Args:
            domain: Domain to search in.
            text: Text to check for duplicates.

        Returns:
            Existing entry if a duplicate is found, None otherwise.
        """
        normalized = _normalize_for_comparison(text)
        if not normalized:
            return None
        for entry in self.entries:
            if entry.domain == domain:
                other = _normalize_for_comparison(entry.text)
                if not other:
                    continue
                if normalized in other or other in normalized:
                    return entry
        return None

    def _resolve_duplicate(
        self,
        existing: PlaybookEntry,
        text: str,
        step: int,
    ) -> PlaybookEntry:
        """
        Decide which of two overlapping lessons to keep.

        Containment is checked in both directions, so a long specific lesson
        that happens to contain a short vague one counts as a duplicate of it.
        Always keeping the incumbent meant that once "check the units" was in
        the playbook, "For density problems, check the units: convert g/cm3 to
        kg/m3 by x1000" was rejected as a duplicate and the vague entry
        survived -- directly fighting the vagueness penalty that the retention
        score is supposed to apply.

        Keep whichever is less vague, and on a tie keep the incumbent so that
        its accumulated success/failure history is not thrown away.

        Args:
            existing: The entry already in the playbook.
            text: The candidate lesson text.
            step: Current step, recorded as the entry's last touch.

        Returns:
            The surviving entry (the same object either way).
        """
        existing.last_used_at = step
        existing.last_seen_step = step

        candidate_vagueness = compute_vagueness_score(text)
        if candidate_vagueness < existing.vagueness_score:
            # The new phrasing is more specific: adopt its text, keep the
            # entry's identity and its feedback history.
            existing.text = text
            existing.vagueness_score = candidate_vagueness
            existing.is_generic = candidate_vagueness > 0.5
            if self.tokenizer is not None:
                existing.recount_tokens(self.tokenizer)
            else:
                existing.token_count = existing._estimate_tokens()

        return existing

    def _evict_lowest_score_entries(
        self,
        domain: str,
        tokens_needed: int,
        current_step: int,
    ) -> int:
        """
        Evict lowest-scoring entries to free up tokens.

        Args:
            domain: Domain to evict from.
            tokens_needed: Minimum tokens to free.
            current_step: Current step for scoring.

        Returns:
            Number of tokens freed.
        """
        # Get domain entries ordered by eviction key (ascending = dropped first).
        # Under fifo_memory this is oldest-first; otherwise lowest-score-first.
        domain_entries = [e for e in self.entries if e.domain == domain]
        domain_entries.sort(
            key=lambda e: e.eviction_key(current_step, self.scoring_params), reverse=False
        )

        tokens_freed = 0
        entries_to_remove = set()

        for entry in domain_entries:
            if tokens_freed >= tokens_needed:
                break
            entries_to_remove.add(entry.id)
            tokens_freed += entry.token_count

        # Remove evicted entries
        self.entries = [e for e in self.entries if e.id not in entries_to_remove]

        return tokens_freed

    def add_entry(
        self,
        domain: str,
        text: str,
        step: int,
        enforce_budget: bool = True,
    ) -> Optional[PlaybookEntry]:
        """
        Add a new entry to the playbook, with deduplication and optional eviction.

        If store_token_capacity is set and enforce_budget is True, this will
        evict entries (lowest-score, or oldest under FIFO) to make room.

        NOTE: New entries are added WITHOUT any success/failure feedback.
        Feedback should only be recorded when the entry is actually USED
        in a subsequent prompt.

        Args:
            domain: Domain name.
            text: Strategy/lesson text.
            step: Current step/index.
            enforce_budget: Whether to evict entries if over token budget.

        Returns:
            The added or existing PlaybookEntry, or None if eviction failed.
        """
        # Check for duplicates
        existing = self._find_duplicate(domain, text)
        if existing is not None:
            return self._resolve_duplicate(existing, text, step)

        # Create new entry (no feedback yet)
        existing_ids = {e.id for e in self.entries}
        while str(self._next_id) in existing_ids:
            self._next_id += 1
        entry_id = str(self._next_id)
        self._next_id += 1

        entry = PlaybookEntry(
            id=entry_id,
            domain=domain,
            text=text,
            success_count=0,
            failure_count=0,
            last_used_at=step,
            last_seen_step=step,
        )
        if self.tokenizer is not None:
            entry.recount_tokens(self.tokenizer)

        # Enforce the STORE capacity here. The prompt budget is applied at
        # retrieval time instead, so that ranking has candidates to reject.
        if self.store_token_capacity is not None and enforce_budget:
            domain_tokens = self.get_domain_tokens(domain)
            if domain_tokens + entry.token_count > self.store_token_capacity:
                # Need to evict some entries
                tokens_needed = (domain_tokens + entry.token_count) - self.store_token_capacity
                tokens_freed = self._evict_lowest_score_entries(domain, tokens_needed, step)

                if tokens_freed < tokens_needed:
                    # The budget is the invariant this class exists to
                    # enforce, so say so instead of silently exceeding it.
                    # (Happens when a single entry is larger than the budget.)
                    print(
                        f"Warning: playbook store capacity exceeded for domain "
                        f"'{domain}': needed {tokens_needed} tokens, freed "
                        f"{tokens_freed}. Entry added anyway; the capacity "
                        f"invariant no longer holds for this run."
                    )

        self.entries.append(entry)
        return entry

    def record_feedback(self, entry_id: str, helpful: bool) -> bool:
        """
        Record feedback for an entry (success or failure).

        This should ONLY be called when an entry was actually used in a prompt.

        Args:
            entry_id: ID of the entry.
            helpful: True if the answer was correct, False otherwise.

        Returns:
            True if entry was found and updated, False otherwise.
        """
        for entry in self.entries:
            if entry.id == entry_id:
                if helpful:
                    entry.success_count += 1
                    entry.helpful_count += 1  # Legacy
                else:
                    entry.failure_count += 1
                    entry.harmful_count += 1  # Legacy
                return True

        return False

    def mark_entry_used(self, entry_id: str, step: int) -> bool:
        """
        Mark an entry as used at a specific step (for recency tracking).

        Args:
            entry_id: ID of the entry.
            step: Current step counter.

        Returns:
            True if entry was found and updated, False otherwise.
        """
        for entry in self.entries:
            if entry.id == entry_id:
                entry.last_used_at = step
                entry.last_seen_step = step  # Legacy
                return True
        return False

    def prune(
        self,
        max_entries_per_domain: int = 32,
        *,
        current_step: int,
    ) -> int:
        """
        Prune playbook to keep only top entries per domain.

        `current_step` is keyword-only and has no default on purpose. It used
        to default to 0, and the runner relied on that default, which made
        every entry's age 0 and therefore gave every entry the identical
        recency bonus gamma. A constant added to every candidate cannot change
        a ranking, so pruning silently ignored recency altogether -- and the
        tinyace_ablate_no_recency arm was partly measuring a term that was
        already inert on the eviction path.

        Args:
            max_entries_per_domain: Maximum entries to keep per domain.
            current_step: Current step, for the recency term.

        Returns:
            Number of entries removed.
        """
        original_count = len(self.entries)
        domains = set(e.domain for e in self.entries)
        pruned_entries = []

        for domain in domains:
            # Pruning is an eviction decision: keep the entries with the
            # highest eviction key, drop the rest.
            domain_entries = [e for e in self.entries if e.domain == domain]
            domain_entries.sort(
                key=lambda e: e.eviction_key(current_step, self.scoring_params), reverse=True
            )
            pruned_entries.extend(domain_entries[:max_entries_per_domain])

        self.entries = pruned_entries
        return original_count - len(self.entries)

    def get_stats(self, domain: Optional[str] = None, current_step: int = 0) -> dict:
        """
        Get statistics about the playbook.

        Args:
            domain: Optional domain to filter by.
            current_step: Step to evaluate the recency term at. Reported
                `avg_score` is otherwise not the score any decision used.

        Returns:
            Dictionary with playbook statistics.
        """
        entries = self.entries
        if domain:
            entries = [e for e in entries if e.domain == domain]

        if not entries:
            return {
                "num_entries": 0,
                "total_tokens": 0,
                "avg_score": 0.0,
                "avg_success_rate": 0.0,
            }

        total_uses = sum(e.total_uses() for e in entries)
        total_successes = sum(e.success_count for e in entries)

        return {
            "num_entries": len(entries),
            "total_tokens": sum(e.token_count for e in entries),
            "avg_score": (
                sum(e.score(current_step, self.scoring_params) for e in entries) / len(entries)
            ),
            "avg_success_rate": total_successes / max(1, total_uses),
            "domains": list(set(e.domain for e in entries)),
        }
