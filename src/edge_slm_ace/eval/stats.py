"""Uncertainty and significance for accuracy comparisons.

At n=50 and p=0.74 the 95% CI on a single accuracy is roughly +/-12
percentage points, so a "+4%" difference is two questions and a "-2%"
difference is one. Reporting point estimates at that scale, ranking
configurations by them and drawing causal conclusions from the ordering is
not supportable, so the primitives to do it properly live here and every
comparison should go through them.

Pure standard library: these must never be skipped because an optional
dependency is missing.
"""

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Two-sided z for common confidence levels.
_Z = {0.90: 1.6448536269514722, 0.95: 1.959963984540054, 0.99: 2.5758293035489004}


def _z_for(confidence: float) -> float:
    if confidence not in _Z:
        raise ValueError(f"confidence must be one of {sorted(_Z)}, got {confidence}")
    return _Z[confidence]


def wilson_interval(
    successes: int,
    n: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Wilson score interval for a binomial proportion.

    Preferred over the normal approximation, which misbehaves badly at the
    sample sizes and near-ceiling accuracies used here (it can produce bounds
    above 1.0, and it collapses to zero width when p hits 0 or 1).

    Args:
        successes: Number of correct predictions.
        n: Number of predictions.
        confidence: Confidence level (0.90, 0.95 or 0.99).

    Returns:
        (lower, upper), both in [0, 1]. Returns (0.0, 1.0) when n == 0.
    """
    if n <= 0:
        return 0.0, 1.0

    z = _z_for(confidence)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, center - half), min(1.0, center + half)


def _binom_cdf(k: int, n: int, p: float = 0.5) -> float:
    """P(X <= k) for X ~ Binomial(n, p)."""
    return sum(math.comb(n, i) * (p**i) * ((1 - p) ** (n - i)) for i in range(k + 1))


def mcnemar_exact(arm_a: Sequence[int], arm_b: Sequence[int]) -> Dict:
    """
    Exact McNemar test for two arms scored on the same items.

    The correct test for this design: the arms answer identical questions, so
    the comparison is paired and only the discordant pairs carry information.
    An unpaired proportion test throws that pairing away and is less powerful.

    Args:
        arm_a: Per-item correctness for arm A (1/0), aligned with arm_b.
        arm_b: Per-item correctness for arm B (1/0).

    Returns:
        Dict with:
          n                 items compared
          acc_a, acc_b      accuracies
          delta             acc_b - acc_a, in proportion units
          b                 items A got right and B got wrong
          c                 items A got wrong and B got right
          n_discordant      b + c, the effective sample size of the test
          p_value           two-sided exact p
          significant_05    p_value < 0.05

    Raises:
        ValueError: If the arms are not the same length.
    """
    if len(arm_a) != len(arm_b):
        raise ValueError(
            f"Arms must be aligned on the same items: got {len(arm_a)} vs {len(arm_b)}"
        )

    n = len(arm_a)
    if n == 0:
        return {
            "n": 0,
            "acc_a": 0.0,
            "acc_b": 0.0,
            "delta": 0.0,
            "b": 0,
            "c": 0,
            "n_discordant": 0,
            "p_value": 1.0,
            "significant_05": False,
        }

    b = sum(1 for x, y in zip(arm_a, arm_b) if x and not y)
    c = sum(1 for x, y in zip(arm_a, arm_b) if y and not x)
    n_disc = b + c

    if n_disc == 0:
        p_value = 1.0
    else:
        # Two-sided exact binomial test on b out of (b + c) at p = 0.5.
        p_value = min(1.0, 2.0 * _binom_cdf(min(b, c), n_disc, 0.5))

    acc_a = sum(arm_a) / n
    acc_b = sum(arm_b) / n

    return {
        "n": n,
        "acc_a": acc_a,
        "acc_b": acc_b,
        "delta": acc_b - acc_a,
        "b": b,
        "c": c,
        "n_discordant": n_disc,
        "p_value": p_value,
        "significant_05": p_value < 0.05,
    }


def holm_bonferroni(p_values: Sequence[float]) -> List[float]:
    """
    Holm-Bonferroni adjusted p-values for a family of tests.

    A grid of 11 arms produces 10 tests against a reference. At alpha=0.05 the
    chance of at least one false positive under the null is 1 - 0.95^10 ~= 40%,
    so an uncorrected "significant" result from a sweep is close to expected
    rather than surprising. Holm is uniformly more powerful than Bonferroni and
    needs no independence assumption, which matters here because the arms are
    scored on the same items.

    The family is every comparison reported together. Splitting one sweep into
    several invocations to shrink it does not make the correction smaller; it
    just hides the count.

    Args:
        p_values: Raw two-sided p-values, in any order.

    Returns:
        Adjusted p-values, index-aligned with the input, each in [0, 1] and
        monotone in the raw ordering.
    """
    ranked = sorted(enumerate(p_values), key=lambda pair: pair[1])
    m = len(ranked)
    adjusted = [0.0] * m
    running = 0.0
    for rank, (index, p) in enumerate(ranked):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[index] = running
    return adjusted


def summarize_accuracy(
    correctness: Iterable[int],
    confidence: float = 0.95,
) -> Dict:
    """
    Point estimate plus interval for one arm.

    Args:
        correctness: Per-item 1/0 correctness.
        confidence: Confidence level.

    Returns:
        Dict with n, successes, accuracy, ci_low, ci_high and ci_halfwidth.
        `ci_halfwidth` is the number to compare a claimed improvement against
        before believing it.
    """
    values = [int(bool(v)) for v in correctness if v is not None]
    n = len(values)
    successes = sum(values)
    low, high = wilson_interval(successes, n, confidence)
    return {
        "n": n,
        "successes": successes,
        "accuracy": successes / n if n else 0.0,
        "ci_low": low,
        "ci_high": high,
        "ci_halfwidth": (high - low) / 2,
        "confidence": confidence,
    }


def align_on_key(
    results_a: Sequence[Dict],
    results_b: Sequence[Dict],
    metric: str = "oma_correct",
    key: str = "qid",
) -> Tuple[List[int], List[int], List[str]]:
    """
    Align two arms' per-item results on a shared identifier.

    Items missing from either arm, or missing the metric in either arm, are
    dropped from both -- a paired test on unpaired rows is meaningless.

    Args:
        results_a: Per-item result dicts for arm A.
        results_b: Per-item result dicts for arm B.
        metric: Correctness field to compare ("oma_correct" or "is_correct").
        key: Identifier field to join on.

    Returns:
        (arm_a_values, arm_b_values, shared_keys), all index-aligned.
    """
    index_a = {r[key]: r for r in results_a if key in r}
    index_b = {r[key]: r for r in results_b if key in r}

    shared = [k for k in index_a if k in index_b]
    shared.sort(key=str)

    a_vals, b_vals, used = [], [], []
    for k in shared:
        va, vb = index_a[k].get(metric), index_b[k].get(metric)
        if va is None or vb is None:
            continue
        a_vals.append(int(bool(va)))
        b_vals.append(int(bool(vb)))
        used.append(k)

    return a_vals, b_vals, used


def compare_arms(
    results_a: Sequence[Dict],
    results_b: Sequence[Dict],
    name_a: str = "A",
    name_b: str = "B",
    metric: str = "oma_correct",
    key: str = "qid",
    confidence: float = 0.95,
) -> Dict:
    """
    Full paired comparison of two arms, ready to print or tabulate.

    Args:
        results_a: Per-item results for arm A (the reference arm).
        results_b: Per-item results for arm B.
        name_a: Label for arm A.
        name_b: Label for arm B.
        metric: Correctness field to compare.
        key: Identifier field to join on.
        confidence: Confidence level for the intervals.

    Returns:
        Dict with per-arm summaries, the McNemar result, and `verdict`: a
        one-line reading of whether the difference is distinguishable from
        noise at this sample size.
    """
    a_vals, b_vals, used = align_on_key(results_a, results_b, metric=metric, key=key)

    summary_a = summarize_accuracy(a_vals, confidence)
    summary_b = summarize_accuracy(b_vals, confidence)
    test = mcnemar_exact(a_vals, b_vals)

    if test["n"] == 0:
        verdict = "no comparable items"
    elif test["significant_05"]:
        direction = "better" if test["delta"] > 0 else "worse"
        verdict = (
            f"{name_b} is {direction} than {name_a} "
            f"(p={test['p_value']:.4f}, {test['n_discordant']} discordant items)"
        )
    else:
        verdict = (
            f"no detectable difference between {name_a} and {name_b} "
            f"(p={test['p_value']:.4f}, only {test['n_discordant']} discordant "
            f"items out of {test['n']})"
        )

    return {
        "name_a": name_a,
        "name_b": name_b,
        "metric": metric,
        "n_compared": len(used),
        "arm_a": summary_a,
        "arm_b": summary_b,
        "mcnemar": test,
        "verdict": verdict,
    }


def format_comparison(comparison: Dict) -> str:
    """Render `compare_arms` output as a short human-readable block."""
    a, b, t = comparison["arm_a"], comparison["arm_b"], comparison["mcnemar"]
    return "\n".join(
        [
            f"{comparison['name_a']:<28} {a['accuracy']:6.1%}  "
            f"[{a['ci_low']:.1%}, {a['ci_high']:.1%}]  (n={a['n']})",
            f"{comparison['name_b']:<28} {b['accuracy']:6.1%}  "
            f"[{b['ci_low']:.1%}, {b['ci_high']:.1%}]  (n={b['n']})",
            f"{'delta':<28} {t['delta']:+6.1%}  " f"({t['delta'] * t['n']:+.0f} of {t['n']} items)",
            f"{'McNemar':<28} b={t['b']} c={t['c']} p={t['p_value']:.4f}",
            f"  -> {comparison['verdict']}",
        ]
    )
