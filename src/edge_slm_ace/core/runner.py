"""Main runner for baseline and ACE-style evaluation pipelines."""

import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer

from edge_slm_ace.core.ace_roles import (
    build_generator_prompt,
    build_reflector_prompt,
    parse_reflector_output_to_lessons,
    choose_lessons_for_playbook,
    build_self_refine_critique_prompt,
    build_self_refine_rewrite_prompt,
    extract_answer,
    parse_used_strategies,
    build_curator_prompt,
    parse_curator_output,
)
from edge_slm_ace.utils.config import ModelConfig
from edge_slm_ace.utils.metrics import (
    compute_accuracy,
    compute_average_latency,
    semantic_answer_score,
    compute_bleu_score,
)
from edge_slm_ace.utils.mcq_eval import (
    is_sciq_task,
    has_mcq_options,
    extract_mcq_options,
    extract_mcq_options_with_indices,
    build_prompt_with_choices,
    evaluate_mcq_with_indices,
    MCQEvaluator,
    compute_mcq_aggregate_metrics,
    compute_mapping_tier_distribution,
)
from edge_slm_ace.utils.stats import summarize_accuracy
from edge_slm_ace.models.model_manager import generate, count_tokens
from edge_slm_ace.memory.playbook import Playbook
from edge_slm_ace.memory.relevance import LessonRelevance


def _index_distribution(values, n: int = 4) -> Dict[str, float]:
    """
    Fraction of predictions landing on each option slot.

    Used to detect positional answering: if the gold answer is uniform over
    slots but predictions are not, the model is answering by position.
    """
    counts = [0] * n
    total = 0
    for value in values:
        if value is None:
            continue
        idx = int(value)
        if 0 <= idx < n:
            counts[idx] += 1
            total += 1
    if total == 0:
        return {}
    return {"ABCD"[i]: counts[i] / total for i in range(n)}


def _generation_health(results: List[Dict]) -> Dict:
    """
    Summarise conditions that invalidate a run.

    `truncation_rate` above zero means prompts were clipped, and the clipped
    tail is where the question lives. `chat_template_rate` below one means the
    model was prompted as a raw completion despite being instruction-tuned.
    Both are reported so a broken run is visible in metrics.json rather than
    only in scrollback.
    """
    if not results:
        return {"truncation_rate": 0.0, "chat_template_rate": 0.0}

    truncated = sum(1 for r in results if r.get("prompt_truncated"))
    templated = sum(1 for r in results if r.get("used_chat_template"))
    rate = truncated / len(results)
    if rate > 0:
        print(
            f"WARNING: {truncated}/{len(results)} prompts were truncated "
            f"({rate:.1%}). Treat these results as invalid."
        )
    return {
        "truncation_rate": rate,
        "chat_template_rate": templated / len(results),
    }


def run_dataset_baseline(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: List[Dict],
    domain: str,
    config: ModelConfig,
    model_id: str,
    task_name: str,
    mode: str = "baseline",
    option_shuffle_seed: Optional[int] = None,
) -> tuple[List[Dict], Dict]:
    """
    Run baseline evaluation (no ACE) on a dataset.

    This function processes each example in the dataset by:
    1. Building a simple prompt (question + optional context)
    2. Generating an answer using the model
    3. Comparing answer to ground truth (exact match)
    4. Recording results with consistent schema

    For SciQ tasks, also computes MCQ-aware metrics:
    - Option-Mapped Accuracy (OMA)
    - Gold Option Margin (GOM)
    - Answerable Choice Rate (ACR)

    Args:
        model: Loaded language model (from model_manager.load_model_and_tokenizer).
        tokenizer: Loaded tokenizer (from model_manager.load_model_and_tokenizer).
        dataset: List of example dicts, each with keys:
            - id (str): Example identifier
            - question (str): Question text
            - answer (str): Ground truth answer
            - context (str, optional): Additional context
            - domain (str, optional): Domain name
            For SciQ format, also includes:
            - correct_answer (str): The correct answer text
            - distractor1, distractor2, distractor3 (str): Incorrect options
            - support (str): Supporting context
        domain: Domain name (e.g., "finance", "medical", "iot").
        config: ModelConfig with generation parameters (max_new_tokens, temperature, top_p).
        model_id: HuggingFace model ID (for result tracking).
        task_name: Task name (e.g., "tatqa_tiny", "medqa_tiny", "iot_tiny") for result tracking.
        mode: Evaluation mode (default: "baseline").

    Returns:
        Tuple of (results, summary):
        - results: List[Dict] with consistent schema. Each dict contains:
            * Core columns (always present):
              - model_id (str): Model identifier
              - task_name (str): Task identifier
              - domain (str): Domain name
              - mode (str): Evaluation mode ("baseline" or "ace")
              - sample_id (str): Example ID from dataset
              - gold (str): Ground truth answer
              - pred (str): Model prediction
              - latency_ms (float): Generation latency in milliseconds
            * Debug columns (always present):
              - question (str): Original question
              - context (str): Context if provided, else empty string
              - correct (int): 1 if exact match, 0 otherwise
            * MCQ columns (for SciQ tasks only):
              - pred_option (str): Predicted option letter (A/B/C/D)
              - gold_option (str): Correct option letter
              - oma_correct (int): 1 if pred_option == gold_option
              - gom (float): Gold Option Margin
              - acr_hit (int): 1 if choice marker detected
        - summary: Dict with aggregate statistics:
            * accuracy (float): Overall accuracy (0.0 to 1.0)
            * avg_latency_ms (float): Average latency in milliseconds
            * num_examples (int): Number of examples processed
            * oma_accuracy (float, optional): MCQ Option-Mapped Accuracy (SciQ only)
            * avg_gom (float, optional): Average Gold Option Margin (SciQ only)
            * acr_rate (float, optional): Answerable Choice Rate (SciQ only)
    """
    results = []
    predictions = []
    labels = []
    latencies = []
    end_to_end_latencies = []  # Full query processing time
    prompt_tokens_list = []
    prompt_output_tokens_list = []

    # Check if this is a SciQ task for MCQ-aware evaluation
    is_mcq_task = is_sciq_task(task_name)
    mcq_evaluator = None
    if is_mcq_task:
        try:
            mcq_evaluator = MCQEvaluator.get_instance()
        except Exception as e:
            print(f"Warning: Failed to initialize MCQEvaluator: {e}")
            is_mcq_task = False

    total_examples = len(dataset)

    for idx, example in enumerate(dataset, start=1):
        # Track end-to-end latency (entire query processing)
        query_start_time = time.time()

        example_id = example.get("id", "unknown")
        question = example.get("question", "")
        # For SciQ, use "support" as context if "context" not present
        context = example.get("context") or example.get("support")
        ground_truth = example.get("answer", "")

        # Extract MCQ options if this is a SciQ task (supports both formats)
        mcq_options_list = None
        gold_option_idx = None
        mcq_options_dict = None
        gold_option = None

        if is_mcq_task and has_mcq_options(example):
            # Try new format first (options list + gold_option_idx)
            try:
                mcq_options_list, gold_option_idx = extract_mcq_options_with_indices(
                    example, shuffle_seed=option_shuffle_seed
                )
            except (ValueError, KeyError):
                # Fall back to legacy format (correct_answer + distractors)
                mcq_options_dict, gold_option, _ = extract_mcq_options(
                    example, shuffle_seed=option_shuffle_seed
                )

        # Build prompt (with choices if options exist)
        if mcq_options_list:
            prompt = build_prompt_with_choices(question, context, mcq_options_list)
        elif context:
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        else:
            prompt = f"Question: {question}\n\nAnswer:"

        # Generate answer
        start_time = time.time()
        answer, gen_meta = generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            return_meta=True,
        )
        latency_ms = (time.time() - start_time) * 1000
        end_to_end_latency_sec = time.time() - query_start_time

        # Same extraction as every other arm -- see extract_answer().
        raw_answer = answer
        answer, _ = extract_answer(raw_answer)

        # Count tokens
        prompt_tokens = count_tokens(tokenizer, prompt)
        output_tokens = count_tokens(tokenizer, answer)
        context_tokens = count_tokens(tokenizer, context or "")

        # Check correctness
        correct = answer.strip().lower() == ground_truth.strip().lower()

        # Compute semantic score and BLEU score for metrics
        try:
            semantic_score = semantic_answer_score(answer, ground_truth)
        except Exception:
            semantic_score = None

        try:
            bleu_score = compute_bleu_score(answer, ground_truth)
        except Exception:
            # None, not 0.0: a missing sacrebleu is "not measured", and
            # recording it as a real zero would drag any average down.
            bleu_score = None

        # Progress output
        print(f"[{idx}/{total_examples}] latency={latency_ms:.0f}ms correct={correct}")

        # Record result with consistent schema
        result = {
            # Canonical columns (for plotting pipeline)
            "qid": example_id,  # Canonical question ID
            "task": task_name,  # Canonical task name
            "model": model_id,  # Canonical model ID (will be normalized by plotting script)
            "mode": mode,  # Already canonical
            "is_correct": 1 if correct else 0,  # Canonical correctness
            "context_tokens": context_tokens,  # For token efficiency plots
            "latency_ms": latency_ms,  # For latency plots
            "latency_sec": end_to_end_latency_sec,  # End-to-end latency in seconds
            # Legacy columns (for backward compatibility)
            "model_id": model_id,
            "task_name": task_name,
            "domain": domain,
            "sample_id": example_id,
            "gold": ground_truth,
            "pred": answer,
            "correct": 1 if correct else 0,
            # Token counts
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            # Additional fields for debugging
            "question": question,
            "context": context or "",
            "semantic_score": semantic_score,
            "bleu_score": bleu_score,
            # Generation provenance: a truncated prompt or a missing chat
            # template invalidates the row, so both travel with it.
            "prompt_truncated": gen_meta["prompt_truncated"],
            "used_chat_template": gen_meta["used_chat_template"],
        }

        # Compute MCQ metrics for SciQ tasks (new format with indices)
        if (
            is_mcq_task
            and mcq_options_list is not None
            and gold_option_idx is not None
            and mcq_evaluator
        ):
            try:
                mcq_metrics = evaluate_mcq_with_indices(
                    prediction=answer,
                    options=mcq_options_list,
                    gold_option_idx=gold_option_idx,
                    evaluator=mcq_evaluator,
                )
                result["chosen_option_idx"] = mcq_metrics["chosen_option_idx"]
                result["oma_correct"] = mcq_metrics["oma_correct"]
                result["gom"] = mcq_metrics["gom"]
                result["gold_option_idx"] = gold_option_idx
                result["mapping_tier"] = mcq_metrics.get("mapping_tier")
            except Exception as e:
                # Log but don't fail - MCQ metrics are optional
                print(f"Warning: MCQ evaluation failed for {example_id}: {e}")
                result["chosen_option_idx"] = None
                result["oma_correct"] = None
                result["gom"] = None
                result["gold_option_idx"] = gold_option_idx
                result["mapping_tier"] = None

        # Legacy format support (for backward compatibility)
        elif is_mcq_task and mcq_options_dict and gold_option and mcq_evaluator:
            try:
                mcq_metrics = mcq_evaluator.evaluate_mcq(
                    prediction=answer,
                    options=mcq_options_dict,
                    gold_option=gold_option,
                )
                result["pred_option"] = mcq_metrics["pred_option"]
                result["gold_option"] = mcq_metrics["gold_option"]
                result["oma_correct"] = mcq_metrics["oma_correct"]
                result["gom"] = mcq_metrics["gom"]
                result["acr_hit"] = mcq_metrics["acr_hit"]
            except Exception as e:
                # Log but don't fail - MCQ metrics are optional
                print(f"Warning: MCQ evaluation failed for {example_id}: {e}")
                result["pred_option"] = None
                result["gold_option"] = gold_option
                result["oma_correct"] = None
                result["gom"] = None
                result["acr_hit"] = None

        results.append(result)

        predictions.append(answer)
        labels.append(ground_truth)
        latencies.append(latency_ms)
        end_to_end_latencies.append(end_to_end_latency_sec)
        prompt_tokens_list.append(prompt_tokens)
        prompt_output_tokens_list.append(output_tokens)

    # Compute summary
    accuracy = compute_accuracy(predictions, labels)
    avg_latency = compute_average_latency(latencies)

    # Compute latency statistics
    import statistics

    avg_latency_sec = statistics.mean(end_to_end_latencies) if end_to_end_latencies else 0.0
    median_latency_sec = statistics.median(end_to_end_latencies) if end_to_end_latencies else 0.0

    summary = {
        "accuracy": accuracy,
        "avg_latency_ms": avg_latency,
        "avg_latency_sec": avg_latency_sec,
        "median_latency_sec": median_latency_sec,
        **_generation_health(results),
        "num_examples": len(dataset),
        "mean_prompt_token": (
            sum(prompt_tokens_list) / len(prompt_tokens_list) if prompt_tokens_list else 0.0
        ),
        "mean_output_token": (
            sum(prompt_output_tokens_list) / len(prompt_output_tokens_list)
            if prompt_output_tokens_list
            else 0.0
        ),
    }

    # Add MCQ aggregate metrics for SciQ tasks
    if is_mcq_task:
        # Check if we have new format metrics (chosen_option_idx) or legacy format (pred_option)
        has_new_format = any(
            "chosen_option_idx" in r and r.get("chosen_option_idx") is not None for r in results
        )
        has_legacy_format = any(
            "pred_option" in r and r.get("pred_option") is not None for r in results
        )

        # Handle both formats: compute metrics for whichever format(s) are present
        if has_new_format:
            # Compute aggregates for new format
            oma_values = [
                r["oma_correct"]
                for r in results
                if "oma_correct" in r and r["oma_correct"] is not None
            ]
            gom_values = [r["gom"] for r in results if "gom" in r and r["gom"] is not None]

            if oma_values:
                summary["oma_accuracy"] = sum(oma_values) / len(oma_values)
                # Ship the interval next to the point estimate so a delta is
                # never read without its noise floor.
                summary["oma_ci"] = summarize_accuracy(oma_values)
            if gom_values:
                summary["avg_gom"] = sum(gom_values) / len(gom_values)

            # Position-bias diagnostics. With options permuted, gold should be
            # ~uniform over slots; a lopsided `chosen_option_distribution`
            # against a uniform `gold_option_distribution` means the model is
            # picking by position rather than by content, and OMA should not be
            # read as accuracy.
            summary["chosen_option_distribution"] = _index_distribution(
                r.get("chosen_option_idx") for r in results
            )
            summary["gold_option_distribution"] = _index_distribution(
                r.get("gold_option_idx") for r in results
            )
            summary["mapping_tier_distribution"] = compute_mapping_tier_distribution(results)

        if has_legacy_format:
            # Compute legacy format metrics (including acr_rate)
            mcq_agg = compute_mcq_aggregate_metrics(results)
            # Only set if not already set by new format (prefer new format values if both exist)
            if "oma_accuracy" not in summary:
                summary["oma_accuracy"] = mcq_agg["oma_accuracy"]
            if "avg_gom" not in summary:
                summary["avg_gom"] = mcq_agg["avg_gom"]
            # acr_rate only exists in legacy format
            summary["acr_rate"] = mcq_agg["acr_rate"]

    return results, summary


def run_dataset_self_refine(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: List[Dict],
    domain: str,
    config: ModelConfig,
    model_id: str,
    task_name: str,
    mode: str = "self_refine",
    oracle: bool = False,
) -> tuple[List[Dict], Dict]:
    """
    Run self-refinement evaluation on a dataset (SEAL/SPICE-lite baseline).

    For each sample:
    1. Generate an initial answer.
    2. Ask the model to critique its own answer.
    3. Ask the model to rewrite the answer given that critique.
    4. Use the rewritten answer as the prediction.

    No persistent playbook; this is all per-sample.

    Oracle mode
    -----------
    With `oracle=True` the correct answer is revealed to the model during
    critique and rewrite. This measures whether the model can restate a label
    it was just handed, so it is an upper bound, NOT a baseline comparable to
    `baseline` or `ace`. It is reported under the mode name
    "self_refine_oracle" so it cannot be mistaken for one.

    Self-Refine (Madaan et al.) and SEAL do not reveal the label at
    refinement time; that is the point of the method.

    Args:
        model: Loaded language model.
        tokenizer: Loaded tokenizer.
        dataset: List of example dicts.
        domain: Domain name.
        config: ModelConfig with generation parameters.
        model_id: HuggingFace model ID (for result tracking).
        task_name: Task name for result tracking.
        mode: Evaluation mode (default: "self_refine").
        oracle: If True, reveal the correct answer during critique/rewrite.
            Produces an upper bound rather than a baseline.

    Returns:
        Tuple of (results, summary) with same schema as baseline.
    """
    results = []
    predictions = []
    labels = []
    latencies = []

    total_examples = len(dataset)

    for idx, example in enumerate(dataset, start=1):
        example_id = example.get("id", "unknown")
        question = example.get("question", "")
        context = example.get("context")
        ground_truth = example.get("answer", "")

        # Step 1: Generate initial answer
        if context:
            initial_prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        else:
            initial_prompt = f"Question: {question}\n\nAnswer:"

        start_time = time.time()
        initial_answer = generate(
            model,
            tokenizer,
            initial_prompt,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
        )
        initial_latency_ms = (time.time() - start_time) * 1000

        # Step 2: Self-refinement (critique + rewrite).
        # `oracle_answer` is None in the honest setting, so neither prompt can
        # contain the label that the rewritten answer will be scored against.
        oracle_answer = ground_truth if oracle else None

        if initial_answer:
            # Generate critique
            critique_prompt = build_self_refine_critique_prompt(
                domain=domain,
                question=question,
                context=context,
                initial_answer=initial_answer,
                ground_truth=oracle_answer,
            )

            critique_start = time.time()
            critique = generate(
                model,
                tokenizer,
                critique_prompt,
                max_new_tokens=config.max_new_tokens // 2,  # Shorter for critiques
                temperature=config.temperature,
                top_p=config.top_p,
            )
            critique_latency_ms = (time.time() - critique_start) * 1000

            # Generate rewritten answer
            rewrite_prompt = build_self_refine_rewrite_prompt(
                domain=domain,
                question=question,
                context=context,
                initial_answer=initial_answer,
                critique=critique,
                ground_truth=oracle_answer,
            )

            rewrite_start = time.time()
            final_answer = generate(
                model,
                tokenizer,
                rewrite_prompt,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
            )
            rewrite_latency_ms = (time.time() - rewrite_start) * 1000
            final_answer, _ = extract_answer(final_answer)

            total_latency_ms = initial_latency_ms + critique_latency_ms + rewrite_latency_ms
        else:
            # Nothing to refine
            final_answer = initial_answer
            critique = ""
            total_latency_ms = initial_latency_ms

        # Count tokens
        if initial_answer:
            # Count tokens for all prompts
            initial_prompt_tokens = count_tokens(tokenizer, initial_prompt)
            critique_prompt_tokens = count_tokens(tokenizer, critique_prompt)
            rewrite_prompt_tokens = count_tokens(tokenizer, rewrite_prompt)
            prompt_tokens = initial_prompt_tokens + critique_prompt_tokens + rewrite_prompt_tokens
        else:
            prompt_tokens = count_tokens(tokenizer, initial_prompt)

        output_tokens = count_tokens(tokenizer, final_answer)
        context_tokens = count_tokens(tokenizer, context or "")

        # Check correctness
        correct = final_answer.strip().lower() == ground_truth.strip().lower()

        # Compute semantic score and BLEU score for metrics
        try:
            semantic_score = semantic_answer_score(final_answer, ground_truth)
        except Exception:
            semantic_score = None

        try:
            bleu_score = compute_bleu_score(final_answer, ground_truth)
        except Exception:
            bleu_score = None

        # Progress output
        print(f"[{idx}/{total_examples}] latency={total_latency_ms:.0f}ms correct={correct}")

        # Record result with canonical format
        result = {
            # Canonical columns (for plotting pipeline)
            "qid": example_id,
            "task": task_name,
            "model": model_id,
            "mode": mode,
            "is_correct": 1 if correct else 0,
            "context_tokens": context_tokens,
            "latency_ms": total_latency_ms,
            # Legacy columns (for backward compatibility)
            "model_id": model_id,
            "task_name": task_name,
            "domain": domain,
            "sample_id": example_id,
            "gold": ground_truth,
            "pred": final_answer,
            "correct": 1 if correct else 0,
            # Token counts
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "question": question,
            "context": context or "",
            "semantic_score": semantic_score,
            "bleu_score": bleu_score,
        }
        results.append(result)

        predictions.append(final_answer)
        labels.append(ground_truth)
        latencies.append(total_latency_ms)

    # Compute summary
    accuracy = compute_accuracy(predictions, labels)
    avg_latency = compute_average_latency(latencies)

    summary = {
        "accuracy": accuracy,
        "avg_latency_ms": avg_latency,
        "num_examples": len(dataset),
    }

    return results, summary


def run_dataset_ace(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: List[Dict],
    domain: str,
    config: ModelConfig,
    playbook: Playbook,
    playbook_path: Path,
    model_id: str,
    task_name: str,
    mode: str = "ace",
    ace_mode: str = "ace_full",
    token_budget: int = 500,
    top_k: int = 5,
    reflect_on_correct_every_n: int = 5,
    prune_every_n: int = 10,
    max_entries_per_domain: int = 32,
    option_shuffle_seed: Optional[int] = None,
    enable_learning: bool = True,
    use_curator: bool = True,
) -> tuple[List[Dict], Dict]:
    """
    Run ACE-style adaptive evaluation on a dataset.

    This is ACE-inspired and not a full reproduction of the original ACE, SEAL, or SPICE algorithms.

    For each example, the ACE pipeline:
    1. Generator: Build prompt with playbook context → generate answer
    2. Check correctness against ground truth
    3. Reflector: If incorrect (or periodically if correct), generate lessons
    4. Curator: Add filtered lessons to playbook, record feedback
    5. Periodically prune playbook to keep top entries

    ACE modes:
    - ace_full: Uses top-k entries (unbounded playbook)
    - ace_working_memory: Uses token-budgeted entries (limited playbook)

    Control arm
    -----------
    With `enable_learning=False` the loop reflects nothing, writes nothing to
    the playbook and records no feedback, so the playbook stays empty. The
    Generator still receives the full ACE scaffold: the role preamble, the
    domain-specific instructions, the mandated step-by-step reasoning, the
    same choices block and the same answer parser.

    That arm exists because "ACE vs baseline" otherwise varies five things at
    once, only one of which is the playbook. The claim the playbook helps is
    `ace - cot_control`, not `ace - baseline`. If `ace - cot_control` is flat
    while `cot_control - baseline` is large, the finding is that chain-of-
    thought prompting helps and the playbook does not.

    TODO(Sathwik): Improve ACE logic in ace_roles.py:
    - Refine generator prompts to better incorporate playbook strategies
    - Improve reflector prompts to generate more specific, actionable lessons
    - Enhance lesson filtering and deduplication
    - Consider multi-turn reflection for complex errors

    TODO(Archit): Extend metrics and logging:
    - Track playbook evolution over time (size, quality metrics)
    - Measure impact of playbook on accuracy improvement
    - Add per-step latency breakdown (generation vs reflection)
    - Log playbook entries added/removed during pruning

    Args:
        model: Loaded language model (used for both generation and reflection).
        tokenizer: Loaded tokenizer.
        dataset: List of example dicts, each with keys:
            - id (str): Example identifier
            - question (str): Question text
            - answer (str): Ground truth answer
            - context (str, optional): Additional context
            - domain (str, optional): Domain name
        domain: Domain name (used to filter playbook entries).
        config: ModelConfig with generation parameters.
        playbook: The ACE playbook (will be updated in-place).
        playbook_path: Path to save playbook after run.
        model_id: HuggingFace model ID (for result tracking).
        task_name: Task name (e.g., "tatqa_tiny", "medqa_tiny", "iot_tiny") for result tracking.
        mode: Evaluation mode (default: "ace").
        ace_mode: ACE mode ("ace_full" or "ace_working_memory", default: "ace_full").
        token_budget: Token budget for working memory mode (default: 500).
        top_k: Number of top playbook entries to use in ace_full mode (default: 5).
        reflect_on_correct_every_n: Reflect on correct answers every N examples (for learning).
        prune_every_n: Prune playbook every N examples to limit size.
        max_entries_per_domain: Maximum entries per domain after pruning (default: 32).
        option_shuffle_seed: Seed for MCQ option-order permutation.
        enable_learning: If False, run as the prompt-matched control arm --
            same prompt scaffold, no reflection, no playbook writes.
        use_curator: Run the Curator pass over candidate lessons before they
            enter the playbook. Costs one extra generation per reflection
            step. Set False to ablate it.

    Returns:
        Tuple of (results, summary):
        - results: List[Dict] with consistent schema. Each dict contains:
            * Core columns (always present):
              - model_id (str): Model identifier
              - task_name (str): Task identifier
              - domain (str): Domain name
              - mode (str): Evaluation mode ("ace")
              - sample_id (str): Example ID from dataset
              - gold (str): Ground truth answer
              - pred (str): Model prediction
              - latency_ms (float): Generation latency in milliseconds
            * Debug columns (always present):
              - question (str): Original question
              - context (str): Context if provided, else empty string
              - correct (int): 1 if exact match, 0 otherwise
            * ACE-specific columns (always present in ACE mode):
              - reflection_latency_ms (float): Time spent on reflection (0.0 if no reflection)
              - reflected (bool): Whether reflection occurred for this example
        - summary: Dict with aggregate statistics:
            * accuracy (float): Overall accuracy (0.0 to 1.0)
            * avg_latency_ms (float): Average latency in milliseconds
            * num_examples (int): Number of examples processed
            * playbook_size (int): Final number of entries in playbook
    """
    results = []
    predictions = []
    labels = []
    latencies = []
    end_to_end_latencies = []  # Full query processing time
    prompt_tokens_list = []
    prompt_output_tokens_list = []
    playbook_log = []  # Per-step playbook stats

    # Check if this is a SciQ task for MCQ-aware evaluation
    is_mcq_task = is_sciq_task(task_name)
    mcq_evaluator = None
    if is_mcq_task:
        try:
            mcq_evaluator = MCQEvaluator.get_instance()
        except Exception as e:
            print(f"Warning: Failed to initialize MCQEvaluator: {e}")
            is_mcq_task = False

    for step, example in enumerate(dataset, start=1):
        # Track end-to-end latency (entire query processing)
        query_start_time = time.time()

        example_id = example.get("id", "unknown")
        question = example.get("question", "")
        # For SciQ, use "support" as context if "context" not present
        context = example.get("context") or example.get("support")
        ground_truth = example.get("answer", "")

        # Extract MCQ options if this is a SciQ task (supports both formats)
        mcq_options_list = None
        gold_option_idx = None
        mcq_options_dict = None
        gold_option = None

        if is_mcq_task and has_mcq_options(example):
            # Try new format first (options list + gold_option_idx)
            try:
                mcq_options_list, gold_option_idx = extract_mcq_options_with_indices(
                    example, shuffle_seed=option_shuffle_seed
                )
            except (ValueError, KeyError):
                # Fall back to legacy format (correct_answer + distractors)
                mcq_options_dict, gold_option, _ = extract_mcq_options(
                    example, shuffle_seed=option_shuffle_seed
                )

        # Capture playbook state before processing
        playbook_before = {
            "num_entries": len(playbook.entries),
            "total_tokens": playbook.total_tokens,
        }

        # Step 1: Generator - build prompt with playbook context.
        # Choices are rendered by build_generator_prompt via the same shared
        # block the baseline arm uses, rather than appended here with
        # different wording.
        generator_prompt = build_generator_prompt(
            domain=domain,
            playbook=playbook,
            question=question,
            context=context,
            ace_mode=ace_mode,
            token_budget=token_budget,
            top_k=top_k,
            current_step=step,
            options=mcq_options_list,
        )

        # Track which entries were used (retrieved and included in prompt)
        # This is done BEFORE generation so we know exactly which lessons were used
        used_entry_ids = []
        if ace_mode == "ace_working_memory":
            # Working memory mode: token-budgeted selection
            used_entries = playbook.get_top_entries_for_budget(
                domain=domain,
                token_budget=token_budget,
                current_step=step,
                query=question,
            )
            used_entry_ids = [e.id for e in used_entries]
        else:
            # Full mode: top-k entries
            used_entries = playbook.get_top_k(domain, k=top_k, current_step=step, query=question)
            used_entry_ids = [e.id for e in used_entries]

        # Step 2: Generate answer
        start_time = time.time()
        raw_answer, gen_meta = generate(
            model,
            tokenizer,
            generator_prompt,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            return_meta=True,
        )
        latency_ms = (time.time() - start_time) * 1000

        # Extract reasoning and answer from generator output
        answer, reasoning = extract_answer(raw_answer)

        # Step 3: Check correctness
        correct = answer.strip().lower() == ground_truth.strip().lower()

        # Progress output (before reflection to show immediate feedback)
        print(f"[{step}/{len(dataset)}] latency={latency_ms:.0f}ms correct={correct}")

        # Compute semantic score and BLEU score for metrics
        try:
            score = semantic_answer_score(answer, ground_truth)
        except Exception:
            score = None

        try:
            bleu_score = compute_bleu_score(answer, ground_truth)
        except Exception:
            # None, not 0.0: a missing sacrebleu is "not measured", and
            # recording it as a real zero would drag any average down.
            bleu_score = None

        # Set result_mode based on ace_mode
        result_mode = ace_mode  # Use ace_mode directly (ace_full or ace_working_memory)

        # Count tokens
        prompt_tokens = count_tokens(tokenizer, generator_prompt)
        output_tokens = count_tokens(tokenizer, answer)
        # Count playbook context tokens (strategies section)
        playbook_context_text = ""
        if used_entries:
            for entry in used_entries:
                playbook_context_text += entry.text + "\n"
        context_tokens = count_tokens(tokenizer, playbook_context_text)
        # Also count original context if provided
        original_context_tokens = count_tokens(tokenizer, context or "")

        # Step 4: Reflector - generate lessons.
        # The control arm never reflects, so its playbook stays empty and the
        # only difference from the ACE arm is the absence of learned content.
        should_reflect = enable_learning and (
            not correct or (step % reflect_on_correct_every_n == 0)
        )

        if should_reflect:
            reflector_prompt = build_reflector_prompt(
                domain=domain,
                question=question,
                context=context,
                model_answer=answer,
                ground_truth=ground_truth,
                reasoning=reasoning,  # Use extracted reasoning from generator output
            )

            # Generate reflection
            reflection_start = time.time()
            reflection_text = generate(
                model,
                tokenizer,
                reflector_prompt,
                max_new_tokens=config.max_new_tokens // 2,  # Shorter for reflections
                temperature=config.temperature,
                top_p=config.top_p,
            )
            reflection_latency_ms = (time.time() - reflection_start) * 1000

            # Parse lessons
            raw_lessons = parse_reflector_output_to_lessons(reflection_text)
            filtered_lessons = choose_lessons_for_playbook(
                domain=domain,
                lessons=raw_lessons,
                existing_playbook=playbook,
            )

            # Step 5: Curator - screen candidate lessons, then add them.
            #
            # This pass was implemented but never invoked, while the README and
            # the architecture diagram both described the loop as
            # Generate -> Reflect -> Curate -> Memorize. What actually ran was
            # choose_lessons_for_playbook, a hardcoded keyword filter.
            curated_lessons = filtered_lessons
            num_rejected_by_curator = 0
            curator_latency_ms = 0.0

            if use_curator and filtered_lessons:
                curator_prompt = build_curator_prompt(domain, filtered_lessons)
                curator_start = time.time()
                curator_text = generate(
                    model,
                    tokenizer,
                    curator_prompt,
                    max_new_tokens=max(64, config.max_new_tokens // 4),
                    temperature=config.temperature,
                    top_p=config.top_p,
                )
                curator_latency_ms = (time.time() - curator_start) * 1000

                is_generic_flags = parse_curator_output(len(filtered_lessons), curator_text)
                curated_lessons = [
                    lesson
                    for lesson, is_generic in zip(filtered_lessons, is_generic_flags)
                    if not is_generic
                ]
                num_rejected_by_curator = len(filtered_lessons) - len(curated_lessons)

            # NOTE: We do NOT record feedback for newly added lessons here.
            # Lessons should only receive feedback when they are actually USED
            # in a subsequent prompt. Recording feedback at creation time would
            # bias the lesson based on the example it was derived from, not on
            # whether it actually helps future examples.
            for lesson in curated_lessons:
                playbook.add_entry(
                    domain=domain,
                    text=lesson,
                    step=step,
                )
        else:
            reflection_text = ""
            reflection_latency_ms = 0.0
            curated_lessons = []
            num_rejected_by_curator = 0
            curator_latency_ms = 0.0

        # Credit assignment.
        #
        # Crediting every retrieved entry equally means each live entry gets
        # the same increment on every step, so their success ratios all
        # converge to the model's running accuracy and differ only by birth
        # time. The alpha and beta terms of the retention score then carry
        # almost no signal that distinguishes one entry from another, which is
        # a structural reason the scoring ablations show no effect.
        #
        # Prefer the entries the Generator says it applied. Fall back to
        # crediting everything retrieved only when it gave no citation, and
        # record which happened so the two can be told apart in analysis.
        cited = parse_used_strategies(raw_answer, len(used_entry_ids))
        if cited is None:
            credited_ids = used_entry_ids
            credit_mode = "uniform"
        else:
            credited_ids = [used_entry_ids[i] for i in cited if i < len(used_entry_ids)]
            credit_mode = "cited"

        if enable_learning:
            # Recency tracks what was shown; feedback tracks what was used.
            for entry_id in used_entry_ids:
                playbook.mark_entry_used(entry_id, step)
            for entry_id in credited_ids:
                playbook.record_feedback(entry_id, helpful=correct)

        # Step 6: Occasional pruning
        num_evictions = 0
        if enable_learning and step % prune_every_n == 0:
            entries_before_prune = len(playbook.entries)
            playbook.prune(max_entries_per_domain=max_entries_per_domain)
            num_evictions = entries_before_prune - len(playbook.entries)

        # Capture playbook state after processing
        playbook_after = {
            "num_entries": len(playbook.entries),
            "total_tokens": playbook.total_tokens,
        }

        # Track evictions that happened during add_entry (working memory mode)
        # This is approximate - we track the difference in entry count
        entries_added = len(curated_lessons) if should_reflect else 0
        evictions_during_add = max(
            0, playbook_before["num_entries"] + entries_added - playbook_after["num_entries"]
        )
        total_evictions = num_evictions + evictions_during_add

        # Log playbook stats for this step
        # Spread of retention scores across live entries. If this stays near
        # zero the score is not discriminating between lessons and the
        # "retention scoring" contribution is decorative, whatever the
        # ablation table says.
        live_scores = [
            e.score(step, playbook.scoring_params) for e in playbook.entries if e.domain == domain
        ]
        score_std = statistics.pstdev(live_scores) if len(live_scores) > 1 else 0.0

        playbook_log.append(
            {
                "step_index": step,
                "num_entries": playbook_after["num_entries"],
                "total_tokens": playbook_after["total_tokens"],
                "num_evictions": total_evictions,
                "retention_score_std": score_std,
                "num_retrieved": len(used_entry_ids),
                "num_credited": len(credited_ids),
                "credit_mode": credit_mode,
                "curator_latency_ms": curator_latency_ms,
                "lessons_rejected_by_curator": num_rejected_by_curator,
            }
        )

        # Calculate end-to-end latency
        end_to_end_latency_sec = time.time() - query_start_time

        # Record result with consistent schema
        result = {
            # Canonical columns (for plotting pipeline)
            "qid": example_id,
            "task": task_name,
            "model": model_id,
            "mode": result_mode,
            "is_correct": 1 if correct else 0,
            "context_tokens": context_tokens,  # Playbook context tokens
            "latency_ms": latency_ms,
            "latency_sec": end_to_end_latency_sec,  # End-to-end latency in seconds
            # Legacy columns (for backward compatibility)
            "model_id": model_id,
            "task_name": task_name,
            "domain": domain,
            "sample_id": example_id,
            "gold": ground_truth,
            "pred": answer,
            "correct": 1 if correct else 0,
            # Token counts
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            # Additional fields for debugging
            "question": question,
            "context": context or "",
            "reflection_latency_ms": reflection_latency_ms,
            "reflected": should_reflect,
            "num_retrieved": len(used_entry_ids),
            "num_credited": len(credited_ids),
            "credit_mode": credit_mode,
            "semantic_score": score,
            "bleu_score": bleu_score,
            # Generation provenance: a truncated prompt or a missing chat
            # template invalidates the row, so both travel with it.
            "prompt_truncated": gen_meta["prompt_truncated"],
            "used_chat_template": gen_meta["used_chat_template"],
        }

        # Compute MCQ metrics for SciQ tasks (new format with indices)
        if (
            is_mcq_task
            and mcq_options_list is not None
            and gold_option_idx is not None
            and mcq_evaluator
        ):
            try:
                mcq_metrics = evaluate_mcq_with_indices(
                    prediction=answer,
                    options=mcq_options_list,
                    gold_option_idx=gold_option_idx,
                    evaluator=mcq_evaluator,
                )
                result["chosen_option_idx"] = mcq_metrics["chosen_option_idx"]
                result["oma_correct"] = mcq_metrics["oma_correct"]
                result["gom"] = mcq_metrics["gom"]
                result["gold_option_idx"] = gold_option_idx
                result["mapping_tier"] = mcq_metrics.get("mapping_tier")
            except Exception as e:
                # Log but don't fail - MCQ metrics are optional
                print(f"Warning: MCQ evaluation failed for {example_id}: {e}")
                result["chosen_option_idx"] = None
                result["oma_correct"] = None
                result["gom"] = None
                result["gold_option_idx"] = gold_option_idx
                result["mapping_tier"] = None

        # Legacy format support (for backward compatibility)
        elif is_mcq_task and mcq_options_dict and gold_option and mcq_evaluator:
            try:
                mcq_metrics = mcq_evaluator.evaluate_mcq(
                    prediction=answer,
                    options=mcq_options_dict,
                    gold_option=gold_option,
                )
                result["pred_option"] = mcq_metrics["pred_option"]
                result["gold_option"] = mcq_metrics["gold_option"]
                result["oma_correct"] = mcq_metrics["oma_correct"]
                result["gom"] = mcq_metrics["gom"]
                result["acr_hit"] = mcq_metrics["acr_hit"]
            except Exception as e:
                # Log but don't fail - MCQ metrics are optional
                print(f"Warning: MCQ evaluation failed for {example_id}: {e}")
                result["pred_option"] = None
                result["gold_option"] = gold_option
                result["oma_correct"] = None
                result["gom"] = None
                result["acr_hit"] = None

        results.append(result)

        predictions.append(answer)
        labels.append(ground_truth)
        latencies.append(latency_ms)
        end_to_end_latencies.append(end_to_end_latency_sec)
        prompt_tokens_list.append(prompt_tokens)
        prompt_output_tokens_list.append(output_tokens)

    # Save playbook after run. The control arm learns nothing, so writing its
    # empty playbook would only risk clobbering a real one.
    if enable_learning:
        playbook.save(playbook_path)

    # Compute summary
    accuracy = compute_accuracy(predictions, labels)
    avg_latency = compute_average_latency(latencies)

    # Compute latency statistics
    import statistics

    avg_latency_sec = statistics.mean(end_to_end_latencies) if end_to_end_latencies else 0.0
    median_latency_sec = statistics.median(end_to_end_latencies) if end_to_end_latencies else 0.0

    summary = {
        "accuracy": accuracy,
        "avg_latency_ms": avg_latency,
        "avg_latency_sec": avg_latency_sec,
        "median_latency_sec": median_latency_sec,
        **_generation_health(results),
        "num_examples": len(dataset),
        "enable_learning": enable_learning,
        # False means retrieval was not query-conditioned, whatever the config
        # said, because no encoder was available.
        "relevance_active": bool(
            playbook.scoring_params.relevance_weight > 0
            and LessonRelevance.get_instance().available
        ),
        "relevance_weight": playbook.scoring_params.relevance_weight,
        # Share of steps where the Generator named the strategies it used. A
        # low rate means credit was mostly assigned uniformly and the
        # success/failure terms should not be read as per-lesson evidence.
        "citation_rate": (
            sum(1 for r in playbook_log if r["credit_mode"] == "cited") / len(playbook_log)
            if playbook_log
            else 0.0
        ),
        # Mean spread of retention scores. Near zero means the score does not
        # distinguish between lessons.
        "use_curator": use_curator,
        "lessons_rejected_by_curator": sum(
            r.get("lessons_rejected_by_curator", 0) for r in results
        ),
        "mean_retention_score_std": (
            statistics.mean(r["retention_score_std"] for r in playbook_log) if playbook_log else 0.0
        ),
        "playbook_size": len(playbook.entries),
        "final_playbook_num_entries": len(playbook.entries),
        "final_playbook_total_tokens": playbook.total_tokens,
        "mean_prompt_token": (
            sum(prompt_tokens_list) / len(prompt_tokens_list) if prompt_tokens_list else 0.0
        ),
        "mean_output_token": (
            sum(prompt_output_tokens_list) / len(prompt_output_tokens_list)
            if prompt_output_tokens_list
            else 0.0
        ),
        "playbook_log": playbook_log,  # Include playbook log in summary for saving
    }

    # Add MCQ aggregate metrics for SciQ tasks
    if is_mcq_task:
        # Check if we have new format metrics (chosen_option_idx) or legacy format (pred_option)
        has_new_format = any(
            "chosen_option_idx" in r and r.get("chosen_option_idx") is not None for r in results
        )
        has_legacy_format = any(
            "pred_option" in r and r.get("pred_option") is not None for r in results
        )

        # Handle both formats: compute metrics for whichever format(s) are present
        if has_new_format:
            # Compute aggregates for new format
            oma_values = [
                r["oma_correct"]
                for r in results
                if "oma_correct" in r and r["oma_correct"] is not None
            ]
            gom_values = [r["gom"] for r in results if "gom" in r and r["gom"] is not None]

            if oma_values:
                summary["oma_accuracy"] = sum(oma_values) / len(oma_values)
                # Ship the interval next to the point estimate so a delta is
                # never read without its noise floor.
                summary["oma_ci"] = summarize_accuracy(oma_values)
            if gom_values:
                summary["avg_gom"] = sum(gom_values) / len(gom_values)

            # Position-bias diagnostics. With options permuted, gold should be
            # ~uniform over slots; a lopsided `chosen_option_distribution`
            # against a uniform `gold_option_distribution` means the model is
            # picking by position rather than by content, and OMA should not be
            # read as accuracy.
            summary["chosen_option_distribution"] = _index_distribution(
                r.get("chosen_option_idx") for r in results
            )
            summary["gold_option_distribution"] = _index_distribution(
                r.get("gold_option_idx") for r in results
            )
            summary["mapping_tier_distribution"] = compute_mapping_tier_distribution(results)

        if has_legacy_format:
            # Compute legacy format metrics (including acr_rate)
            mcq_agg = compute_mcq_aggregate_metrics(results)
            # Only set if not already set by new format (prefer new format values if both exist)
            if "oma_accuracy" not in summary:
                summary["oma_accuracy"] = mcq_agg["oma_accuracy"]
            if "avg_gom" not in summary:
                summary["avg_gom"] = mcq_agg["avg_gom"]
            # acr_rate only exists in legacy format
            summary["acr_rate"] = mcq_agg["acr_rate"]

    return results, summary
