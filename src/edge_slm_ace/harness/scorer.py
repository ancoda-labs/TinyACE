"""Option scoring for the adaptation loop, using the harness's own model API.

Evaluation scores an answer by the loglikelihood of the letters A-D. Adaptation
must decide "was this right?" the same way, or the playbook is tuned against one
notion of correctness and reported against another -- a mode mismatch that would
look exactly like an effect of the playbook.

So adaptation does not generate an answer and parse it. It scores the same four
continuations through the same `HFLM`, and only calls generation for the
Reflector, whose job is to explain a decision that has already been made.

Sameness has to include the prompt. This class used to assemble its own context
by string interpolation while evaluation ran with `apply_chat_template=True`, so
adaptation scored in completion mode what evaluation scored in chat mode, on a
different separator, against continuations with a leading space evaluation does
not use. Context assembly is now `prompts.build_context`, which calls lm-eval's
own helpers, and `apply_chat_template` is carried on the scorer so it cannot be
set on one path and not the other.
"""

from typing import Dict, List, Optional, Sequence, Tuple

from edge_slm_ace.harness.prompts import (
    build_context,
    build_system_instruction,
    choice_continuations,
    render_question,
)


class OptionScorer:
    """
    Picks an option by loglikelihood, exactly as the evaluation task does.

    Wraps `lm_eval.models.huggingface.HFLM` so that adaptation and evaluation
    share one model implementation, one tokenizer, and one prompt rendering.
    """

    def __init__(
        self,
        model_id: str,
        device: Optional[str] = None,
        batch_size: int = 8,
        lm=None,
        apply_chat_template: bool = True,
    ):
        """
        Args:
            model_id: HuggingFace model id.
            device: "cuda", "mps", "cpu", or None to let the harness decide.
            batch_size: Requests per forward batch.
            lm: A pre-built LM, for tests. Bypasses model loading.
            apply_chat_template: Must match what the evaluation of this arm will
                pass to `simple_evaluate`. It is a constructor argument rather
                than a per-call one so a caller cannot adapt in one mode and
                evaluate in the other.
        """
        if lm is not None:
            self.lm = lm
        else:
            from lm_eval.models.huggingface import HFLM

            kwargs = {"pretrained": model_id, "batch_size": batch_size}
            if device is not None:
                kwargs["device"] = device
            self.lm = HFLM(**kwargs)
        self.model_id = model_id
        self.apply_chat_template = apply_chat_template

    def _context(self, example: Dict, system_instruction: str) -> str:
        """Assemble the context exactly as `fewshot_context` would, at 0-shot."""
        return build_context(
            render_question(example),
            system_instruction,
            apply_chat_template=self.apply_chat_template,
            chat_template=getattr(self.lm, "apply_chat_template", None),
        )

    def score(
        self,
        examples: Sequence[Dict],
        lessons: Optional[Sequence[str]] = None,
        include_scaffold: bool = True,
    ) -> List[Tuple[int, List[float]]]:
        """
        Score a batch of examples.

        Args:
            examples: Normalised Belebele examples.
            lessons: Playbook lessons to prepend, or None.
            include_scaffold: False scores the bare baseline prompt.

        Returns:
            One `(chosen_idx, logprobs)` per example, in input order.
        """
        from lm_eval.api.instance import Instance

        if not examples:
            return []

        instruction = build_system_instruction(lessons, include_scaffold=include_scaffold)
        continuations = choice_continuations(self.apply_chat_template)

        requests = []
        for example in examples:
            context = self._context(example, instruction)
            for continuation in continuations:
                requests.append(
                    Instance(
                        request_type="loglikelihood",
                        doc=example,
                        arguments=(context, continuation),
                        idx=len(requests),
                    )
                )

        flat = self.lm.loglikelihood(requests)

        results = []
        width = len(continuations)
        for i in range(len(examples)):
            logprobs = [flat[i * width + j][0] for j in range(width)]
            results.append((max(range(width), key=logprobs.__getitem__), logprobs))
        return results

    def score_one(
        self,
        example: Dict,
        lessons: Optional[Sequence[str]] = None,
        include_scaffold: bool = True,
    ) -> Tuple[int, List[float]]:
        """Score a single example. See `score`."""
        return self.score([example], lessons, include_scaffold)[0]


def option_margin(logprobs: Sequence[float], gold_idx: int) -> float:
    """
    Gold option margin: gold logprob minus the best distractor's.

    Positive means the model preferred the correct option, and the size says by
    how much. Unlike an embedding-space margin this is in the model's own
    decision units, so it is comparable across languages -- the continuations
    are the Latin letters A-D in both.

    Args:
        logprobs: Per-option loglikelihoods.
        gold_idx: Index of the correct option.

    Returns:
        The margin, in log units.
    """
    best_distractor = max(lp for i, lp in enumerate(logprobs) if i != gold_idx)
    return logprobs[gold_idx] - best_distractor
