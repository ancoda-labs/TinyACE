#!/usr/bin/env python3
"""Measure Nepali/English token cost on every study model.

The debug model is random weights and is not a study tokenizer, so it is
skipped. Each remaining model is loaded at the revision pinned in MODELS.
The passages are Belebele's unique FLORES pairs (``eng_Latn`` / ``npi_Deva``),
not one row per question.

Writes ``results/fertility/fertility.csv`` and ``fertility.json``. The json
carries the provenance the csv cannot: dataset revision, bootstrap settings,
and ``capture_environment()``.

Usage:
    python -m scripts.measure_fertility
"""

import csv
import json
from typing import Callable, Dict, List, Sequence, Tuple

from edge_slm_ace.data.belebele import BELEBELE_REVISION, HF_DATASET, load_belebele
from edge_slm_ace.eval.fertility import token_ratio, unique_parallel_passages
from edge_slm_ace.reporting.layout import fertility_csv, fertility_json
from edge_slm_ace.utils.config import MODELS, REPO_ROOT, ModelSpec
from edge_slm_ace.utils.repro import capture_environment

# The defaults of token_ratio. Named here so the json records the values
# the run actually used, rather than a second copy that can drift.
N_BOOT = 10_000
SEED = 0

_COLUMNS = (
    "model",
    "hf_id",
    "revision",
    "n_pairs",
    "en_tokens",
    "ne_tokens",
    "en_words",
    "ne_words",
    "en_characters",
    "ne_characters",
    "en_tokens_per_word",
    "ne_tokens_per_word",
    "en_tokens_per_character",
    "ne_tokens_per_character",
    "f",
    "f_ci_low",
    "f_ci_high",
    "n_boot",
    "seed",
)


def study_models() -> List[ModelSpec]:
    """MODELS entries whose generation is not ``debug``."""
    return [spec for spec in MODELS.values() if spec.generation != "debug"]


def load_tokenizer(spec: ModelSpec):
    """The study tokenizer at its pinned revision. No chat template is applied."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(spec.hf_id, revision=spec.revision)


def measure_all(
    models: Sequence[ModelSpec],
    passages: Tuple[Sequence[Dict], Sequence[Dict]],
    tokenize: Callable[[ModelSpec], object],
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> List[Dict]:
    """
    ``token_ratio`` for each model, on one shared set of passage pairs.

    Args:
        models: Specs to measure. The caller decides which are study models.
        passages: ``(en_examples, ne_examples)`` from ``load_belebele``.
        tokenize: ``spec -> tokenizer``. Tests pass a fake; the script passes
            ``load_tokenizer``.
        n_boot: Forwarded to ``token_ratio``.
        seed: Forwarded to ``token_ratio``.

    Returns:
        One dict per model, with the spec's identity and the ratio fields.
    """
    en_texts, ne_texts = unique_parallel_passages(passages[0], passages[1])
    rows: List[Dict] = []
    for spec in models:
        # The tokenizer download is the slow part. Name the model before it
        # starts so a gated-repo failure says which checkpoint stopped.
        print(f"measuring {spec.key}", flush=True)
        stats = token_ratio(tokenize(spec), en_texts, ne_texts, n_boot=n_boot, seed=seed)
        rows.append(
            {
                "model": spec.key,
                "hf_id": spec.hf_id,
                "revision": spec.revision,
                **stats,
            }
        )
    return rows


def write_fertility(root, rows: Sequence[Dict], provenance: Dict) -> None:
    """Write the csv and the json under ``root`` via the layout helpers."""
    csv_path = fertility_csv(root)
    json_path = fertility_json(root)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    payload = {"provenance": provenance, "rows": list(rows)}
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def provenance(n_pairs: int) -> Dict:
    """What a later reader needs in order to regenerate the table."""
    return {
        "belebele_dataset": HF_DATASET,
        "belebele_revision": BELEBELE_REVISION,
        "languages": ["eng_Latn", "npi_Deva"],
        "passage_field": "flores_passage",
        "pairing": "passage_id",
        "tokenization": "encode(add_special_tokens=False); no chat template",
        "words": "str.split()",
        "characters": "Unicode code points, spaces included",
        "f": "total_ne_tokens / total_en_tokens",
        "bootstrap": "text pairs with replacement; 2.5 and 97.5 percentiles",
        "n_pairs": n_pairs,
        "n_boot": N_BOOT,
        "seed": SEED,
        "environment": capture_environment(),
    }


def main(argv=None) -> int:
    """Measure every non-debug model and write the fertility table."""
    del argv  # no flags: the passages, the seed and the models are fixed
    en = load_belebele("en")
    ne = load_belebele("ne")
    rows = measure_all(study_models(), (en, ne), load_tokenizer)
    root = REPO_ROOT / "results"
    write_fertility(root, rows, provenance(rows[0]["n_pairs"] if rows else 0))
    print(f"wrote {fertility_csv(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
