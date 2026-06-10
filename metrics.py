from __future__ import annotations

import re
import string

import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize

_NLTK_RESOURCES = [
    ("wordnet", "corpora/wordnet"),
    ("punkt", "tokenizers/punkt"),
    ("punkt_tab", "tokenizers/punkt_tab"),
    ("omw-1.4", "corpora/omw-1.4"),
]
for _name, _path in _NLTK_RESOURCES:
    try:
        nltk.data.find(_path)
    except LookupError:
        raise RuntimeError(
            f"NLTK resource '{_name}' not found. Run `python prefetch.py` on the login node first."
        )

# BERTScorer is expensive to initialise. Use get_bert_scorer() so it is only
# loaded when evaluation actually runs.
_bert_scorer = None


def get_bert_scorer():
    global _bert_scorer
    if _bert_scorer is None:
        from bert_score import BERTScorer
        _bert_scorer = BERTScorer(
            lang="en",
            model_type="microsoft/deberta-xlarge-mnli",
            rescale_with_baseline=True,
        )
    return _bert_scorer


def normalize_text(text: str) -> str:
    """Lowercase, strip articles / punctuation / extra whitespace."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def compute_exact_match(prediction: str, ground_truth: str) -> int:
    return int(normalize_text(prediction) == normalize_text(ground_truth))


def compute_sub_em(prediction: str, ground_truth: str) -> int:
    return int(normalize_text(ground_truth) in normalize_text(prediction))


def compute_meteor(prediction: str, ground_truth: str) -> float:
    ref_tokens = word_tokenize(ground_truth.lower())
    hyp_tokens = word_tokenize(prediction.lower())
    return meteor_score([ref_tokens], hyp_tokens)
