from __future__ import annotations

import re
import unicodedata
from collections import Counter


_LATIN_TOKEN_PATTERN = re.compile(r"[a-z0-9']+")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_LATIN_PATTERN = re.compile(r"[a-z]")

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "am",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "did",
        "do",
        "does",
        "for",
        "had",
        "has",
        "have",
        "he",
        "her",
        "his",
        "i",
        "if",
        "in",
        "is",
        "it",
        "my",
        "of",
        "on",
        "or",
        "our",
        "she",
        "that",
        "the",
        "their",
        "there",
        "these",
        "they",
        "this",
        "those",
        "to",
        "was",
        "we",
        "were",
        "will",
        "with",
        "would",
        "you",
        "your",
    }
)

_NEGATION_TOKENS = frozenset(
    {
        "arent",
        "cant",
        "cannot",
        "couldnt",
        "didnt",
        "doesnt",
        "dont",
        "hadnt",
        "hasnt",
        "havent",
        "isnt",
        "never",
        "no",
        "not",
        "shouldnt",
        "wasnt",
        "werent",
        "without",
        "wont",
        "wouldnt",
    }
)

_DIGIT_TRANSLATION = str.maketrans(
    {
        "〇": "0",
        "零": "0",
        "一": "1",
        "二": "2",
        "三": "3",
        "四": "4",
        "五": "5",
        "六": "6",
        "七": "7",
        "八": "8",
        "九": "9",
    }
)

_ORTHOGRAPHIC_REPLACEMENTS = (
    ("’", "'"),
    ("‘", "'"),
    ("“", '"'),
    ("”", '"'),
    ("入り口", "入口"),
    ("ハードドライブ", "ハードディスク"),
    ("バルコン", "ベランダ"),
)


def compute_must_preserve_rate(records) -> float:
    summary = compute_must_preserve_summary(records)
    return summary["rate"]


def compute_must_preserve_summary(records) -> dict[str, float | int]:
    total_terms = 0
    matched_terms = 0

    for record in records:
        for term in record.must_preserve:
            total_terms += 1
            if term_is_preserved(term, record.translated_text):
                matched_terms += 1

    return {
        "hits": matched_terms,
        "total": total_terms,
        "rate": (matched_terms / total_terms) if total_terms else 0.0,
    }


def term_is_preserved(term: str, translated_text: str) -> bool:
    normalized_term = normalize_preserve_text(term)
    normalized_text = normalize_preserve_text(translated_text)
    if not normalized_term or not normalized_text:
        return False

    if normalized_term in normalized_text:
        return True

    if _LATIN_PATTERN.search(normalized_term):
        return _latin_term_is_preserved(normalized_term, normalized_text)

    return _non_latin_term_is_preserved(normalized_term, normalized_text)


def normalize_preserve_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).translate(_DIGIT_TRANSLATION).casefold()
    for source, target in _ORTHOGRAPHIC_REPLACEMENTS:
        normalized = normalized.replace(source, target)
    return _WHITESPACE_PATTERN.sub(" ", normalized).strip()


def _latin_term_is_preserved(normalized_term: str, normalized_text: str) -> bool:
    term_tokens = _latin_content_tokens(normalized_term)
    text_tokens = _latin_content_tokens(normalized_text)
    if not term_tokens or not text_tokens:
        return False

    token_recall = _multiset_recall(term_tokens, text_tokens)
    if token_recall >= 1.0:
        return True

    if len(term_tokens) >= 3 and token_recall >= (2.0 / 3.0):
        return True

    term_bigrams = _token_bigrams(term_tokens)
    text_bigrams = _token_bigrams(text_tokens)
    if term_bigrams and token_recall >= 0.5:
        bigram_recall = len(term_bigrams & text_bigrams) / len(term_bigrams)
        if bigram_recall >= 0.5:
            return True

    return False


def _non_latin_term_is_preserved(normalized_term: str, normalized_text: str) -> bool:
    term_bigrams = _character_bigrams(normalized_term)
    text_bigrams = _character_bigrams(normalized_text)
    if not term_bigrams or not text_bigrams:
        return False

    bigram_recall = len(term_bigrams & text_bigrams) / len(term_bigrams)
    threshold = 0.6 if len(normalized_term) >= 6 else 0.75
    return bigram_recall >= threshold


def _latin_content_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for raw_token in _LATIN_TOKEN_PATTERN.findall(text):
        token = _normalize_latin_token(raw_token)
        if not token or token in _STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def _normalize_latin_token(token: str) -> str:
    normalized = token.replace("'", "")
    if normalized in _NEGATION_TOKENS:
        return "_neg_"
    if normalized.endswith("ies") and len(normalized) > 4:
        return normalized[:-3] + "y"
    if normalized.endswith("ing") and len(normalized) > 5:
        return normalized[:-3]
    if normalized.endswith("ed") and len(normalized) > 4:
        return normalized[:-2]
    if normalized.endswith("es") and len(normalized) > 4:
        return normalized[:-2]
    if normalized.endswith("s") and len(normalized) > 3:
        return normalized[:-1]
    return normalized


def _multiset_recall(term_tokens: list[str], text_tokens: list[str]) -> float:
    available_tokens = Counter(text_tokens)
    matched_count = 0
    for token in term_tokens:
        if available_tokens[token] > 0:
            matched_count += 1
            available_tokens[token] -= 1
    return matched_count / len(term_tokens)


def _token_bigrams(tokens: list[str]) -> set[tuple[str, str]]:
    if len(tokens) < 2:
        return set()
    return {(tokens[index], tokens[index + 1]) for index in range(len(tokens) - 1)}


def _character_bigrams(text: str) -> set[str]:
    characters = [character for character in text if not character.isspace()]
    if not characters:
        return set()
    if len(characters) == 1:
        return {characters[0]}
    return {"".join(characters[index : index + 2]) for index in range(len(characters) - 1)}
