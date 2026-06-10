from collections import Counter

from analysis_helpers.text_utils import ALLOWED_ASPECT_POS, ASPECT_STOPWORDS, get_morph_analyzer, normalize_token, tokenize


def extract_aspects_simple(texts, min_frequency: int = 2, max_aspects_per_review: int = 12):
    """
    Извлекает аспектные термины через лёгкий корпусный фильтр:
    - токенизация,
    - опциональная лемматизация,
    - фильтр по части речи,
    - отсев слишком общих и редких слов,
    - включение частотных словосочетаний.
    """
    morph = get_morph_analyzer()
    per_review_candidates: list[list[str]] = []
    corpus_counts: Counter[str] = Counter()

    for text in texts:
        words = tokenize(text)
        normalized_tokens: list[tuple[str, str | None]] = []
        for word in words:
            normalized, pos = normalize_token(word)
            if len(normalized) < 3 or normalized in ASPECT_STOPWORDS:
                continue
            if morph is None:
                if normalized.endswith((
                    "ый", "ий", "ой", "ая", "ое", "ые",
                    "ого", "ему", "ими", "ом", "ым", "им",
                    "ых", "их", "ую", "юю", "о",
                )):
                    continue
            elif pos is not None and pos not in ALLOWED_ASPECT_POS:
                continue
            normalized_tokens.append((normalized, pos))

        review_candidates: list[str] = []
        for token, _ in normalized_tokens:
            review_candidates.append(token)

        unique_review_candidates: list[str] = []
        seen: set[str] = set()
        for candidate in review_candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            unique_review_candidates.append(candidate)

        per_review_candidates.append(unique_review_candidates)
        corpus_counts.update(unique_review_candidates)

    allowed_terms = {
        term for term, count in corpus_counts.items()
        if count >= min_frequency and term not in ASPECT_STOPWORDS
    }

    results: list[list[str]] = []
    for review_candidates in per_review_candidates:
        filtered = [term for term in review_candidates if term in allowed_terms]
        if not filtered:
            filtered = review_candidates[:max_aspects_per_review]
        results.append(filtered[:max_aspects_per_review])

    return results
