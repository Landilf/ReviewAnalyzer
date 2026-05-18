import re
import spacy

def _load_ru_spacy_model():
    try:
        return spacy.load("ru_core_news_sm")
    except OSError:
        # Some environments install spaCy models as importable Python packages.
        try:
            import ru_core_news_sm  # type: ignore

            return ru_core_news_sm.load()
        except Exception as exc:  # noqa: BLE001
            raise OSError(
                "spaCy model 'ru_core_news_sm' is not installed. "
                "Install it with:\n"
                "  python -m spacy download ru_core_news_sm\n"
                "or via pip (if your environment requires it):\n"
                "  pip install ru-core-news-sm"
            ) from exc

ASPECT_STOPWORDS = {
    "достоинство", "недостаток", "комментарий", "плюс", "минус",
    "отзыв", "товар", "день", "время", "цена", "качество",
    "достоинства", "недостатки", "комментарии", "плюсы", "минусы"
}

def extract_aspects_spacy(texts):
    nlp = _load_ru_spacy_model()
    aspects = []

    for doc in nlp.pipe(texts):
        lemmas = [
            token.lemma_.lower() 
            for token in doc 
            if token.pos_ == "NOUN" and token.lemma_.lower() not in ASPECT_STOPWORDS
        ]
        aspects.append(lemmas)
    return aspects


def extract_aspects_simple(texts):
    # Без spaCy — просто частотные существительные по шаблону
    pattern = r"\b[A-Za-zА-Яа-я]{4,}\b"
    results = []
    for text in texts:
        words = re.findall(pattern, text.lower())
        results.append([w for w in words if w not in ASPECT_STOPWORDS])
    return results
