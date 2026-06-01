import re

ASPECT_STOPWORDS = {
    "достоинство", "недостаток", "комментарий", "плюс", "минус",
    "отзыв", "товар", "день", "время", "цена", "качество",
    "достоинства", "недостатки", "комментарии", "плюсы", "минусы"
}


def extract_aspects_simple(texts):
    # Lightweight аспекты без spaCy: частотные слова по шаблону + стоп-слова.
    pattern = r"\b[A-Za-zА-Яа-я]{4,}\b"
    results = []
    for text in texts:
        words = re.findall(pattern, text.lower())
        results.append([w for w in words if w not in ASPECT_STOPWORDS])
    return results
