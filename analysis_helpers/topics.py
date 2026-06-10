import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from analysis_helpers.text_utils import TOPIC_STOPWORDS, normalize_token, tokenize

def _topic_analyzer(text: str) -> list[str]:
    normalized_tokens: list[str] = []
    for token in tokenize(text):
        normalized, _ = normalize_token(token)
        if len(normalized) < 3:
            continue
        if normalized in TOPIC_STOPWORDS:
            continue
        normalized_tokens.append(normalized)

    tokens = normalized_tokens[:]
    tokens.extend(
        f"{left} {right}"
        for left, right in zip(normalized_tokens, normalized_tokens[1:])
        if left != right and left not in TOPIC_STOPWORDS and right not in TOPIC_STOPWORDS
    )
    return tokens


def topic_modeling(texts, n_topics=5):
    document_count = len(texts)
    if document_count == 0:
        raise ValueError("Пустой набор текстов.")

    min_df = 2 if document_count >= 8 else 1
    vectorizer = CountVectorizer(analyzer=_topic_analyzer, max_df=0.9, min_df=min_df)
    X = vectorizer.fit_transform(texts)

    actual_topics = min(n_topics, max(1, X.shape[0]), max(1, X.shape[1]))
    lda = LatentDirichletAllocation(n_components=actual_topics, random_state=42)
    doc_topics = lda.fit_transform(X)

    return lda, vectorizer, doc_topics


def describe_topics(lda, vectorizer, top_words=8):
    feature_names = vectorizer.get_feature_names_out()
    rows = []
    for topic_idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[:-(top_words + 1):-1]
        for word_idx in top_indices:
            rows.append(
                {
                    "topic": f"Тема {topic_idx + 1}",
                    "term": feature_names[word_idx],
                    "weight": round(float(topic[word_idx]), 4),
                }
            )
    return pd.DataFrame(rows)
