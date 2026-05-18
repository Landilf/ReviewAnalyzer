from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

def topic_modeling(texts, n_topics=5):
    vectorizer = CountVectorizer(max_df=0.95, min_df=1)
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
