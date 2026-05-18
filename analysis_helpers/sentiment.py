from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def sentiment_with_transformer():
    return pipeline(
        "sentiment-analysis",
        model="cointegrated/rubert-tiny-sentiment-balanced"
    )


def sentiment_without_spacy():
    vectorizer = TfidfVectorizer(max_features=5000)
    model = LogisticRegression()
    return vectorizer, model
