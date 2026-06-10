from __future__ import annotations

import json
import re
from urllib.parse import urljoin

import pandas as pd

from app_logger import get_logger


logger = get_logger("parser.extractors")


def extract_reviews(html: str, source: str, limit: int | None = None) -> pd.DataFrame:
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise ValueError("Для загрузки отзывов по ссылке установите зависимость `beautifulsoup4`.") from exc

    soup = BeautifulSoup(html, "html.parser")
    logger.info("Extracting reviews: source=%s html_length=%s limit=%s", source, len(html), limit)
    if source.endswith("otzovik.com"):
        reviews = _extract_otzovik_reviews(soup)
        logger.info("Otzovik reviews found: %s", len(reviews))
        frame = prepare_reviews(reviews, source, limit)
        logger.info("Prepared reviews frame: rows=%s columns=%s", len(frame), list(frame.columns))
        return frame

    reviews = _extract_reviews_from_json_ld(soup)
    logger.info("JSON-LD reviews found: %s", len(reviews))
    if not reviews:
        reviews = _extract_reviews_from_html(soup)
        logger.info("HTML reviews found: %s", len(reviews))
    if not reviews:
        reviews = _extract_reviews_from_embedded_json(html)
        logger.info("Embedded JSON reviews found: %s", len(reviews))
    frame = prepare_reviews(reviews, source, limit)
    logger.info("Prepared reviews frame: rows=%s columns=%s", len(frame), list(frame.columns))
    return frame


def prepare_reviews(
    reviews: list[dict],
    source: str,
    limit: int | None = None,
    min_len: int = 20,
) -> pd.DataFrame:
    total_input = len(reviews)
    dropped_by_validation = 0
    rows = []
    for review in reviews:
        text = clean_text(str(review.get("text", "")), preserve_linebreaks=True)
        if min_len <= 1:
            if not text.strip():
                dropped_by_validation += 1
                continue
        else:
            if not looks_like_review(text, min_len=min_len):
                dropped_by_validation += 1
                continue
        row = {
            "text": text,
            "source": source,
            "category": source,
            "rating": review.get("rating"),
            "date": review.get("date"),
        }
        for key in ("uid", "url", "link", "review_id"):
            value = review.get(key)
            if value:
                row[key] = value
        rows.append(row)

    frame = pd.DataFrame(rows)
    before_dedup = len(frame)
    dedup_key = "text+rating+date"
    if "uid" in frame.columns and frame["uid"].notna().any():
        frame = frame.drop_duplicates(subset=["uid"])
        dedup_key = "uid"
    elif "url" in frame.columns and frame["url"].notna().any():
        frame = frame.drop_duplicates(subset=["url"])
        dedup_key = "url"
    elif "link" in frame.columns and frame["link"].notna().any():
        frame = frame.drop_duplicates(subset=["link"])
        dedup_key = "link"
    else:
        frame = frame.drop_duplicates(subset=["text", "rating", "date"])
    after_dedup = len(frame)
    if limit is not None:
        frame = frame.head(limit)
    if "rating" in frame.columns:
        frame["rating"] = pd.to_numeric(frame["rating"], errors="coerce")
        frame["label"] = frame["rating"].map(_rating_to_label)
        if frame["label"].isna().all():
            frame = frame.drop(columns=["label"])
    logger.info(
        "prepare_reviews stats: source=%s input=%s kept_after_validation=%s dropped_by_validation=%s dedup_key=%s dedup_removed=%s final=%s limit=%s",
        source,
        total_input,
        before_dedup,
        dropped_by_validation,
        dedup_key,
        before_dedup - after_dedup,
        len(frame),
        limit,
    )
    return frame


def extract_links(html: str, base_url: str, href_pattern: str) -> list[str]:
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise ValueError("Для загрузки отзывов по ссылке установите зависимость `beautifulsoup4`.") from exc

    soup = BeautifulSoup(html, "html.parser")
    pattern = re.compile(href_pattern)
    links = []
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if pattern.search(href):
            links.append(urljoin(base_url, href))
    return _deduplicate(links)


def clean_text(text: str, preserve_linebreaks: bool = False) -> str:
    text = text.replace("\\/", "/")
    if preserve_linebreaks:
        text = text.replace("\\n", "\n").replace("\r\n", "\n").replace("\r", "\n")
        lines = [re.sub(r"[ \t]+", " ", line).strip(" \t\"'") for line in text.split("\n")]
        lines = [line for line in lines if line]
        return "\n".join(lines).strip()

    text = re.sub(r"\s+", " ", text)
    text = text.replace("\\n", " ")
    return text.strip(" \t\r\n\"'")


def looks_like_review(text: str, min_len: int = 20) -> bool:
    if len(text) < min_len or len(text) > 3000:
        return False
    lowered = text.lower()
    noise_words = ["cookie", "javascript", "войдите", "подписаться", "корзина", "каталог"]
    return not any(word in lowered for word in noise_words)


def _extract_reviews_from_json_ld(soup) -> list[dict]:
    reviews = []
    for script in soup.find_all("script", type="application/ld+json"):
        payload = script.string or script.get_text(strip=True)
        if not payload:
            continue
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        reviews.extend(_walk_json_for_reviews(data))
    return reviews


def _walk_json_for_reviews(data) -> list[dict]:
    reviews = []
    if isinstance(data, dict):
        if "reviewBody" in data or data.get("@type") == "Review":
            text = data.get("reviewBody") or data.get("description") or data.get("name")
            if text:
                reviews.append(
                    {
                        "text": text,
                        "rating": _extract_rating(data.get("reviewRating")),
                        "date": data.get("datePublished"),
                    }
                )
        for value in data.values():
            reviews.extend(_walk_json_for_reviews(value))
    elif isinstance(data, list):
        for item in data:
            reviews.extend(_walk_json_for_reviews(item))
    return reviews


def _extract_reviews_from_html(soup) -> list[dict]:
    selectors = [
        "[itemprop='reviewBody']",
        "[data-widget*='review']",
        "[data-widget*='webReview']",
        "[data-widget*='webListReviews']",
        "[data-widget*='webReviewProductScore']",
        "[data-code*='review']",
        "[data-code*='opinion']",
        "[data-role*='review']",
        "[data-role*='opinion']",
        "[class*='review']",
        "[class*='Review']",
        "[class*='comment']",
        "[class*='Comment']",
        "[class*='opinion']",
        "[class*='Opinion']",
        "[class*='feedback']",
        "[class*='Feedback']",
        "[class*='reviewText']",
        "[class*='ReviewText']",
    ]
    texts = []
    for selector in selectors:
        selector_count = 0
        for element in soup.select(selector):
            text = _normalize_review_text(element.get_text(" ", strip=True))
            if looks_like_review(text):
                texts.append(text)
                selector_count += 1
        if selector_count:
            logger.info("Selector matched reviews: selector=%s count=%s", selector, selector_count)
    return [{"text": text} for text in _deduplicate(texts)]


def _extract_otzovik_reviews(soup) -> list[dict]:
    reviews = []

    detail_text = _extract_otzovik_detail_review(soup)
    if detail_text:
        reviews.append({"text": detail_text, "rating": _extract_otzovik_rating(soup)})
        return reviews

    containers = []
    for selector in [".review-list-chunk", ".review-teaser", ".review-list", "[class*='review']"]:
        containers.extend(soup.select(selector))

    for container in containers:
        text = _normalize_review_text(container.get_text(" ", strip=True))
        if looks_like_review(text):
            reviews.append({"text": text})

    return reviews


def _extract_otzovik_detail_review(soup) -> str:
    parts = []
    # Сначала пытаемся собрать структурированные части
    plus = soup.select_one(".review-plus")
    minus = soup.select_one(".review-minus")
    body = soup.select_one("[itemprop='reviewBody'], .review-body, .review-text")

    if plus:
        parts.append(f"Достоинства: {clean_text(plus.get_text())}")
    if minus:
        parts.append(f"Недостатки: {clean_text(minus.get_text())}")
    if body:
        parts.append(clean_text(body.get_text(" ", strip=True)))

    if parts:
        return " ".join(parts)

    # Фолбэк на поиск по селекторам
    candidates = []
    for selector in [
        ".review-body",
        "[itemprop='reviewBody']",
        "[class*='review-body']",
    ]:
        for element in soup.select(selector):
            text = clean_text(element.get_text(" ", strip=True))
            if looks_like_review(text):
                candidates.append(text)

    if candidates:
        return " ".join(_deduplicate(candidates))

    text = clean_text(soup.get_text(" ", strip=True))
    if "Достоинства" in text and "Недостатки" in text:
        return text[:3000]
    return ""


def _extract_otzovik_rating(soup) -> float | None:
    for selector in ["[itemprop='ratingValue']", ".product-rating", "[class*='rating']"]:
        element = soup.select_one(selector)
        if not element:
            continue
        text = element.get("content") or element.get_text(" ", strip=True)
        match = re.search(r"\d+(?:[.,]\d+)?", text)
        if match:
            return float(match.group(0).replace(",", "."))
    return None


def _extract_reviews_from_embedded_json(html: str) -> list[dict]:
    patterns = [
        r'"reviewBody"\s*:\s*"([^"]{20,})"',
        r'"text"\s*:\s*"([^"]{20,})"',
        r'"comment"\s*:\s*"([^"]{20,})"',
        r'"description"\s*:\s*"([^"]{20,})"',
        r'"content"\s*:\s*"([^"]{20,})"',
        r'"positive"\s*:\s*"([^"]{10,})".{0,500}?"negative"\s*:\s*"([^"]{10,})"',
        r'"positiveText"\s*:\s*"([^"]{10,})".{0,500}?"negativeText"\s*:\s*"([^"]{10,})"',
        r'"pros"\s*:\s*"([^"]{10,})".{0,300}?"cons"\s*:\s*"([^"]{10,})"',
        r'"advantages"\s*:\s*"([^"]{10,})".{0,500}?"disadvantages"\s*:\s*"([^"]{10,})"',
    ]
    texts = []
    for pattern in patterns:
        for match in re.finditer(pattern, html, flags=re.IGNORECASE | re.DOTALL):
            if len(match.groups()) == 2:
                text = f"Достоинства: {match.group(1)}. Недостатки: {match.group(2)}"
            else:
                text = match.group(1)
            texts.append(_decode_json_text(text))
    return [{"text": text} for text in _deduplicate(texts) if looks_like_review(text)]


def _extract_rating(review_rating) -> float | None:
    if isinstance(review_rating, dict):
        value = review_rating.get("ratingValue")
        return float(value) if value is not None else None
    return None


def _normalize_review_text(text: str) -> str:
    # Убираем только лишние пробелы и странные символы, оставляем структуру 'Достоинства:' и т.д.
    text = clean_text(text)
    text = re.sub(r"\s+\.", ".", text)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"^\.\s*", "", text)
    return clean_text(text)


def _rating_to_label(rating: float | None) -> str | None:
    if pd.isna(rating):
        return None
    if rating <= 2:
        return "negative"
    if rating == 3:
        return "neutral"
    return "positive"


def _decode_json_text(text: str) -> str:
    try:
        return json.loads(f'"{text}"')
    except json.JSONDecodeError:
        return text


def _deduplicate(texts: list[str]) -> list[str]:
    seen = set()
    unique = []
    for text in texts:
        normalized = text.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(text)
    return unique
