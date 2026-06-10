from __future__ import annotations

from urllib.parse import urlparse, urlunparse


def build_candidate_urls(url: str) -> list[str]:
    parsed = urlparse(url)
    host = parsed.netloc.replace("www.", "")
    clean_url = _without_query(parsed)
    candidates = [url, clean_url]

    if host.endswith("ozon.ru"):
        normalized = clean_url.rstrip("/")
        if not normalized.endswith("/reviews"):
            candidates.append(f"{normalized}/reviews/")

    if host.endswith("otzovik.com"):
        candidates.extend(_build_otzovik_candidates(parsed, clean_url))

    return _deduplicate(candidates)


def _without_query(parsed) -> str:
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))


def _build_otzovik_candidates(parsed, clean_url: str) -> list[str]:
    path = parsed.path
    candidates = []

    if "/review_" in path:
        return candidates

    normalized = clean_url.rstrip("/") + "/"
    candidates.append(normalized)
    if "/reviews/" in path:
        candidates.append(f"{normalized}2/")
    return candidates


def _deduplicate(urls: list[str]) -> list[str]:
    seen = set()
    result = []
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        result.append(url)
    return result
