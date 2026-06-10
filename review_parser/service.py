from __future__ import annotations

from typing import Callable
from urllib.parse import urlparse

from app_logger import get_logger
from review_parser.browser_loader import download_page_with_browser
from review_parser.extractors import extract_links, extract_reviews
from review_parser.http_loader import download_page
from review_parser.models import ScrapeResult
from review_parser.otzovik_crawler import crawl_otzovik_reviews
from review_parser.site_rules import build_candidate_urls


logger = get_logger("parser.service")


ProgressCallback = Callable[[float, str], None]
CancelCheck = Callable[[], bool] | None


def fetch_reviews_from_url(
    url: str,
    limit: int | None = None,
    use_browser: bool = True,
    progress_callback: ProgressCallback | None = None,
    headless: bool = True,
    cancel_check: CancelCheck = None,
) -> ScrapeResult:
    progress = progress_callback or _noop_progress
    _check_cancel(cancel_check)
    normalized_url = _normalize_url(url)
    source = urlparse(normalized_url).netloc.replace("www.", "")

    # Priority 1: Specialized Otzovik Crawler (Robust & Modern)
    if source.endswith("otzovik.com") and use_browser:
        logger.info("Using dedicated Otzovik crawler")
        progress(0.02, "Запуск защищенного парсера Otzovik")
        dedicated = crawl_otzovik_reviews(
            normalized_url,
            limit=limit,
            progress_callback=progress,
            headless=headless,
            cancel_check=cancel_check,
        )
        if not dedicated.empty:
            warning = None
            if limit is not None and len(dedicated) < limit * 0.8:
                warning = "Сайт ограничил доступ во время загрузки. Анализ выполнен на основе частично собранных данных."
            return ScrapeResult(
                reviews=dedicated,
                source=source,
                message=f"Загружено {len(dedicated)} отзывов с сайта {source}.",
                warning=warning,
            )
        raise ValueError("Не удалось найти отзывы на странице Otzovik.")

    # Priority 2: General Crawler (Fallback)
    candidates = build_candidate_urls(normalized_url)
    progress(0.02, "Подготовка ссылок для загрузки")
    logger.info("Start fetching reviews (fallback): source=%s limit=%s", source, limit)
    
    last_error = None
    for candidate_index, candidate_url in enumerate(candidates, start=1):
        _check_cancel(cancel_check)
        candidate_base_progress = 0.05 + (candidate_index - 1) * (0.8 / max(len(candidates), 1))
        try:
            progress(candidate_base_progress, f"Загрузка кандидата {candidate_index}/{len(candidates)}")
            html = _load_html(candidate_url, use_browser)
            reviews = _extract_reviews_for_source(
                html,
                source,
                candidate_url,
                limit,
                use_browser,
                progress,
                candidate_base_progress,
                cancel_check,
            )
            
            if reviews.empty and use_browser:
                progress(candidate_base_progress + 0.12, "Повторная попытка (браузер)")
                html = download_page_with_browser(candidate_url)
                reviews = _extract_reviews_for_source(
                    html,
                    source,
                    candidate_url,
                    limit,
                    use_browser,
                    progress,
                    candidate_base_progress + 0.12,
                    cancel_check,
                )
                
            if not reviews.empty:
                progress(1.0, f"Готово: загружено {len(reviews)} отзывов")
                return ScrapeResult(reviews=reviews, source=source, message=f"Загружено {len(reviews)} отзывов с {source}.")
                
        except Exception as exc:
            last_error = exc
            logger.warning("Candidate failed: %s | %s", candidate_url, exc)
            continue

    details = f" Ошибка: {last_error}" if last_error else ""
    raise ValueError(f"Не удалось найти отзывы на странице.{details}")


def _normalize_url(url: str) -> str:
    normalized_url = url.strip()
    if not normalized_url:
        raise ValueError("Введите ссылку на страницу товара.")
    if not normalized_url.startswith(("http://", "https://")):
        normalized_url = "https://" + normalized_url
    return normalized_url


def _load_html(url: str, use_browser: bool) -> str:
    if use_browser:
        try:
            logger.info("Loading via HTTP first: %s", url)
            return download_page(url)
        except ValueError:
            logger.info("HTTP loading failed, switching to browser: %s", url)
            return download_page_with_browser(url)
    logger.info("Loading via HTTP only: %s", url)
    return download_page(url)


def _extract_reviews_for_source(
    html: str,
    source: str,
    base_url: str,
    limit: int | None,
    use_browser: bool,
    progress: ProgressCallback,
    progress_base: float,
    cancel_check: CancelCheck = None,
):
    _check_cancel(cancel_check)
    progress(min(progress_base + 0.06, 0.98), "Анализ структуры страницы")
    if source.endswith("otzovik.com") and "/reviews/" in urlparse(base_url).path:
        linked_reviews = _fetch_linked_otzovik_reviews(
            html, source, base_url, limit, use_browser, progress, progress_base, cancel_check
        )
        if not linked_reviews.empty:
            return linked_reviews
    progress(min(progress_base + 0.1, 0.98), "Извлечение отзывов из текущей страницы")
    return extract_reviews(html, source, limit)


def _fetch_linked_otzovik_reviews(
    html: str,
    source: str,
    base_url: str,
    limit: int | None,
    use_browser: bool,
    progress: ProgressCallback,
    progress_base: float,
    cancel_check: CancelCheck = None,
):
    _check_cancel(cancel_check)
    links = extract_links(html, base_url, r"/review_\d+\.html")
    logger.info("Otzovik linked review URLs found: %s", len(links))
    if not links:
        progress(min(progress_base + 0.12, 0.98), "На странице нет ссылок на отдельные отзывы")
        return _empty_frame()

    frames = []
    target_links = links if limit is None else links[:limit]
    total = len(target_links)
    for idx, link in enumerate(target_links, start=1):
        _check_cancel(cancel_check)
        try:
            step_progress = progress_base + 0.12 + (idx / max(total, 1)) * 0.55
            progress(min(step_progress, 0.98), f"Загрузка отзыва {idx}/{total}")
            logger.info("Loading linked Otzovik review: %s", link)
            detail_html = _load_html(link, use_browser)
            frame = extract_reviews(detail_html, source, 1)
            if not frame.empty:
                frames.append(frame)
        except Exception as exc:
            logger.warning("Linked Otzovik review failed: %s | error=%s", link, exc)
    if not frames:
        return _empty_frame()

    import pandas as pd

    frame = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["text"])
    return frame if limit is None else frame.head(limit)


def _empty_frame():
    import pandas as pd

    return pd.DataFrame()


def _noop_progress(_: float, __: str) -> None:
    return


def _check_cancel(cancel_check: CancelCheck) -> None:
    if cancel_check and cancel_check():
        raise RuntimeError("Операция отменена пользователем.")
