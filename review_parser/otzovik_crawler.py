from __future__ import annotations

import asyncio
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Callable
from urllib.parse import urlparse, urlunparse

import pandas as pd

from app_logger import get_logger
from review_parser.extractors import clean_text, looks_like_review, prepare_reviews
from review_parser.site_rules import build_candidate_urls
from review_parser.utils import UserAgentManager, Timer


logger = get_logger("parser.otzovik")
ProgressCallback = Callable[[float, str], None]
CancelCheck = Callable[[], bool] | None
ua_manager = UserAgentManager()
timer = Timer()


@dataclass
class CrawlConfig:
    timeout_ms: int = 45000
    max_retries: int = 4
    base_retry_delay: float = 10.0
    max_listing_pages: int = 50
    headless: bool = True


def crawl_otzovik_reviews(
    seed_url: str,
    limit: int | None = None,
    progress_callback: ProgressCallback | None = None,
    headless: bool = True,
    cancel_check: CancelCheck = None,
) -> pd.DataFrame:
    callback = progress_callback or _noop_progress
    return _run_async(_crawl_otzovik_reviews(seed_url, limit, callback, headless, cancel_check))


async def _crawl_otzovik_reviews(
    seed_url: str,
    limit: int | None,
    progress: ProgressCallback,
    headless: bool = True,
    cancel_check: CancelCheck = None,
) -> pd.DataFrame:
    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise ValueError("Для браузерного парсинга Otzovik установите зависимость `playwright`.") from exc

    config = CrawlConfig()
    config.headless = headless
    progress(0.02, "Запуск браузерного краулера Otzovik")
    _check_cancel(cancel_check)
    logger.info("Otzovik crawler config: limit=%s headless=%s max_listing_pages=%s", limit, headless, config.max_listing_pages)

    seed_candidates = [url for url in build_candidate_urls(seed_url) if "otzovik.com" in urlparse(url).netloc]
    if not seed_candidates:
        seed_candidates = [seed_url]
    listing_url = next((url for url in seed_candidates if "/reviews/" in urlparse(url).path), seed_candidates[0])
    logger.info("Otzovik crawler listing URL: %s", listing_url)

    async with async_playwright() as playwright:
        executable_path = os.getenv("PLAYWRIGHT_CHROMIUM_EXECUTABLE")
        browser = None
        context = None
        main_page = None
        try:
            browser = await playwright.chromium.launch(
                headless=config.headless,
                executable_path=executable_path,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--window-position=0,0",
                ],
            )
            # Randomized viewport
            width = random.randint(1280, 1920)
            height = random.randint(720, 1080)

            context = await browser.new_context(
                user_agent=ua_manager.get_random_ua(),
                viewport={"width": width, "height": height},
                locale="ru-RU",
                timezone_id="Europe/Moscow",
                java_script_enabled=True,
            )

            # Initial human-like pause before doing anything
            await asyncio.sleep(random.uniform(1.0, 3.0))

            # Advanced Stealth Script
            await context.add_init_script(
                """
                // 1. Hide webdriver
                Object.defineProperty(navigator, 'webdriver', { get: () => undefined });

                // 2. Mock plugins
                Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });

                // 3. Mock languages
                Object.defineProperty(navigator, 'languages', { get: () => ['ru-RU', 'ru'] });

                // 4. Mock chrome object
                window.chrome = { runtime: {} };

                // 5. Mock permissions
                const originalQuery = window.navigator.permissions.query;
                window.navigator.permissions.query = (parameters) => (
                  parameters.name === 'notifications' ?
                    Promise.resolve({ state: Notification.permission }) :
                    originalQuery(parameters)
                );
                """
            )
            main_page = await context.new_page()

            await _safe_goto(main_page, listing_url, config, "listing_page", progress)
            await _soft_scroll(main_page)

            # Step 1: Extract reviews available directly on the listing page
            listing_html = await main_page.content()
            rows = _extract_reviews_from_listing(listing_html, listing_url)
            logger.info("Extracted %s reviews directly from listing page", len(rows))
            _check_cancel(cancel_check)

            if limit is None or len(rows) < limit:
                remaining = None if limit is None else max(limit - len(rows), 0)
                review_links = await _collect_review_links(main_page, listing_url, remaining, config, progress, cancel_check)
                logger.info("Collected additional review links: %s", len(review_links))
                total_reviews = len(rows) + len(review_links)
                progress(0.0, f"Начинаю парсинг отзывов: 0 из {total_reviews}")

                try:
                    parsed_detail_reviews = 0
                    failed_detail_reviews = 0
                    parsing_started_at = time.monotonic()
                    # Используем ту же самую страницу main_page для перехода по деталям
                    for idx, link in enumerate(review_links, start=1):
                        _check_cancel(cancel_check)
                        if limit is not None and len(rows) >= limit:
                            break

                        item_progress = idx / max(len(review_links), 1)
                        elapsed = max(time.monotonic() - parsing_started_at, 0.001)
                        remaining_reviews = max(len(review_links) - idx, 0)
                        eta_seconds = (elapsed / idx) * remaining_reviews if idx else 0
                        eta_text = _format_eta(eta_seconds)
                        progress_text = f"Парсинг отзывов: {idx} из {len(review_links)}"
                        if eta_text:
                            progress_text += f" · Осталось {eta_text}"
                        progress(
                            min(item_progress, 0.97),
                            progress_text,
                        )
                        row = await _parse_review_detail_reused(main_page, link, config, cancel_check)
                        if row:
                            rows.append(row)
                            parsed_detail_reviews += 1
                        else:
                            failed_detail_reviews += 1
                        if idx == len(review_links) or idx % 10 == 0:
                            logger.info(
                                "Detail parsing progress: processed=%s/%s success=%s failed=%s accumulated_rows=%s",
                                idx,
                                len(review_links),
                                parsed_detail_reviews,
                                failed_detail_reviews,
                                len(rows),
                            )
                except Exception as exc:
                    logger.warning("Otzovik detail parsing interrupted: %s. Returning partial results (%s rows).", exc, len(rows))
                    if not rows:
                        raise exc
                else:
                    logger.info(
                        "Detail parsing summary: links=%s success=%s failed=%s total_raw_rows=%s",
                        len(review_links),
                        parsed_detail_reviews,
                        failed_detail_reviews,
                        len(rows),
                    )

            frame = prepare_reviews(rows, "otzovik.com", limit, min_len=1)
            logger.info(
                "Otzovik crawler prepared rows: final=%s raw_rows=%s listing_rows=%s",
                len(frame),
                len(rows),
                len(_extract_reviews_from_listing(listing_html, listing_url)),
            )

            if limit is not None and len(rows) < limit and len(rows) > 0:
                progress(1.0, f"Загружено частично: {len(frame)} отзывов")
            else:
                progress(1.0, f"Готово: загружено {len(frame)} отзывов")

            return frame
        finally:
            if main_page is not None:
                await main_page.close()
            if context is not None:
                await context.close()
            if browser is not None:
                await browser.close()


async def _collect_review_links(
    page,
    listing_url: str,
    limit: int | None,
    config: CrawlConfig,
    progress: ProgressCallback,
    cancel_check: CancelCheck = None,
) -> list[str]:
    collected: list[str] = []
    seen = set()
    page_stats: list[dict[str, int | str]] = []

    base_path = urlparse(listing_url).path.rstrip("/")
    is_review_catalog = "/reviews/" in base_path
    max_pages_to_visit = 1
    page_idx = 1

    while page_idx <= max_pages_to_visit:
        _check_cancel(cancel_check)
        current_url = _otzovik_page_url(listing_url, page_idx) if is_review_catalog else listing_url
        if page_idx > 1:
            await _safe_goto(page, current_url, config, f"listing_page_{page_idx}", progress)
            await _soft_scroll(page)
        if page_idx == 1:
            scan_progress = 0.05
            scan_message = "Сканирую страницу 1"
        else:
            scan_progress = min(((page_idx - 1) / max(max_pages_to_visit, 1)) + 0.02, 0.97)
            scan_message = f"Сканирую страницу {page_idx} из {max_pages_to_visit}"
        progress(scan_progress, scan_message)

        html = await page.content()
        title = await page.title()
        logger.info("Listing page %s loaded: title=%s body_len=%s", page_idx, title, len(html))
        if _is_blocked_page(title, html):
             logger.warning("Listing page %s is BLOCKED", page_idx)
             break

        page_links = _extract_review_links(html, current_url)
        
        # Если ссылок нет на первой странице и мы не в headless режиме, 
        # даем пользователю шанс прокрутить или решить капчу
        if not page_links and page_idx == 1 and not config.headless:
            logger.info("No links found on first page, waiting for user intervention...")
            progress(min(page_idx / max(max_pages_to_visit, 1), 0.97), "Отзывы не найдены. Если видите капчу — решите её в окне браузера.")
            if await _wait_for_captcha_solve(page, progress, timeout_m=2):
                html = await page.content()
                page_links = _extract_review_links(html, current_url)

        logger.info("Listing page %s links: %s", page_idx, len(page_links))
        discovered_pages = _extract_listing_page_count(html, listing_url)
        if is_review_catalog and discovered_pages:
            max_pages_to_visit = max(max_pages_to_visit, min(discovered_pages, config.max_listing_pages))
            logger.info(
                "Listing page %s pagination discovered total_pages=%s effective_limit=%s",
                page_idx,
                discovered_pages,
                max_pages_to_visit,
            )
        progress(min(page_idx / max(max_pages_to_visit, 1), 0.97), f"Страница {page_idx} из {max_pages_to_visit}")
        for link in page_links:
            if link in seen:
                continue
            seen.add(link)
            collected.append(link)
        page_stats.append(
            {
                "page_idx": page_idx,
                "page_links": len(page_links),
                "unique_total": len(collected),
                "discovered_pages": discovered_pages,
            }
        )
        logger.info(
            "Listing page summary: page=%s links_on_page=%s unique_total=%s discovered_pages=%s next_page_limit=%s url=%s",
            page_idx,
            len(page_links),
            len(collected),
            discovered_pages,
            max_pages_to_visit,
            current_url,
        )

        if limit is not None and len(collected) >= limit:
            break
        if not page_links and page_idx > 1:
            break
        page_idx += 1

    logger.info("Listing crawl summary: pages=%s total_unique_links=%s stats=%s", len(page_stats), len(collected), page_stats)
    return collected if limit is None else collected[:limit]


async def _parse_review_detail_reused(page, link: str, config: CrawlConfig, cancel_check: CancelCheck = None) -> dict | None:
    try:
        _check_cancel(cancel_check)
        await _safe_goto(page, link, config, "review_detail")
        await _soft_scroll(page)
        html = await page.content()
        title = await page.title()
        if _is_blocked_page(title, html):
            logger.warning("Blocked detail page: %s title=%s", link, title)
            return None
        row = _parse_otzovik_detail_html(html, link)
        if row:
            logger.info(
                "Detail parsed: url=%s text_len=%s has_rating=%s has_date=%s",
                link,
                len(row.get("text", "")),
                row.get("rating") is not None,
                bool(row.get("date")),
            )
        else:
            logger.warning("Detail parsed empty: url=%s", link)
        return row
    except Exception as exc:
        logger.warning("Failed to parse detail page %s: %s", link, exc)
        return None


def _parse_otzovik_detail_html(html: str, link: str) -> dict | None:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    plus = _strip_otzovik_label(_first_text(soup, [".review-plus", "[class*='review-plus']"]), "Достоинства")
    minus = _strip_otzovik_label(_first_text(soup, [".review-minus", "[class*='review-minus']"]), "Недостатки")
    
    # Основной текст отзыва на Отзовике обычно лежит в itemprop="reviewBody" или .review-body
    comment = _first_text(soup, [
        "[itemprop='reviewBody']", 
        ".review-body", 
        ".review-body-description", 
        ".review-text"
    ])
    comment = _normalize_otzovik_comment(comment, plus, minus)

    header_parts = []
    if plus:
        header_parts.append(f"Достоинства: {plus}")
    if minus:
        header_parts.append(f"Недостатки: {minus}")

    parts = []
    if header_parts:
        parts.append("\n".join(header_parts))
    if comment:
        parts.append(comment)

    if not parts:
        fallback = clean_text(soup.get_text("\n", strip=True), preserve_linebreaks=True)
        if looks_like_review(fallback):
            parts.append(fallback[:2800])
        else:
            return None

    text = clean_text("\n\n".join(parts), preserve_linebreaks=True)
    rating = _extract_rating(soup)
    date_str = _first_text(soup, ["[itemprop='datePublished']", ".review-postdate", "[class*='date']"])
    date = _parse_russian_date(date_str)
    return {"text": text, "rating": rating, "date": date, "url": link}


def _normalize_otzovik_comment(comment: str, plus: str, minus: str) -> str:
    if not comment:
        return ""

    normalized = clean_text(comment, preserve_linebreaks=True)
    if plus:
        normalized = re.sub(
            rf"^\s*Достоинства:\s*{re.escape(plus)}\s*",
            "",
            normalized,
            flags=re.IGNORECASE,
        )
    if minus:
        normalized = re.sub(
            rf"^\s*Недостатки:\s*{re.escape(minus)}\s*",
            "",
            normalized,
            flags=re.IGNORECASE,
        )
        normalized = re.sub(
            rf"\bНедостатки:\s*{re.escape(minus)}\s*",
            "",
            normalized,
            flags=re.IGNORECASE,
        )
    if plus:
        normalized = re.sub(
            rf"\bДостоинства:\s*{re.escape(plus)}\s*",
            "",
            normalized,
            flags=re.IGNORECASE,
        )
    return clean_text(normalized, preserve_linebreaks=True)


def _strip_otzovik_label(value: str, label: str) -> str:
    if not value:
        return ""
    pattern = rf"^\s*{re.escape(label)}\s*:\s*"
    return clean_text(re.sub(pattern, "", value, flags=re.IGNORECASE), preserve_linebreaks=True)


def _parse_russian_date(date_str: str) -> str:
    """
    Конвертирует строку даты вида '8 июн 2019' в ISO формат '2019-06-08'.
    """
    if not date_str:
        return ""
    
    months = {
        "янв": "01", "фев": "02", "мар": "03", "апр": "04", "май": "05", "июн": "06",
        "июл": "07", "авг": "08", "сен": "09", "окт": "10", "ноя": "11", "дек": "12"
    }
    
    date_str = date_str.lower().replace(".", "")
    match = re.search(r"(\d{1,2})\s+([а-я]{3})\s+(\d{4})", date_str)
    if match:
        day, mon_str, year = match.groups()
        month = months.get(mon_str[:3], "01")
        return f"{year}-{month}-{day.zfill(2)}"
        
    return date_str


async def _safe_goto(
    page,
    url: str,
    config: CrawlConfig,
    action_name: str,
    progress: ProgressCallback | None = None,
) -> None:
    current_retry_delay = config.base_retry_delay
    last_error = None
    
    for attempt in range(1, config.max_retries + 1):
        try:
            logger.info("Goto %s attempt %s/%s: %s", action_name, attempt, config.max_retries, url)
            
            # Masking with extra headers
            await page.set_extra_http_headers({
                "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
                "Upgrade-Insecure-Requests": "1",
            })
            
            response = await page.goto(url, wait_until="domcontentloaded", timeout=config.timeout_ms, referer="https://www.google.com/")
            status = response.status if response else None
            
            if status in {401, 403, 429}:
                raise ValueError(f"HTTP {status}")
                
            title = await page.title()
            html = await page.content()
            
            if _is_blocked_page(title, html):
                if not config.headless:
                    # WAIT FOR USER
                    if progress:
                        progress(0.05, "Обнаружена капча: решите её в окне браузера для продолжения")
                    logger.warning("Captcha detected in headful mode. Waiting for user...")
                    if await _wait_for_captcha_solve(page, progress or _noop_progress):
                         # Re-check status after solve
                         return
                
                raise ValueError(f"Blocked page title={title}")
                
            # Success - small human pause after load
            await timer.sleep(1.5, 3.0)
            return
            
        except Exception as exc:
            last_error = exc
            logger.warning("Goto failed %s attempt %s: %s", action_name, attempt, exc)
            if attempt < config.max_retries:
                # If it's a block, we wait even longer
                wait_multiplier = 1.5 if "Block" in str(exc) else 1.0
                delay = current_retry_delay * wait_multiplier
                logger.info("Waiting %.2f seconds (cooldown) before retry...", delay)
                await timer.sleep(delay, delay + 5.0)
                current_retry_delay *= 2  # Exponential backoff
                
    raise ValueError(f"Не удалось загрузить страницу ({action_name}): {last_error}")


async def _wait_for_captcha_solve(page, progress: ProgressCallback, timeout_m: int = 5) -> bool:
    """
    Цикл ожидания решения капчи пользователем. 
    Опрашивает страницу каждые 2 секунды.
    """
    import time
    start_time = time.time()
    while time.time() - start_time < timeout_m * 60:
        await asyncio.sleep(2.0)
        try:
            title = await page.title()
            html = await page.content()
            if not _is_blocked_page(title, html):
                logger.info("Captcha solved by user!")
                progress(0.05, "Капча решена, продолжаю работу...")
                await timer.sleep(1.0, 2.0)
                return True
        except Exception:
            break
    return False


async def _soft_scroll(page) -> None:
    for _ in range(3):
        await page.mouse.wheel(0, random.randint(700, 1300))
        await timer.sleep(0.6, 1.2)


def _extract_reviews_from_listing(html: str, base_url: str) -> list[dict]:
    from bs4 import BeautifulSoup
    from urllib.parse import urljoin

    soup = BeautifulSoup(html, "html.parser")
    rows = []
    
    # Otzovik listing items often have this class
    containers = soup.select(".review-item, .item, .review-teaser, [class*='review-item']")
    for container in containers:
        text_el = container.select_one(".review-body, .review-text, .description, [class*='text']")
        if not text_el:
            continue
            
        text = _normalize_review_text(text_el.get_text(" ", strip=True))
        if not looks_like_review(text):
            continue
            
        rating = _extract_rating(container)
        date_el = container.select_one(".review-postdate, .date, [class*='date']")
        date_str = date_el.get_text(strip=True) if date_el else ""
        
        link_el = container.select_one("a[href*='/review_']")
        link = urljoin(base_url, link_el["href"]) if link_el else ""
        
        rows.append({
            "text": text,
            "rating": rating,
            "date": _parse_russian_date(date_str),
            "url": link
        })
        
    return rows


def _extract_review_links(html: str, base_url: str) -> list[str]:
    from bs4 import BeautifulSoup
    from urllib.parse import urljoin

    soup = BeautifulSoup(html, "html.parser")
    links = []
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        
        # STRICT FILTERING:
        # 1. Must match the review detail pattern
        # 2. Must NOT contain query parameters (sorting, filters)
        # 3. Must NOT contain fragments (#comments, etc.)
        if re.search(r"/review_\d+\.html$", href.split('?')[0].split('#')[0]):
            full_url = urljoin(base_url, href).split('?')[0].split('#')[0]
            links.append(full_url)
             
    return _deduplicate(links)


def _otzovik_page_url(listing_url: str, page_idx: int) -> str:
    if page_idx <= 1:
        return listing_url
    parsed = urlparse(listing_url)
    base_path = parsed.path.rstrip("/")
    paged_path = f"{base_path}/{page_idx}/"
    return urlunparse(parsed._replace(path=paged_path))


def _extract_listing_page_count(html: str, listing_url: str) -> int:
    from bs4 import BeautifulSoup
    from urllib.parse import urljoin

    soup = BeautifulSoup(html, "html.parser")
    parsed_listing = urlparse(listing_url)
    base_path = parsed_listing.path.rstrip("/")
    max_page = 1

    for anchor in soup.find_all("a", href=True):
        full_url = urljoin(listing_url, anchor["href"])
        parsed = urlparse(full_url)
        if parsed.netloc != parsed_listing.netloc:
            continue

        path = parsed.path.rstrip("/")
        if not path.startswith(base_path):
            continue

        match = re.search(r"/(\d+)$", path)
        if match:
            max_page = max(max_page, int(match.group(1)))

    return max_page


def _extract_rating(soup) -> float | None:
    selectors = [
        ".review-wrap [itemprop='ratingValue']",
        ".review-wrap .rating-score",
        ".review-wrap abbr.rating",
        "[itemprop='reviewRating'] [itemprop='ratingValue']",
        "abbr.rating",
        ".rating-score",
        "[itemprop='ratingValue']",
        "[class*='rating']",
    ]
    for selector in selectors:
        element = soup.select_one(selector)
        if not element:
            continue
        text = (
            element.get("content")
            or element.get("title")
            or clean_text(element.get_text(" ", strip=True))
        )
        match = re.search(r"\d+(?:[.,]\d+)?", text)
        if match:
            return float(match.group(0).replace(",", "."))
    return None


def _first_text(soup, selectors: list[str]) -> str:
    for selector in selectors:
        element = soup.select_one(selector)
        if element:
            value = clean_text(element.get_text(" ", strip=True))
            if value:
                return value
    return ""


def _is_blocked_page(title: str, html: str) -> bool:
    sample = f"{title} {html[:1500]}".lower()
    blocked_tokens = [
        "http 403", "доступ ограничен", "captcha", "anti-bot", 
        "security check", "forbidden", "вы робот", "робот?", 
        "слишком много обращений", "подозрительная активность"
    ]
    return any(token in sample for token in blocked_tokens)


def _deduplicate(items: list[str]) -> list[str]:
    seen = set()
    unique = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


def _run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def _noop_progress(_: float, __: str) -> None:
    return


def _check_cancel(cancel_check: CancelCheck) -> None:
    if cancel_check and cancel_check():
        raise RuntimeError("Операция отменена пользователем.")


def _format_eta(seconds: float) -> str:
    total_seconds = max(int(round(seconds)), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    parts = []
    if hours:
        parts.append(f"{hours} ч")
    if minutes:
        parts.append(f"{minutes} мин")
    if secs:
        parts.append(f"{secs} сек")
    return ", ".join(parts)
