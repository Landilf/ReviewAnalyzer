from __future__ import annotations

import asyncio
import os
import random
import re
from dataclasses import dataclass
from typing import Callable
from urllib.parse import urlparse

import pandas as pd

from app_logger import get_logger
from review_parser.extractors import clean_text, looks_like_review, prepare_reviews
from review_parser.site_rules import build_candidate_urls
from review_parser.utils import UserAgentManager, Timer


logger = get_logger("parser.otzovik")
ProgressCallback = Callable[[float, str], None]
ua_manager = UserAgentManager()
timer = Timer()


@dataclass
class CrawlConfig:
    timeout_ms: int = 45000
    max_retries: int = 4
    base_retry_delay: float = 10.0
    max_listing_pages: int = 12
    headless: bool = True


def crawl_otzovik_reviews(
    seed_url: str,
    limit: int,
    progress_callback: ProgressCallback | None = None,
    headless: bool = True,
) -> pd.DataFrame:
    callback = progress_callback or _noop_progress
    return _run_async(_crawl_otzovik_reviews(seed_url, limit, callback, headless))


async def _crawl_otzovik_reviews(seed_url: str, limit: int, progress: ProgressCallback, headless: bool = True) -> pd.DataFrame:
    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise ValueError("Для браузерного парсинга Otzovik установите зависимость `playwright`.") from exc

    config = CrawlConfig()
    config.headless = headless
    callback_step = 0.08
    progress(0.02, "Запуск браузерного краулера Otzovik")

    seed_candidates = [url for url in build_candidate_urls(seed_url) if "otzovik.com" in urlparse(url).netloc]
    if not seed_candidates:
        seed_candidates = [seed_url]
    listing_url = next((url for url in seed_candidates if "/reviews/" in urlparse(url).path), seed_candidates[0])
    logger.info("Otzovik crawler listing URL: %s", listing_url)

    async with async_playwright() as playwright:
        executable_path = os.getenv("PLAYWRIGHT_CHROMIUM_EXECUTABLE")
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
        listing_page = await context.new_page()
        try:
            await _safe_goto(listing_page, listing_url, config, "listing_page")
            await _soft_scroll(listing_page)
            
            # Step 1: Extract reviews available directly on the listing page
            listing_html = await listing_page.content()
            rows = _extract_reviews_from_listing(listing_html, listing_url)
            logger.info("Extracted %s reviews directly from listing page", len(rows))
            
            if len(rows) < limit:
                progress(callback_step, f"Собрано {len(rows)} из списка, ищу ссылки на подробности")
                review_links = await _collect_review_links(listing_page, listing_url, limit - len(rows), config, progress)
                logger.info("Collected additional review links: %s", len(review_links))

                try:
                    for idx, link in enumerate(review_links, start=1):
                        if len(rows) >= limit:
                            break
                        
                        item_progress = 0.1 + (idx / max(len(review_links), 1)) * 0.84
                        progress(
                            min(item_progress, 0.97), 
                            f"Загружено отзывов: {len(rows)}"
                        )
                        row = await _parse_review_detail(context, link, config)
                        if row:
                            rows.append(row)
                except Exception as exc:
                    logger.warning("Otzovik detail parsing interrupted: %s. Returning partial results (%s rows).", exc, len(rows))
                    if not rows:
                        raise exc
            
            frame = prepare_reviews(rows, "otzovik.com", limit)
            logger.info("Otzovik crawler prepared rows: %s", len(frame))
            
            if len(rows) < limit and len(rows) > 0:
                 progress(1.0, f"Загружено частично: {len(frame)} отзывов (был заблокирован или прерван)")
            else:
                 progress(1.0, f"Готово: загружено {len(frame)} отзывов")
                 
            return frame
        finally:
            await context.close()
            await browser.close()


async def _collect_review_links(page, listing_url: str, limit: int, config: CrawlConfig, progress: ProgressCallback) -> list[str]:
    collected: list[str] = []
    seen = set()

    base_path = urlparse(listing_url).path.rstrip("/")
    is_review_catalog = "/reviews/" in base_path

    for page_idx in range(1, config.max_listing_pages + 1):
        current_url = _otzovik_page_url(listing_url, page_idx) if is_review_catalog else listing_url
        if page_idx > 1:
            await _safe_goto(page, current_url, config, f"listing_page_{page_idx}")
            await _soft_scroll(page)
        progress(min(0.1 + page_idx * 0.04, 0.5), f"Сканирую страницу списка {page_idx}")

        html = await page.content()
        title = await page.title()
        logger.info("Listing page %s loaded: title=%s body_len=%s", page_idx, title, len(html))
        if _is_blocked_page(title, html):
             logger.warning("Listing page %s is BLOCKED", page_idx)
             break

        page_links = _extract_review_links(html, current_url)
        logger.info("Listing page %s links: %s", page_idx, len(page_links))
        for link in page_links:
            if link in seen:
                continue
            seen.add(link)
            collected.append(link)

        if len(collected) >= limit:
            break
        if not page_links and page_idx > 1:
            break

    return collected[:limit]


async def _parse_review_detail(context, link: str, config: CrawlConfig) -> dict | None:
    page = await context.new_page()
    try:
        await _safe_goto(page, link, config, "review_detail")
        await _soft_scroll(page)
        html = await page.content()
        title = await page.title()
        if _is_blocked_page(title, html):
            logger.warning("Blocked detail page: %s title=%s", link, title)
            return None
        return _parse_otzovik_detail_html(html, link)
    finally:
        await page.close()


def _parse_otzovik_detail_html(html: str, link: str) -> dict | None:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    plus = _first_text(soup, [".review-plus", "[class*='review-plus']"])
    minus = _first_text(soup, [".review-minus", "[class*='review-minus']"])
    comment = _first_text(soup, [".review-body-description", ".review-text", "[itemprop='reviewBody']"])

    parts = []
    if plus:
        parts.append(plus)
    if minus:
        parts.append(minus)
    if comment:
        parts.append(comment)

    if not parts:
        fallback = clean_text(soup.get_text(" ", strip=True))
        if looks_like_review(fallback):
            parts.append(fallback[:2800])
        else:
            return None

    text = clean_text(" ".join(parts))
    rating = _extract_rating(soup)
    date_str = _first_text(soup, ["[itemprop='datePublished']", ".review-postdate", "[class*='date']"])
    date = _parse_russian_date(date_str)
    return {"text": text, "rating": rating, "date": date, "url": link}


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


async def _safe_goto(page, url: str, config: CrawlConfig, action_name: str) -> None:
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
                    progress(0.05, "ОБНАРУЖЕНА КАПЧА: Решите её в окне браузера для продолжения...")
                    logger.warning("Captcha detected in headful mode. Waiting for user...")
                    if await _wait_for_captcha_solve(page, progress):
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
    # Ensure there is a trailing slash before appending the page index
    base = listing_url if listing_url.endswith("/") else listing_url + "/"
    return f"{base}{page_idx}/"


def _extract_rating(soup) -> float | None:
    for selector in ["[itemprop='ratingValue']", ".rating-score", "[class*='rating']"]:
        element = soup.select_one(selector)
        if not element:
            continue
        text = element.get("content") or clean_text(element.get_text(" ", strip=True))
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
