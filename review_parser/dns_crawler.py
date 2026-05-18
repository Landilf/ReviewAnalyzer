from __future__ import annotations

import asyncio
import os
import random
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from app_logger import get_logger
from review_parser.extractors import clean_text, looks_like_review, prepare_reviews
from review_parser.utils.ua_manager import UserAgentManager
from review_parser.utils.timer import Timer


logger = get_logger("parser.dns")
ProgressCallback = Callable[[float, str], None]
ua_manager = UserAgentManager()
timer = Timer()


@dataclass
class DnsCrawlerConfig:
    max_show_more_clicks: int = 3
    max_scrolls: int = 10
    max_reviews: int = 5
    timeout_ms: int = 45000
    max_retries: int = 3
    base_retry_delay: float = 5.0


def crawl_dns_reviews(url: str, progress_callback: ProgressCallback | None = None) -> pd.DataFrame:
    callback = progress_callback or _noop_progress
    config = DnsCrawlerConfig()
    return _run_async(_crawl_dns_reviews(url, config, callback))


async def _crawl_dns_reviews(url: str, config: DnsCrawlerConfig, progress: ProgressCallback) -> pd.DataFrame:
    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise ValueError("Для DNS crawler установите зависимость `playwright`.") from exc

    progress(0.05, "Запуск DNS crawler")
    logger.info("DNS crawler start: url=%s config=%s", url, config)

    async with async_playwright() as playwright:
        executable_path = os.getenv("PLAYWRIGHT_CHROMIUM_EXECUTABLE")
        browser = await playwright.chromium.launch(
            headless=True,
            executable_path=executable_path,
            args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
        )
        context = await browser.new_context(
            user_agent=ua_manager.get_random_ua(),
            viewport={"width": 1600, "height": 900},
            locale="ru-RU",
            timezone_id="Europe/Moscow",
        )
        await context.add_init_script(
            """
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
            """
        )
        page = await context.new_page()
        
        current_retry_delay = config.base_retry_delay
        last_error = None
        
        try:
            for attempt in range(1, config.max_retries + 1):
                try:
                    logger.info("DNS crawler attempt %s/%s for %s", attempt, config.max_retries, url)
                    await page.set_extra_http_headers({
                        "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
                        "Sec-Fetch-Dest": "document",
                        "Sec-Fetch-Mode": "navigate",
                        "Sec-Fetch-Site": "none",
                        "Sec-Fetch-User": "?1",
                        "Upgrade-Insecure-Requests": "1",
                    })
                    await page.goto(url, wait_until="domcontentloaded", timeout=config.timeout_ms, referer="https://www.google.com/")
                    await timer.sleep(2.0, 4.0)
                    
                    title = await page.title()
                    logger.info("DNS crawler loaded: title=%s", title)
                    
                    if _is_blocked_page(title):
                        raise ValueError(f"DNS Anti-Bot Block detected: {title}")
                    
                    # If we reached here, page is loaded successfully
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    logger.warning("DNS crawler attempt %s failed: %s", attempt, exc)
                    if attempt < config.max_retries:
                        logger.info("Waiting %s seconds before retry...", current_retry_delay)
                        await timer.sleep(current_retry_delay, current_retry_delay + 2)
                        current_retry_delay *= 2  # Exponential backoff
            
            if last_error:
                raise last_error

            progress(0.2, "Ищу блок с отзывами")
            await _scroll_to_reviews_block(page, config.max_scrolls)

            progress(0.35, "Раскрываю список отзывов")
            await _click_show_more(page, config.max_show_more_clicks)

            progress(0.75, "Извлекаю отзывы")
            rows = await _extract_dns_reviews(page, config.max_reviews)
            frame = prepare_reviews(rows, "dns-shop.ru", config.max_reviews)
            logger.info("DNS crawler extracted rows=%s", len(frame))
            progress(1.0, f"Готово: загружено {len(frame)} отзывов")
            return frame
        finally:
            await context.close()
            await browser.close()


async def _scroll_to_reviews_block(page, max_scrolls: int) -> None:
    review_anchor_selectors = [
        "text=Отзывы",
        "text=отзывы",
        "[data-role*='review']",
        "[class*='review']",
    ]
    for step in range(max_scrolls):
        for selector in review_anchor_selectors:
            if await page.locator(selector).count() > 0:
                logger.info("Reviews anchor found on scroll step=%s selector=%s", step + 1, selector)
                return
        await page.mouse.wheel(0, random.randint(700, 1200))
        await timer.sleep(0.8, 1.5)


async def _click_show_more(page, max_clicks: int) -> None:
    clicked = 0
    for _ in range(max_clicks):
        button = page.get_by_text("Показать ещё")
        if await button.count() == 0:
            break
        try:
            await button.first.scroll_into_view_if_needed()
            await timer.sleep(0.6, 1.2)
            await button.first.click(timeout=4000)
            clicked += 1
            logger.info("Clicked 'Показать ещё' #%s", clicked)
            await timer.sleep(1.5, 2.5)
        except Exception as exc:
            logger.info("Show more click stopped: %s", exc)
            break


async def _extract_dns_reviews(page, limit: int) -> list[dict]:
    selectors = [
        "[class*='review']",
        "[data-role*='review']",
        "[data-code*='review']",
        "article",
    ]
    texts: list[str] = []
    for selector in selectors:
        nodes = page.locator(selector)
        count = await nodes.count()
        logger.info("DNS extract selector=%s nodes=%s", selector, count)
        for index in range(min(count, 80)):
            text = clean_text(await nodes.nth(index).inner_text())
            if looks_like_review(text):
                texts.append(text)
                if len(texts) >= limit:
                    break
        if len(texts) >= limit:
            break

    deduped = _deduplicate(texts)[:limit]
    return [{"text": text} for text in deduped]


def _is_blocked_page(title: str) -> bool:
    title_normalized = title.lower().strip()
    blocked_tokens = ["http 403", "доступ ограничен", "captcha", "вы робот"]
    return any(token in title_normalized for token in blocked_tokens)


def _deduplicate(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        normalized = item.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(item)
    return result


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
