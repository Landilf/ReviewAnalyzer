from __future__ import annotations

import asyncio
import os
import random

from app_logger import get_logger


from review_parser.utils.ua_manager import UserAgentManager

logger = get_logger("parser.browser")
ua_manager = UserAgentManager()


async def _download_page_with_browser(url: str) -> str:
    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise ValueError("Для браузерной загрузки установите зависимость `playwright`.") from exc

    async with async_playwright() as playwright:
        executable_path = os.getenv("PLAYWRIGHT_CHROMIUM_EXECUTABLE")
        logger.info("Launching browser: executable_path=%s", executable_path or "playwright-managed")
        browser = await playwright.chromium.launch(
            headless=True,
            executable_path=executable_path,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
            ],
        )
        context = await browser.new_context(
            user_agent=ua_manager.get_random_ua(),
            viewport={"width": 1920, "height": 1080},
            locale="ru-RU",
            timezone_id="Europe/Moscow",
            java_script_enabled=True,
        )
        await context.add_init_script(
            """
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
            """
        )
        page = await context.new_page()
        try:
            logger.info("Browser goto: %s", url)
            response = await page.goto(url, wait_until="domcontentloaded", timeout=45000)
            await _soft_scroll(page)
            content = await page.content()
            title = await page.title()
            status = response.status if response else None
            logger.info("Browser loaded: url=%s status=%s title=%s body_length=%s", url, status, title, len(content))
            
            # DNS-specific and general bot-block detection
            if status in {401, 403} or "captcha" in title.lower() or "вы робот" in title.lower():
                raise ValueError(
                    f"Сайт вернул HTTP {status or 'Block'}: браузерная загрузка заблокирована (Anti-Bot). "
                    "Рекомендуется использовать прокси или увеличить задержки."
                )
            
            return content
        finally:
            await context.close()
            await browser.close()
            logger.info("Browser closed")


def download_page_with_browser(url: str) -> str:
    try:
        return asyncio.run(_download_page_with_browser(url))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_download_page_with_browser(url))
        finally:
            loop.close()


async def _soft_scroll(page) -> None:
    for _ in range(4):
        await page.mouse.wheel(0, random.randint(600, 1200))
        await page.wait_for_timeout(random.randint(700, 1400))
