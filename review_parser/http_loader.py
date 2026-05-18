from __future__ import annotations

from app_logger import get_logger

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)

logger = get_logger("parser.http")


def download_page(url: str) -> str:
    try:
        import requests
    except ImportError as exc:
        raise ValueError("Для загрузки отзывов по ссылке установите зависимость `requests`.") from exc

    logger.info("HTTP GET %s", url)
    response = requests.get(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
        },
        timeout=20,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        status = response.status_code
        logger.warning("HTTP GET failed: url=%s status=%s body_length=%s", url, status, len(response.text))
        if status in {401, 403}:
            raise ValueError(
                f"Сайт вернул HTTP {status}: доступ к странице закрыт для простой автоматической загрузки."
            ) from exc
        raise ValueError(f"Сайт вернул HTTP {status}. Не удалось загрузить страницу товара.") from exc
    logger.info("HTTP GET success: url=%s status=%s body_length=%s", url, response.status_code, len(response.text))
    return response.text
