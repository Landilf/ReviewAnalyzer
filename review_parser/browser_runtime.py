from __future__ import annotations

import os
import shutil
from typing import Any


SYSTEM_CHROMIUM_COMMANDS = ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable")


def resolve_chromium_executable() -> str | None:
    """Returns an explicitly configured or locally installed Chromium executable."""
    configured_path = os.getenv("PLAYWRIGHT_CHROMIUM_EXECUTABLE")
    if configured_path:
        return configured_path

    for command in SYSTEM_CHROMIUM_COMMANDS:
        executable_path = shutil.which(command)
        if executable_path:
            return executable_path
    return None


def log_static_resource_failures(page: Any, logger: Any) -> None:
    """Log failed stylesheet and script requests to diagnose incomplete pages."""
    watched_types = {"stylesheet", "script"}

    def on_response(response: Any) -> None:
        request = response.request
        if request.resource_type in watched_types and response.status >= 400:
            logger.warning(
                "Static resource blocked: type=%s status=%s url=%s",
                request.resource_type,
                response.status,
                response.url,
            )

    def on_request_failed(request: Any) -> None:
        if request.resource_type in watched_types:
            logger.warning(
                "Static resource request failed: type=%s error=%s url=%s",
                request.resource_type,
                request.failure or "unknown error",
                request.url,
            )

    page.on("response", on_response)
    page.on("requestfailed", on_request_failed)
