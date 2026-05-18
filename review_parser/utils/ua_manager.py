import random


class UserAgentManager:
    """
    Менеджер для процедурной генерации реалистичных строк User-Agent.

    Комбинирует идентификаторы ОС, версии движков и версии браузеров для создания
    уникальных отпечатков, минимизируя вероятность обнаружения бот-защитой.
    """

    # --- Компоненты ---

    OS_PLATFORMS = {
        "win": [
            "Windows NT 10.0; Win64; x64",
            "Windows NT 11.0; Win64; x64",
        ],
        "mac": [
            "Macintosh; Intel Mac OS X 10_15_7",
            "Macintosh; Intel Mac OS X 11_6",
            "Macintosh; Intel Mac OS X 12_6",
            "Macintosh; Intel Mac OS X 13_5",
            "Macintosh; Intel Mac OS X 14_2",
        ],
        "linux": ["X11; Linux x86_64", "X11; Ubuntu; Linux x86_64", "X11; Fedora; Linux x86_64"],
    }

    # Валидные диапазоны для "современных" браузеров (Конец 2023 - 2024+)
    CHROME_VERSIONS = list(range(120, 126))
    FIREFOX_VERSIONS = list(range(121, 126))

    # Версии Safari и соответствующие им версии WebKit
    SAFARI_VERSIONS = {
        "17.0": "605.1.15",
        "17.1": "605.1.15",
        "17.2": "605.1.15",
        "17.3": "605.1.15",
    }

    def get_random_ua(self) -> str:
        """
        Конструирует свежую строку User-Agent.

        Returns:
            str: Сгенерированная строка User-Agent.
        """
        platform = random.choice(["win", "mac", "linux"])

        # Выбор браузера на основе платформы
        # Windows/Linux: Chrome (70%), Firefox (30%)
        # Mac: Chrome (40%), Safari (40%), Firefox (20%)

        if platform == "mac":
            browser = random.choices(["chrome", "safari", "firefox"], weights=[40, 40, 20])[0]
        else:
            browser = random.choices(["chrome", "firefox"], weights=[70, 30])[0]

        if browser == "chrome":
            return self._build_chrome(platform)
        elif browser == "firefox":
            return self._build_firefox(platform)
        elif browser == "safari":
            return self._build_safari()

        return self._build_chrome("win")  # Запасной вариант

    def _build_chrome(self, platform_key: str) -> str:
        os_str = random.choice(self.OS_PLATFORMS[platform_key])
        version = random.choice(self.CHROME_VERSIONS)
        # Немного рандомизируем номер сборки: Major.0.Build.0
        build = random.randint(5000, 6000)
        patch = random.randint(100, 200)
        full_ver = f"{version}.0.{build}.{patch}"

        return f"Mozilla/5.0 ({os_str}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{full_ver} Safari/537.36"

    def _build_firefox(self, platform_key: str) -> str:
        os_str = random.choice(self.OS_PLATFORMS[platform_key])
        version = random.choice(self.FIREFOX_VERSIONS)
        # Firefox обычно имеет отдельную версию Gecko, часто основанную на дате
        gecko_date = "20100101"

        return f"Mozilla/5.0 ({os_str}; rv:{version}.0) Gecko/{gecko_date} Firefox/{version}.0"

    def _build_safari(self) -> str:
        # Safari обычно встречается только на Mac
        os_str = random.choice(self.OS_PLATFORMS["mac"])
        version_key = random.choice(list(self.SAFARI_VERSIONS.keys()))
        webkit_ver = self.SAFARI_VERSIONS[version_key]

        return (
            f"Mozilla/5.0 ({os_str}) AppleWebKit/{webkit_ver} (KHTML, like Gecko) "
            f"Version/{version_key} Safari/{webkit_ver}"
        )
