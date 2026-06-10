from __future__ import annotations

import sys
import time
import unittest
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FailureInfo:
    test_name: str
    kind: str
    details: str


@dataclass
class RunStats:
    passed: list[str] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    failures: list[FailureInfo] = field(default_factory=list)


class PrettyTestResult(unittest.TestResult):
    def __init__(self):
        super().__init__()
        self.stats = RunStats()

    def getDescription(self, test):  # noqa: N802
        return str(test)

    def addSuccess(self, test):  # noqa: N802
        super().addSuccess(test)
        name = self.getDescription(test)
        self.stats.passed.append(name)
        print(f"✓ {name}")

    def addSkip(self, test, reason):  # noqa: N802
        super().addSkip(test, reason)
        name = self.getDescription(test)
        self.stats.skipped.append((name, reason))
        print(f"↷ {name} — пропущен: {reason}")

    def addFailure(self, test, err):  # noqa: N802
        super().addFailure(test, err)
        name = self.getDescription(test)
        self.stats.failures.append(
            FailureInfo(name, "FAIL", self._exc_info_to_string(err, test))
        )
        print(f"✗ {name}")

    def addError(self, test, err):  # noqa: N802
        super().addError(test, err)
        name = self.getDescription(test)
        self.stats.failures.append(
            FailureInfo(name, "ERROR", self._exc_info_to_string(err, test))
        )
        print(f"⚠ {name}")


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    loader = unittest.defaultTestLoader
    suite = loader.discover("tests")

    print("=" * 72)
    print("ТЕСТИРОВАНИЕ ПРОЕКТА ReviewAnalyzer")
    print("=" * 72)

    started_at = time.perf_counter()
    result = PrettyTestResult()
    suite.run(result)
    elapsed = time.perf_counter() - started_at

    print("-" * 72)
    print("Сводка")
    print(f"- Пройдено: {len(result.stats.passed)}")
    print(f"- Пропущено: {len(result.stats.skipped)}")
    print(f"- Ошибок и падений: {len(result.stats.failures)}")
    print(f"- Время выполнения: {elapsed:.3f} с")

    if result.stats.failures:
        print("-" * 72)
        print("Детали ошибок")
        for index, failure in enumerate(result.stats.failures, start=1):
            print(f"{index}. {failure.kind}: {failure.test_name}")
            print(failure.details.rstrip())
            print("-" * 72)
        print("Итог: тестирование завершено с ошибками")
        return 1

    print("Итог: все тесты успешно пройдены")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
