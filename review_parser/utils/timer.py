import asyncio
import random
from app_logger import get_logger

logger = get_logger("parser.utils.timer")

class Timer:
    """
    Управляет искусственными задержками со случайным разбросом для имитации поведения человека.
    """

    def __init__(self, variance: float = 0.25):
        """
        Args:
            variance (float): Процент отклонения от базового времени (по умолчанию 25%).
        """
        self.variance = variance

    async def sleep(self, min_s: float, max_s: float):
        """
        Ожидает в течение случайного промежутка времени в интервале [min_s, max_s]
        плюс/минус настроенный процент отклонения (variance).

        Args:
            min_s (float): Минимальное базовое время ожидания.
            max_s (float): Максимальное базовое время ожидания.
        """
        if min_s > max_s:
            min_s, max_s = max_s, min_s

        # 1. Выбираем базовое время из предоставленного интервала
        base_time = random.uniform(min_s, max_s)

        # 2. Вычисляем пределы отклонения
        delta = base_time * self.variance

        # 3. Применяем отклонение (случайно добавляем или вычитаем в пределах дельты)
        final_time = base_time + random.uniform(-delta, delta)

        # Убеждаемся, что время ожидания никогда не будет отрицательным или слишком малым
        final_time = max(0.5, final_time)

        logger.debug(
            f"Sleeping for {final_time:.2f}s (Base: {base_time:.2f}s +/- {self.variance * 100}%)"
        )
        await asyncio.sleep(final_time)
