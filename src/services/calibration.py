# src/services/calibration.py
"""Калибровка отдаваемой уверенности по isotonic-кривой."""

import bisect
import json
import logging

logger = logging.getLogger(__name__)


class ConfidenceCalibrator:
    """
    Переводит сырой p_yes реранкера в калиброванный P(correct) по isotonic-кривой
    (линейная интерполяция) и относит результат к бэнду high/medium/low.

    Кривая монотонна, поэтому решения об отсечке (принятые на сыром p_yes) не
    меняет. Если кривая не загружена — calibrate() возвращает вход без изменений.
    """

    def __init__(
        self,
        curve_path: str | None = None,
        band_high: float = 0.6,
        band_medium: float = 0.3,
    ) -> None:
        self._x: list[float] | None = None
        self._y: list[float] | None = None
        self.band_high = band_high
        self.band_medium = band_medium
        if curve_path:
            self._load(curve_path)

    def _load(self, path: str) -> None:
        """Загружает isotonic-кривую (x, y) из JSON."""
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            xs, ys = data.get("x", []), data.get("y", [])
            if xs and ys and len(xs) == len(ys):
                self._x, self._y = xs, ys
                logger.info(
                    f"Калибровка уверенности: isotonic ({len(xs)} точек) из {path}"
                )
            else:
                logger.warning(
                    f"Калибровка: пустая/битая кривая в {path}, отдаём сырой p_yes"
                )
        except FileNotFoundError:
            logger.error(
                f"Калибровка включена, но файл кривой не найден: {path}. "
                f"Уверенность и бэнды будут по сырому p_yes (переуверенность)."
            )
        except Exception as e:
            logger.warning(f"Калибровка: не удалось загрузить {path} ({e})")

    def calibrate(self, p_yes: float) -> float:
        """
        p_yes → P(correct) по кривой (линейная интерполяция).

        Если кривая не загружена — возвращает p_yes без изменений.
        """
        xs, ys = self._x, self._y
        if not xs or not ys:
            return p_yes
        if p_yes <= xs[0]:
            return ys[0]
        if p_yes >= xs[-1]:
            return ys[-1]
        i = bisect.bisect_left(xs, p_yes)
        x0, x1 = xs[i - 1], xs[i]
        y0, y1 = ys[i - 1], ys[i]
        if x1 == x0:
            return y1
        return y0 + (y1 - y0) * (p_yes - x0) / (x1 - x0)

    def band(self, calibrated_conf: float) -> str:
        """Бэнд уверенности для UI по калиброванному P(correct): high/medium/low."""
        if calibrated_conf >= self.band_high:
            return "high"
        if calibrated_conf >= self.band_medium:
            return "medium"
        return "low"
