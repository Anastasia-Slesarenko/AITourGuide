# tests/unit/test_calibration.py
"""Юнит-тесты калибровки уверенности (src/services/calibration.py)."""

import json

import pytest

from src.services.calibration import ConfidenceCalibrator


class TestCalibrateWithoutCurve:
    """Без кривой calibrate() — тождество (сырой p_yes без изменений)."""

    def test_identity_when_no_curve(self):
        cal = ConfidenceCalibrator(curve_path=None)
        for p in (0.0, 0.25, 0.5, 0.9, 1.0):
            assert cal.calibrate(p) == p

    def test_identity_when_file_missing(self):
        """Файл кривой не найден — не падаем, отдаём сырой p_yes."""
        cal = ConfidenceCalibrator(curve_path="/no/such/curve.json")
        assert cal.calibrate(0.8) == 0.8


class TestBands:
    """Бэнд по калиброванному P(correct): границы 0.6 / 0.3."""

    @pytest.mark.parametrize(
        ("conf", "expected"),
        [
            (1.0, "high"),
            (0.6, "high"),
            (0.599, "medium"),
            (0.3, "medium"),
            (0.299, "low"),
            (0.0, "low"),
        ],
    )
    def test_band_boundaries(self, conf, expected):
        cal = ConfidenceCalibrator(curve_path=None, band_high=0.6, band_medium=0.3)
        assert cal.band(conf) == expected


def _write_curve(tmp_path, xs, ys):
    """Пишет isotonic-кривую (x, y) во временный JSON, возвращает путь."""
    path = tmp_path / "curve.json"
    path.write_text(json.dumps({"x": xs, "y": ys}), encoding="utf-8")
    return str(path)


class TestCalibrateWithCurve:
    """Линейная интерполяция по загруженной кривой."""

    def test_exact_knot_points(self, tmp_path):
        cal = ConfidenceCalibrator(
            curve_path=_write_curve(tmp_path, [0.0, 0.5, 1.0], [0.0, 0.25, 1.0])
        )
        assert cal.calibrate(0.0) == pytest.approx(0.0)
        assert cal.calibrate(0.5) == pytest.approx(0.25)
        assert cal.calibrate(1.0) == pytest.approx(1.0)

    def test_linear_interpolation_between_knots(self, tmp_path):
        cal = ConfidenceCalibrator(
            curve_path=_write_curve(tmp_path, [0.0, 0.5, 1.0], [0.0, 0.25, 1.0])
        )
        # между (0.0, 0.0) и (0.5, 0.25): линейно -> 0.125
        assert cal.calibrate(0.25) == pytest.approx(0.125)
        # между (0.5, 0.25) и (1.0, 1.0): 0.25 + 0.75 * 0.5 = 0.625
        assert cal.calibrate(0.75) == pytest.approx(0.625)

    def test_clamps_outside_range(self, tmp_path):
        cal = ConfidenceCalibrator(
            curve_path=_write_curve(tmp_path, [0.1, 0.9], [0.05, 0.8])
        )
        assert cal.calibrate(0.0) == pytest.approx(0.05)  # <= xs[0] -> ys[0]
        assert cal.calibrate(1.0) == pytest.approx(0.80)  # >= xs[-1] -> ys[-1]

    def test_corrects_overconfidence(self, tmp_path):
        """Переуверенный p_yes=0.9 калибруется вниз (как в README: 0.9->0.49)."""
        cal = ConfidenceCalibrator(
            curve_path=_write_curve(tmp_path, [0.0, 0.9, 1.0], [0.0, 0.49, 1.0])
        )
        assert cal.calibrate(0.9) == pytest.approx(0.49)

    def test_monotonic_preserves_order(self, tmp_path):
        """Кривая монотонна: порядок входов сохраняется в выходах."""
        cal = ConfidenceCalibrator(
            curve_path=_write_curve(tmp_path, [0.0, 0.4, 0.7, 1.0], [0.0, 0.1, 0.5, 1.0])
        )
        inputs = [0.05, 0.2, 0.45, 0.6, 0.85, 0.95]
        outputs = [cal.calibrate(p) for p in inputs]
        assert outputs == sorted(outputs)


class TestBrokenCurveFallsBackToIdentity:
    """Битая/пустая кривая — тождество, без исключений."""

    def test_empty_arrays(self, tmp_path):
        cal = ConfidenceCalibrator(curve_path=_write_curve(tmp_path, [], []))
        assert cal.calibrate(0.7) == 0.7

    def test_length_mismatch(self, tmp_path):
        cal = ConfidenceCalibrator(curve_path=_write_curve(tmp_path, [0.0, 1.0], [0.0]))
        assert cal.calibrate(0.7) == 0.7

    def test_malformed_json(self, tmp_path):
        path = tmp_path / "curve.json"
        path.write_text("{not valid json", encoding="utf-8")
        cal = ConfidenceCalibrator(curve_path=str(path))
        assert cal.calibrate(0.7) == 0.7
