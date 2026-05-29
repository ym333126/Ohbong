from __future__ import annotations

from typing import Any

import pandas as pd


STAGE_LABELS = {
    "normal": "정상",
    "attention": "관심",
    "caution": "주의",
    "alert": "경계",
    "severe": "심각",
    "unknown": "판단불가",
}

STAGE_RANKS = {
    "unknown": -1,
    "normal": 0,
    "attention": 1,
    "caution": 2,
    "alert": 3,
    "severe": 4,
}


def _to_float(value: Any) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return float(numeric)


def _stage_result(
    stage_code: str,
    basis: str,
    basis_value: float | None,
    threshold_description: str,
    rate: float | None = None,
    normal_rate: float | None = None,
    normal_ratio: float | None = None,
) -> dict[str, Any]:
    return {
        "stage_code": stage_code,
        "stage_label": STAGE_LABELS.get(stage_code, "판단불가"),
        "basis": basis,
        "basis_value": basis_value,
        "normal_rate": normal_rate,
        "rate": rate,
        "normal_ratio": normal_ratio,
        "threshold_description": threshold_description,
    }


def classify_gangneung_rate(rate: Any) -> dict[str, Any]:
    numeric_rate = _to_float(rate)
    if numeric_rate is None:
        return _stage_result("unknown", "absolute_rate", None, "강릉시 저수율 기준")

    if numeric_rate >= 35:
        stage_code = "normal"
    elif numeric_rate >= 30:
        stage_code = "attention"
    elif numeric_rate >= 25:
        stage_code = "caution"
    elif numeric_rate >= 20:
        stage_code = "alert"
    else:
        stage_code = "severe"

    return _stage_result(
        stage_code=stage_code,
        basis="absolute_rate",
        basis_value=numeric_rate,
        threshold_description="강릉시 저수율 기준",
        rate=numeric_rate,
    )


def classify_relative_ratio(ratio: Any) -> dict[str, Any]:
    numeric_ratio = _to_float(ratio)
    if numeric_ratio is None:
        return _stage_result("unknown", "normal_ratio", None, "평년 저수율 대비 비율 기준")

    if numeric_ratio >= 70:
        stage_code = "normal"
    elif numeric_ratio >= 60:
        stage_code = "attention"
    elif numeric_ratio >= 50:
        stage_code = "caution"
    elif numeric_ratio >= 40:
        stage_code = "alert"
    else:
        stage_code = "severe"

    return _stage_result(
        stage_code=stage_code,
        basis="normal_ratio",
        basis_value=numeric_ratio,
        threshold_description="평년 저수율 대비 비율 기준",
        normal_ratio=numeric_ratio,
    )


def classify_reservoir_stage(
    rate: Any,
    city: str | None = None,
    normal_rate: Any = None,
) -> dict[str, Any]:
    numeric_rate = _to_float(rate)
    numeric_normal_rate = _to_float(normal_rate)

    if city == "강릉시":
        return classify_gangneung_rate(numeric_rate)

    if numeric_rate is None or numeric_normal_rate is None or numeric_normal_rate <= 0:
        return _stage_result(
            stage_code="unknown",
            basis="unknown",
            basis_value=None,
            threshold_description="평년 저수율 정보가 없어 단계 판단 불가",
            rate=numeric_rate,
            normal_rate=numeric_normal_rate,
            normal_ratio=None,
        )

    ratio = numeric_rate / numeric_normal_rate * 100
    result = classify_relative_ratio(ratio)
    result.update(
        {
            "rate": numeric_rate,
            "normal_rate": numeric_normal_rate,
            "normal_ratio": ratio,
        }
    )
    return result


def get_stage_rank(stage_code: str | None) -> int:
    return STAGE_RANKS.get(stage_code or "unknown", -1)


def is_supply_allowed(stage_code: str | None, min_allowed_stage: str = "attention") -> bool:
    allowed_rank = get_stage_rank(min_allowed_stage)
    current_rank = get_stage_rank(stage_code)
    return 0 <= current_rank <= allowed_rank
