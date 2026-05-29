from __future__ import annotations

from typing import Any

import pandas as pd


DATE_COL = "날짜"
RESERVOIR_COL = "저수지명"


def _prepare_base(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    required = [RESERVOIR_COL, DATE_COL, target_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"평년 저수율 계산에 필요한 컬럼이 없습니다: {missing}")

    prepared = df.copy()
    prepared[DATE_COL] = pd.to_datetime(prepared[DATE_COL], errors="coerce")
    prepared[target_col] = pd.to_numeric(prepared[target_col], errors="coerce")
    prepared = prepared.dropna(subset=[RESERVOIR_COL, DATE_COL, target_col])
    prepared["month_day"] = prepared[DATE_COL].dt.strftime("%m-%d")
    prepared["month"] = prepared[DATE_COL].dt.month
    return prepared


def add_normal_rate_columns(df: pd.DataFrame, target_col: str = "저수율_5일EMA") -> pd.DataFrame:
    """Add normal_rate based on reservoir and month-day historical averages.

    운영 배포 시에는 기준일 이후 데이터 제외 필요.
    """
    prepared = _prepare_base(df, target_col)
    result = df.copy()
    result[DATE_COL] = pd.to_datetime(result[DATE_COL], errors="coerce")
    result["month_day"] = result[DATE_COL].dt.strftime("%m-%d")
    result["month"] = result[DATE_COL].dt.month

    day_mean = (
        prepared.groupby([RESERVOIR_COL, "month_day"], as_index=False)[target_col]
        .mean()
        .rename(columns={target_col: "normal_rate_day"})
    )
    month_mean = (
        prepared.groupby([RESERVOIR_COL, "month"], as_index=False)[target_col]
        .mean()
        .rename(columns={target_col: "normal_rate_month"})
    )
    reservoir_mean = (
        prepared.groupby(RESERVOIR_COL, as_index=False)[target_col]
        .mean()
        .rename(columns={target_col: "normal_rate_reservoir"})
    )

    result = result.merge(day_mean, on=[RESERVOIR_COL, "month_day"], how="left")
    result = result.merge(month_mean, on=[RESERVOIR_COL, "month"], how="left")
    result = result.merge(reservoir_mean, on=RESERVOIR_COL, how="left")
    result["normal_rate"] = result["normal_rate_day"].fillna(result["normal_rate_month"]).fillna(
        result["normal_rate_reservoir"]
    )
    return result.drop(columns=["normal_rate_day", "normal_rate_month", "normal_rate_reservoir"], errors="ignore")


def get_normal_rate_for_date(
    df: pd.DataFrame,
    reservoir_name: str,
    target_date: Any,
    target_col: str = "저수율_5일EMA",
) -> float | None:
    prepared = _prepare_base(df, target_col)
    date = pd.Timestamp(target_date)
    month_day = date.strftime("%m-%d")
    month = date.month

    reservoir_df = prepared[prepared[RESERVOIR_COL] == reservoir_name]
    if reservoir_df.empty:
        return None

    day_value = reservoir_df.loc[reservoir_df["month_day"] == month_day, target_col].mean()
    if not pd.isna(day_value):
        return float(day_value)

    month_value = reservoir_df.loc[reservoir_df["month"] == month, target_col].mean()
    if not pd.isna(month_value):
        return float(month_value)

    total_value = reservoir_df[target_col].mean()
    if not pd.isna(total_value):
        return float(total_value)
    return None
