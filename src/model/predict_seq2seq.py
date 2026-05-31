from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import joblib
import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.db.supabase_client import load_master_dataset  # noqa: E402
from src.model.seq2seq_model import Seq2SeqTransformer  # noqa: E402
from src.utils.normal_rate import get_normal_rate_for_date  # noqa: E402
from src.utils.risk_policy import classify_reservoir_stage  # noqa: E402


DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "transformer_v4_seq2seq"

DATE_COL = "날짜"
RESERVOIR_COL = "저수지명"
CITY_COL = "시군"
TARGET_COL = "저수율_5일EMA"
OBSERVED_RATE_COL = "저수율"
RAIN_COL = "강수량"
TEMP_COL = "평균기온(℃)"
TEMP_ROLLING_COL = "기온_7일평균"
RAIN_3D_COL = "강수량_3일누적"
RAIN_7D_COL = "강수량_7일누적"
CAPACITY_COL = "유효저수량(천m3)"
LAT_COL = "위도"
LON_COL = "경도"

FEATURE_COL_FALLBACK = [
    TARGET_COL,
    TEMP_ROLLING_COL,
    RAIN_3D_COL,
    RAIN_7D_COL,
    "SPI3",
    "SPI6",
    CAPACITY_COL,
    LAT_COL,
    LON_COL,
    "delta_1d_lag1",
    "delta_7d_mean",
    "저수율_rolling_std14",
    "month_sin",
    "month_cos",
    "day_of_year_sin",
    "day_of_year_cos",
]


@dataclass(frozen=True)
class Seq2SeqArtifacts:
    model: Seq2SeqTransformer
    scaler: object
    config: dict
    residual_stats: dict
    device: torch.device


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Seq2Seq 모델 파일이 없습니다: {path}. "
            "models/transformer_v4_seq2seq 폴더에 모델 artifact를 배치하세요."
        )


def load_seq2seq_artifacts(
    model_dir: Optional[Path | str] = None,
    device: Optional[str | torch.device] = None,
) -> Seq2SeqArtifacts:
    selected_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    config_path = selected_dir / "transformer_v4_seq2seq_config.json"
    scaler_path = selected_dir / "transformer_v4_seq2seq_scaler.joblib"
    model_path = selected_dir / "transformer_v4_seq2seq.pt"
    residual_path = selected_dir / "transformer_v4_seq2seq_residual_stats.json"

    for path in [config_path, scaler_path, model_path, residual_path]:
        _require_file(path)

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    with residual_path.open("r", encoding="utf-8") as file:
        residual_stats = json.load(file)

    scaler = joblib.load(scaler_path)
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = Seq2SeqTransformer.from_config(config).to(selected_device)
    state_dict = torch.load(model_path, map_location=selected_device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)
    model.eval()

    return Seq2SeqArtifacts(
        model=model,
        scaler=scaler,
        config=config,
        residual_stats=residual_stats,
        device=selected_device,
    )


def _safe_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _scenario_value(scenario: Optional[dict], key: str, default: float) -> float:
    if not scenario:
        return default
    value = scenario.get(key, default)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def scenario_has_effect(scenario: Optional[dict]) -> bool:
    if not scenario:
        return False
    return (
        _scenario_value(scenario, "rainfall_multiplier", 1.0) != 1.0
        or (scenario.get("rainfall_sum_30d") is not None if scenario else False)
        or _scenario_value(scenario, "temperature_delta", 0.0) != 0.0
        or _scenario_value(scenario, "spi3_delta", 0.0) != 0.0
        or _scenario_value(scenario, "spi6_delta", 0.0) != 0.0
    )


def normalized_scenario(scenario: Optional[dict]) -> dict[str, float]:
    return {
        "rainfall_multiplier": _scenario_value(scenario, "rainfall_multiplier", 1.0),
        "rainfall_sum_30d": np.nan if not scenario or scenario.get("rainfall_sum_30d") is None else _scenario_value(scenario, "rainfall_sum_30d", np.nan),
        "temperature_delta": _scenario_value(scenario, "temperature_delta", 0.0),
        "spi3_delta": _scenario_value(scenario, "spi3_delta", 0.0),
        "spi6_delta": _scenario_value(scenario, "spi6_delta", 0.0),
    }


def apply_scenario_to_recent_history(
    prepared: pd.DataFrame,
    scenario: Optional[dict],
    as_of_date: Optional[str | pd.Timestamp],
    apply_days: int = 60,
) -> pd.DataFrame:
    """Apply scenario values to the model input window.

    This scenario method does not inject future 30-day weather features into the
    model. To reuse the existing trained Seq2Seq model without retraining, it
    adjusts rainfall, temperature, and SPI values in the recent input window and
    then recalculates derived rolling features before prediction.
    """
    if not scenario_has_effect(scenario):
        return prepared

    adjusted = prepared.copy()
    values = normalized_scenario(scenario)
    basis_date = pd.Timestamp(as_of_date) if as_of_date is not None else adjusted[DATE_COL].max()
    window_start = basis_date - pd.Timedelta(days=apply_days - 1)
    window_mask = (adjusted[DATE_COL] >= window_start) & (adjusted[DATE_COL] <= basis_date)

    if RAIN_COL in adjusted.columns:
        adjusted[RAIN_COL] = pd.to_numeric(adjusted[RAIN_COL], errors="coerce")
        if np.isnan(values["rainfall_sum_30d"]):
            adjusted.loc[window_mask, RAIN_COL] = adjusted.loc[window_mask, RAIN_COL] * values["rainfall_multiplier"]
        else:
            for _, index in adjusted.loc[window_mask].groupby(RESERVOIR_COL).groups.items():
                recent_sum = float(adjusted.loc[index, RAIN_COL].tail(30).fillna(0.0).sum())
                multiplier = values["rainfall_sum_30d"] / recent_sum if recent_sum > 0 else 1.0
                adjusted.loc[index, RAIN_COL] = adjusted.loc[index, RAIN_COL] * multiplier
    if TEMP_COL in adjusted.columns:
        adjusted.loc[window_mask, TEMP_COL] = (
            pd.to_numeric(adjusted.loc[window_mask, TEMP_COL], errors="coerce") + values["temperature_delta"]
        )
    if "SPI3" in adjusted.columns:
        adjusted.loc[window_mask, "SPI3"] = pd.to_numeric(adjusted.loc[window_mask, "SPI3"], errors="coerce") + values["spi3_delta"]
    if "SPI6" in adjusted.columns:
        adjusted.loc[window_mask, "SPI6"] = pd.to_numeric(adjusted.loc[window_mask, "SPI6"], errors="coerce") + values["spi6_delta"]

    for key, value in values.items():
        adjusted[f"scenario_{key}"] = value
    return adjusted


def _recalculate_rolling_features(prepared: pd.DataFrame) -> pd.DataFrame:
    prepared = prepared.sort_values([RESERVOIR_COL, DATE_COL]).reset_index(drop=True)
    group = prepared.groupby(RESERVOIR_COL, sort=False)

    if TARGET_COL not in prepared.columns and OBSERVED_RATE_COL in prepared.columns:
        prepared[TARGET_COL] = group[OBSERVED_RATE_COL].transform(
            lambda s: _safe_numeric(s).ewm(span=5, adjust=False).mean()
        )
    if TEMP_COL in prepared.columns:
        prepared[TEMP_ROLLING_COL] = group[TEMP_COL].transform(
            lambda s: _safe_numeric(s).rolling(7, min_periods=1).mean()
        )
    if RAIN_COL in prepared.columns:
        prepared[RAIN_3D_COL] = group[RAIN_COL].transform(
            lambda s: _safe_numeric(s).rolling(3, min_periods=1).sum()
        )
        prepared[RAIN_7D_COL] = group[RAIN_COL].transform(
            lambda s: _safe_numeric(s).rolling(7, min_periods=1).sum()
        )

    rate_group = prepared.groupby(RESERVOIR_COL, sort=False)[TARGET_COL]
    prepared["delta_1d_lag1"] = rate_group.diff().groupby(prepared[RESERVOIR_COL]).shift(1).fillna(0.0)
    prepared["delta_7d_mean"] = group[TARGET_COL].transform(
        lambda s: _safe_numeric(s).diff().rolling(7, min_periods=1).mean()
    ).fillna(0.0)
    prepared["저수율_rolling_std14"] = group[TARGET_COL].transform(
        lambda s: _safe_numeric(s).rolling(14, min_periods=2).std()
    ).fillna(0.0)
    return prepared


def prepare_seq2seq_dataframe(
    df: pd.DataFrame,
    as_of_date: Optional[str | pd.Timestamp] = None,
    scenario: Optional[dict] = None,
) -> pd.DataFrame:
    required = [DATE_COL, RESERVOIR_COL, OBSERVED_RATE_COL, RAIN_COL]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"seq2seq 예측에 필요한 컬럼이 없습니다: {missing}")

    prepared = df.copy()
    prepared[DATE_COL] = pd.to_datetime(prepared[DATE_COL], errors="coerce")
    if prepared[DATE_COL].isna().any():
        raise ValueError("날짜 컬럼에 datetime 변환 실패 값이 있습니다.")

    if as_of_date is not None:
        prepared = prepared[prepared[DATE_COL] <= pd.Timestamp(as_of_date)].copy()

    prepared = prepared.sort_values([RESERVOIR_COL, DATE_COL]).reset_index(drop=True)
    prepared = apply_scenario_to_recent_history(prepared, scenario=scenario, as_of_date=as_of_date)

    group = prepared.groupby(RESERVOIR_COL, sort=False)
    numeric_base_cols = [
        OBSERVED_RATE_COL,
        TARGET_COL,
        TEMP_COL,
        TEMP_ROLLING_COL,
        RAIN_COL,
        RAIN_3D_COL,
        RAIN_7D_COL,
        "SPI3",
        "SPI6",
        CAPACITY_COL,
        LAT_COL,
        LON_COL,
    ]
    for column in numeric_base_cols:
        if column in prepared.columns:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
            prepared[column] = group[column].ffill()
            median = prepared[column].median()
            prepared[column] = prepared[column].fillna(0.0 if pd.isna(median) else median)

    prepared = _recalculate_rolling_features(prepared)

    month = prepared[DATE_COL].dt.month
    day_of_year = prepared[DATE_COL].dt.dayofyear
    prepared["month_sin"] = np.sin(2 * np.pi * month / 12)
    prepared["month_cos"] = np.cos(2 * np.pi * month / 12)
    prepared["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 366)
    prepared["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 366)

    for column in FEATURE_COL_FALLBACK:
        if column not in prepared.columns:
            prepared[column] = 0.0
        prepared[column] = _safe_numeric(prepared[column])

    return prepared


def resolve_feature_columns(config: dict, df: pd.DataFrame) -> list[str]:
    configured = list(config.get("feature_cols") or config.get("feature_columns") or [])
    if configured and all(column in df.columns for column in configured):
        return configured
    if all(column in df.columns for column in FEATURE_COL_FALLBACK):
        return FEATURE_COL_FALLBACK.copy()
    missing = [column for column in FEATURE_COL_FALLBACK if column not in df.columns]
    raise ValueError(f"seq2seq feature 컬럼을 구성할 수 없습니다. 누락: {missing}")


def _get_reservoir_id(config: dict, reservoir_name: str) -> int:
    mapping = config.get("reservoir_id_map") or config.get("reservoir_to_id") or {}
    if reservoir_name not in mapping:
        raise ValueError(f"모델 config에 저수지 '{reservoir_name}' id가 없습니다.")
    return int(mapping[reservoir_name])


def _mae_for_day(residual_stats: dict, day_ahead: int) -> float:
    mae_by_day = residual_stats.get("mae_by_day", {})
    return float(mae_by_day.get(str(day_ahead), residual_stats.get("overall_mae", 0.0)))


def _city_map(master_df: pd.DataFrame) -> dict[str, str]:
    if CITY_COL not in master_df.columns:
        return {}
    info = master_df[[RESERVOIR_COL, CITY_COL]].dropna().drop_duplicates(subset=[RESERVOIR_COL])
    return dict(zip(info[RESERVOIR_COL], info[CITY_COL]))


def predict_seq2seq_forecasts(
    master_df: pd.DataFrame,
    artifacts: Optional[Seq2SeqArtifacts] = None,
    as_of_date: Optional[str | pd.Timestamp] = None,
    reservoir_names: Optional[Iterable[str]] = None,
    model_dir: Optional[Path | str] = None,
    scenario: Optional[dict] = None,
) -> pd.DataFrame:
    artifacts = artifacts or load_seq2seq_artifacts(model_dir=model_dir)
    config = artifacts.config
    seq_len = int(config.get("seq_len", 60))
    pred_len = int(config.get("pred_len", 30))
    scenario_values = normalized_scenario(scenario)

    df = prepare_seq2seq_dataframe(master_df, as_of_date=as_of_date, scenario=scenario)
    basis_date = pd.Timestamp(as_of_date) if as_of_date is not None else df[DATE_COL].max()
    feature_cols = resolve_feature_columns(config, df)
    selected = list(reservoir_names) if reservoir_names is not None else sorted(df[RESERVOIR_COL].dropna().unique())
    cities = _city_map(master_df)

    rows = []
    with torch.no_grad():
        for reservoir_name in selected:
            reservoir_df = df[df[RESERVOIR_COL] == reservoir_name].sort_values(DATE_COL)
            if len(reservoir_df) < seq_len:
                continue

            window = reservoir_df.tail(seq_len)
            values = window[feature_cols].astype(float).to_numpy(dtype=np.float32)
            scaled = artifacts.scaler.transform(values).astype(np.float32)
            x_tensor = torch.as_tensor(scaled, dtype=torch.float32, device=artifacts.device).unsqueeze(0)
            reservoir_id = torch.tensor(
                [_get_reservoir_id(config, reservoir_name)],
                dtype=torch.long,
                device=artifacts.device,
            )
            prediction = artifacts.model(x_tensor, reservoir_id).detach().cpu().numpy().reshape(-1)
            prediction = np.clip(prediction, 0.0, 100.0)

            city = cities.get(reservoir_name)
            for index, predicted_rate in enumerate(prediction[:pred_len], start=1):
                mae = _mae_for_day(artifacts.residual_stats, index)
                forecast_date = basis_date + pd.Timedelta(days=index)
                normal_rate = get_normal_rate_for_date(master_df, reservoir_name, forecast_date, target_col=TARGET_COL)
                stage = classify_reservoir_stage(predicted_rate, city=city, normal_rate=normal_rate)
                rows.append(
                    {
                        "as_of_date": basis_date.date().isoformat(),
                        RESERVOIR_COL: reservoir_name,
                        CITY_COL: city,
                        "forecast_date": forecast_date.date().isoformat(),
                        "day_ahead": index,
                        "predicted_rate": float(predicted_rate),
                        "lower_bound": float(max(0.0, predicted_rate - mae)),
                        "upper_bound": float(min(100.0, predicted_rate + mae)),
                        "stage_code": stage["stage_code"],
                        "stage_label": stage["stage_label"],
                        "stage_basis": stage["basis"],
                        "stage_basis_value": stage["basis_value"],
                        "normal_rate": stage["normal_rate"],
                        "normal_ratio": stage["normal_ratio"],
                        "threshold_description": stage["threshold_description"],
                        "risk_code": stage["stage_code"],
                        "risk_label": stage["stage_label"],
                        "model_name": config.get("model_name", "transformer_v4_seq2seq"),
                        "scenario_rainfall_multiplier": scenario_values["rainfall_multiplier"],
                        "scenario_rainfall_sum_30d": scenario_values["rainfall_sum_30d"],
                        "scenario_temperature_delta": scenario_values["temperature_delta"],
                        "scenario_spi3_delta": scenario_values["spi3_delta"],
                        "scenario_spi6_delta": scenario_values["spi6_delta"],
                    }
                )

    return pd.DataFrame(rows)


def summarize_min_rates(forecast_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        RESERVOIR_COL,
        CITY_COL,
        "as_of_date",
        "min_forecast_date",
        "predicted_min_rate",
        "lower_bound_at_min",
        "stage_code",
        "stage_label",
        "stage_basis",
        "stage_basis_value",
        "normal_rate_at_min",
        "normal_ratio_at_min",
        "threshold_description",
        "risk_code",
        "risk_label",
        "scenario_rainfall_multiplier",
        "scenario_rainfall_sum_30d",
        "scenario_temperature_delta",
        "scenario_spi3_delta",
        "scenario_spi6_delta",
    ]
    if forecast_df.empty:
        return pd.DataFrame(columns=columns)

    idx = forecast_df.groupby(RESERVOIR_COL)["predicted_rate"].idxmin()
    summary = forecast_df.loc[idx].copy()
    summary = summary.rename(
        columns={
            "forecast_date": "min_forecast_date",
            "predicted_rate": "predicted_min_rate",
            "lower_bound": "lower_bound_at_min",
            "normal_rate": "normal_rate_at_min",
            "normal_ratio": "normal_ratio_at_min",
        }
    )
    return summary[[column for column in columns if column in summary.columns]].reset_index(drop=True)


def predict_seq2seq_min_rates(
    source: str = "auto",
    as_of_date: Optional[str | pd.Timestamp] = None,
    reservoir_names: Optional[Iterable[str]] = None,
    model_dir: Optional[Path | str] = None,
    scenario: Optional[dict] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    master_df = load_master_dataset(source=source)
    forecast_df = predict_seq2seq_forecasts(
        master_df=master_df,
        as_of_date=as_of_date,
        reservoir_names=reservoir_names,
        model_dir=model_dir,
        scenario=scenario,
    )
    return forecast_df, summarize_min_rates(forecast_df)


if __name__ == "__main__":
    try:
        daily_forecasts, min_summary = predict_seq2seq_min_rates()
        print("=== Seq2Seq daily forecast sample ===")
        print(daily_forecasts.head(30).to_string(index=False))
        print("\n=== Seq2Seq min-rate summary ===")
        print(min_summary.to_string(index=False))

        scenario = {
            "rainfall_multiplier": 0.5,
            "temperature_delta": 2.0,
            "spi3_delta": -1.0,
            "spi6_delta": -0.5,
        }
        _, scenario_summary = predict_seq2seq_min_rates(reservoir_names=["오봉"], scenario=scenario)
        print("\n=== Scenario min-rate summary sample ===")
        print(scenario_summary.to_string(index=False))
    except Exception as error:
        print("Seq2Seq 예측 실행에 실패했습니다.")
        print(error)
