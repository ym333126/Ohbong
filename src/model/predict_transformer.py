import json
import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.model.train_transformer import TimeSeriesTransformer
except Exception as exc:
    raise ImportError(
        "TimeSeriesTransformer import에 실패했습니다. "
        "프로젝트 루트에서 `python src/model/predict_transformer.py`로 실행하는지 확인하세요."
    ) from exc

from src.db.supabase_client import load_master_dataset  # noqa: E402
from src.preprocessing.prepare_timeseries import (  # noqa: E402
    add_time_features,
    clean_master_dataset,
)


DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
MODEL_FILENAME = "transformer_ohbong.pt"
CONFIG_FILENAME = "model_config.json"
SCALER_FILENAME = "scaler.joblib"
RESIDUAL_STATS_FILENAME = "residual_stats.json"

RAINFALL_COLUMNS = ["강수량", "강수량_3일누적", "강수량_7일누적"]
TEMPERATURE_COLUMNS = ["평균기온(℃)", "기온_7일평균"]


def _require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{description} 파일이 없습니다: {path}. "
            "먼저 `python src/model/train_transformer.py`를 실행하세요."
        )


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_trained_model(
    model_dir: Optional[Path | str] = None,
) -> tuple[TimeSeriesTransformer, object, dict, dict, torch.device]:
    """Load trained Transformer, scaler, config, and residual statistics."""
    selected_model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    model_path = selected_model_dir / MODEL_FILENAME
    config_path = selected_model_dir / CONFIG_FILENAME
    scaler_path = selected_model_dir / SCALER_FILENAME
    residual_stats_path = selected_model_dir / RESIDUAL_STATS_FILENAME

    _require_file(model_path, "모델")
    _require_file(config_path, "모델 설정")
    _require_file(scaler_path, "scaler")
    _require_file(residual_stats_path, "residual_stats")

    config = _load_json(config_path)
    residual_stats = _load_json(residual_stats_path)
    scaler = joblib.load(scaler_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeSeriesTransformer(
        num_features=int(config["num_features"]),
        input_window=int(config["input_window"]),
        d_model=int(config.get("d_model", 64)),
        nhead=int(config.get("nhead", 4)),
        num_layers=int(config.get("num_layers", 2)),
        dim_feedforward=int(config.get("dim_feedforward", 128)),
        dropout=float(config.get("dropout", 0.1)),
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, scaler, config, residual_stats, device


def prepare_prediction_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same cleaning and time features used during training."""
    cleaned_df = clean_master_dataset(df)
    return add_time_features(cleaned_df)


def make_recent_sequence(
    df: pd.DataFrame,
    reservoir_name: str = "오봉",
    as_of_date: Optional[str | pd.Timestamp] = None,
    input_window: int = 60,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Extract the most recent input_window rows for one reservoir."""
    if input_window <= 0:
        raise ValueError("input_window는 1 이상의 정수여야 합니다.")

    reservoir_df = df[df["저수지명"] == reservoir_name].copy()
    if reservoir_df.empty:
        raise ValueError(f"저수지명 == '{reservoir_name}'인 데이터가 없습니다.")

    reservoir_df["날짜"] = pd.to_datetime(reservoir_df["날짜"], errors="coerce")
    reservoir_df = reservoir_df.sort_values("날짜")

    if as_of_date is not None:
        selected_as_of_date = pd.Timestamp(as_of_date)
        reservoir_df = reservoir_df[reservoir_df["날짜"] <= selected_as_of_date]
    elif reservoir_df["날짜"].notna().any():
        selected_as_of_date = pd.Timestamp(reservoir_df["날짜"].max())
    else:
        raise ValueError("날짜 컬럼에 유효한 값이 없습니다.")

    if len(reservoir_df) < input_window:
        raise ValueError(
            f"{reservoir_name} 저수지의 기준일 {selected_as_of_date.date()} 이하 데이터가 "
            f"{len(reservoir_df)}행뿐입니다. 최근 {input_window}일 sequence를 만들 수 없습니다."
        )

    sequence_df = reservoir_df.tail(input_window).copy().reset_index(drop=True)
    basis_date = pd.Timestamp(sequence_df["날짜"].iloc[-1])
    return sequence_df, basis_date


def apply_scenario_to_sequence(
    sequence_df: pd.DataFrame,
    rainfall_multiplier: float = 1.0,
    spi3_delta: float = 0.0,
    spi6_delta: float = 0.0,
    temperature_delta: float = 0.0,
) -> pd.DataFrame:
    """Return a scenario-adjusted copy of the recent sequence."""
    scenario_df = sequence_df.copy()

    for column in RAINFALL_COLUMNS:
        if column not in scenario_df.columns:
            raise ValueError(f"시나리오 반영에 필요한 컬럼이 없습니다: {column}")
        scenario_df[column] = scenario_df[column] * rainfall_multiplier

    if "SPI3" not in scenario_df.columns or "SPI6" not in scenario_df.columns:
        raise ValueError("시나리오 반영에 필요한 SPI3 또는 SPI6 컬럼이 없습니다.")

    scenario_df["SPI3"] = scenario_df["SPI3"] + spi3_delta
    scenario_df["SPI6"] = scenario_df["SPI6"] + spi6_delta

    for column in TEMPERATURE_COLUMNS:
        if column not in scenario_df.columns:
            raise ValueError(f"시나리오 반영에 필요한 컬럼이 없습니다: {column}")
        scenario_df[column] = scenario_df[column] + temperature_delta

    return scenario_df


def build_model_input(
    sequence_df: pd.DataFrame,
    feature_cols: list[str],
    scaler: object,
) -> torch.Tensor:
    """Build a scaled tensor with shape (1, input_window, num_features)."""
    missing_features = [column for column in feature_cols if column not in sequence_df.columns]
    if missing_features:
        raise ValueError(
            "예측 입력 피처가 누락되었습니다. "
            f"누락 피처: {missing_features}"
        )

    values = sequence_df[feature_cols].to_numpy(dtype=np.float32)
    input_array = values.reshape(1, values.shape[0], values.shape[1])
    scaled_2d = scaler.transform(input_array.reshape(-1, input_array.shape[-1]))
    scaled_input = scaled_2d.reshape(input_array.shape).astype(np.float32)

    return torch.as_tensor(scaled_input, dtype=torch.float32)


def classify_risk(
    predicted_rate: float,
    lower_bound: float,
    caution_threshold: float = 60,
    danger_threshold: float = 40,
) -> dict:
    """Classify predicted reservoir rate risk."""
    if lower_bound < danger_threshold:
        return {
            "risk_code": "danger_possible",
            "risk_label": "위험 가능",
            "risk_message": "예측 하한이 위험 기준 아래로 내려갈 수 있어 선제 대응이 필요합니다.",
        }

    if predicted_rate >= caution_threshold:
        return {
            "risk_code": "stable",
            "risk_label": "안정",
            "risk_message": "예측 저수율이 안정 기준 이상입니다.",
        }

    if predicted_rate >= danger_threshold:
        return {
            "risk_code": "caution",
            "risk_label": "주의",
            "risk_message": "예측 저수율이 주의 구간입니다. 기상 상황을 계속 확인해야 합니다.",
        }

    return {
        "risk_code": "danger",
        "risk_label": "위험",
        "risk_message": "예측 저수율이 위험 기준 아래입니다. 물 확보 방안을 검토해야 합니다.",
    }


def _clip_rate(value: float) -> float:
    return float(np.clip(value, 0.0, 100.0))


def predict_storage_rate(
    df: pd.DataFrame,
    reservoir_name: str = "오봉",
    as_of_date: Optional[str | pd.Timestamp] = None,
    rainfall_multiplier: float = 1.0,
    spi3_delta: float = 0.0,
    spi6_delta: float = 0.0,
    temperature_delta: float = 0.0,
    forecast_horizon: Optional[int] = None,
    model_dir: Optional[Path | str] = None,
) -> dict:
    """Predict future reservoir storage rate with optional scenario changes."""
    model, scaler, config, residual_stats, device = load_trained_model(model_dir=model_dir)
    input_window = int(config["input_window"])
    selected_forecast_horizon = int(
        forecast_horizon if forecast_horizon is not None else config["forecast_horizon"]
    )
    feature_cols = list(config["feature_cols"])

    prediction_df = prepare_prediction_dataframe(df)
    sequence_df, basis_date = make_recent_sequence(
        prediction_df,
        reservoir_name=reservoir_name,
        as_of_date=as_of_date,
        input_window=input_window,
    )
    scenario_df = apply_scenario_to_sequence(
        sequence_df,
        rainfall_multiplier=rainfall_multiplier,
        spi3_delta=spi3_delta,
        spi6_delta=spi6_delta,
        temperature_delta=temperature_delta,
    )
    model_input = build_model_input(scenario_df, feature_cols, scaler).to(device)

    with torch.no_grad():
        predicted_rate = float(model(model_input).cpu().numpy().reshape(-1)[0])

    residual_std = float(residual_stats.get("residual_std", 0.0))
    lower_bound = predicted_rate - 1.96 * residual_std
    upper_bound = predicted_rate + 1.96 * residual_std

    predicted_rate = _clip_rate(predicted_rate)
    lower_bound = _clip_rate(lower_bound)
    upper_bound = _clip_rate(upper_bound)
    target_date = basis_date + pd.Timedelta(days=selected_forecast_horizon)

    risk = classify_risk(predicted_rate, lower_bound)

    return {
        "reservoir_name": reservoir_name,
        "as_of_date": basis_date.date().isoformat(),
        "forecast_horizon": selected_forecast_horizon,
        "target_date": target_date.date().isoformat(),
        "predicted_rate": predicted_rate,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "risk_code": risk["risk_code"],
        "risk_label": risk["risk_label"],
        "risk_message": risk["risk_message"],
        "scenario": {
            "rainfall_multiplier": float(rainfall_multiplier),
            "spi3_delta": float(spi3_delta),
            "spi6_delta": float(spi6_delta),
            "temperature_delta": float(temperature_delta),
        },
        "residual_std": residual_std,
    }


def format_prediction_result(result: dict) -> str:
    """Format prediction output as Korean chatbot text."""
    scenario = result["scenario"]
    return (
        f"기준일: {result['as_of_date']}\n"
        f"예측 대상일: {result['target_date']} "
        f"({result['forecast_horizon']}일 뒤)\n"
        f"예상 저수율: {result['predicted_rate']:.2f}%\n"
        f"예상 범위: {result['lower_bound']:.2f}% ~ {result['upper_bound']:.2f}%\n"
        f"위험 판단: {result['risk_label']}\n"
        f"판단 설명: {result['risk_message']}\n"
        "적용된 시나리오: "
        f"강수량 배율 {scenario['rainfall_multiplier']:.2f}, "
        f"SPI3 변화량 {scenario['spi3_delta']:+.2f}, "
        f"SPI6 변화량 {scenario['spi6_delta']:+.2f}, "
        f"기온 변화량 {scenario['temperature_delta']:+.2f}℃"
    )


if __name__ == "__main__":
    try:
        master_df = load_master_dataset(source="auto")

        default_result = predict_storage_rate(master_df)
        print("[기본 시나리오]")
        print(format_prediction_result(default_result))

        drought_result = predict_storage_rate(
            master_df,
            rainfall_multiplier=0.0,
            spi3_delta=-0.5,
        )
        print("\n[가뭄 시나리오]")
        print("강수량 배율: 0.0")
        print("SPI3 변화량: -0.5")
        print(format_prediction_result(drought_result))
    except Exception as error:
        print("Transformer 예측 테스트에 실패했습니다.")
        print(error)
