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

from src.db.supabase_client import load_master_dataset  # noqa: E402
from src.model.train_transformer_scenario_minrisk import ScenarioMinRiskTransformer  # noqa: E402
from src.preprocessing.prepare_timeseries import add_time_features, clean_master_dataset  # noqa: E402


DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
MODEL_FILENAME = "minrisk_transformer_ohbong.pt"
CONFIG_FILENAME = "minrisk_model_config.json"
X_SCALER_FILENAME = "minrisk_x_scaler.joblib"
SCENARIO_SCALER_FILENAME = "minrisk_scenario_scaler.joblib"
Y_SCALER_FILENAME = "minrisk_y_scaler.joblib"
RESIDUAL_STATS_FILENAME = "minrisk_residual_stats.json"

SCENARIO_COLUMNS = [
    "future_rainfall_sum_30d",
    "future_rainfall_mean_30d",
    "future_rainy_days_30d",
    "future_no_rain_days_30d",
    "future_temp_mean_30d",
    "future_temp_max_30d",
    "future_spi3_delta_30d",
    "future_spi6_delta_30d",
    "future_spi3_min_30d",
    "future_spi6_min_30d",
    "target_month_sin",
    "target_month_cos",
    "target_season_sin",
    "target_season_cos",
    "current_rate",
]


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            "MinRisk 모델 파일이 없습니다. "
            "먼저 python src/model/train_transformer_scenario_minrisk.py를 실행하세요."
        )


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_minrisk_model(model_dir: Optional[Path | str] = None):
    selected_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    model_path = selected_dir / MODEL_FILENAME
    config_path = selected_dir / CONFIG_FILENAME
    x_scaler_path = selected_dir / X_SCALER_FILENAME
    scenario_scaler_path = selected_dir / SCENARIO_SCALER_FILENAME
    y_scaler_path = selected_dir / Y_SCALER_FILENAME
    residual_stats_path = selected_dir / RESIDUAL_STATS_FILENAME

    for path in [model_path, config_path, x_scaler_path, scenario_scaler_path, y_scaler_path, residual_stats_path]:
        _require_file(path)

    config = _load_json(config_path)
    residual_stats = _load_json(residual_stats_path)
    x_scaler = joblib.load(x_scaler_path)
    scenario_scaler = joblib.load(scenario_scaler_path)
    y_scaler = joblib.load(y_scaler_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ScenarioMinRiskTransformer(
        num_seq_features=int(config["num_seq_features"]),
        num_scenario_features=int(config["num_scenario_features"]),
        input_window=int(config["input_window"]),
        d_model=int(config.get("d_model", 64)),
        nhead=int(config.get("nhead", 4)),
        num_layers=int(config.get("num_layers", 2)),
        dim_feedforward=int(config.get("dim_feedforward", 128)),
        dropout=float(config.get("dropout", 0.2)),
        scenario_hidden=int(config.get("scenario_hidden", 32)),
        fusion_hidden=int(config.get("fusion_hidden", 64)),
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, x_scaler, scenario_scaler, y_scaler, config, residual_stats, device


def prepare_prediction_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return add_time_features(clean_master_dataset(df))


def make_recent_sequence(
    df: pd.DataFrame,
    reservoir_name: str = "오봉",
    as_of_date: Optional[str | pd.Timestamp] = None,
    input_window: int = 60,
) -> tuple[pd.DataFrame, pd.Timestamp, float]:
    reservoir_df = df[df["저수지명"] == reservoir_name].copy()
    if reservoir_df.empty:
        raise ValueError(f"저수지명 == '{reservoir_name}'인 데이터가 없습니다.")

    reservoir_df["날짜"] = pd.to_datetime(reservoir_df["날짜"])
    reservoir_df = reservoir_df.sort_values("날짜")
    if as_of_date is not None:
        selected_date = pd.Timestamp(as_of_date)
        reservoir_df = reservoir_df[reservoir_df["날짜"] <= selected_date]
    if len(reservoir_df) < input_window:
        raise ValueError(
            f"{reservoir_name} 저수지 데이터가 {len(reservoir_df)}행뿐이라 "
            f"최근 {input_window}일 sequence를 만들 수 없습니다."
        )

    sequence_df = reservoir_df.tail(input_window).reset_index(drop=True)
    basis_date = pd.Timestamp(sequence_df["날짜"].iloc[-1])
    current_rate = float(sequence_df["저수율_5일EMA"].iloc[-1])
    return sequence_df, basis_date, current_rate


def build_scenario_features_from_user_input(
    sequence_df: pd.DataFrame,
    as_of_date: str | pd.Timestamp,
    forecast_horizon: int = 30,
    rainfall_multiplier: float = 1.0,
    rainfall_sum_30d: Optional[float] = None,
    rainy_days_30d: Optional[int] = None,
    temperature_delta: float = 0.0,
    spi3_delta: float = 0.0,
    spi6_delta: float = 0.0,
) -> dict:
    missing = [col for col in ["저수율_5일EMA", "강수량", "평균기온(℃)", "SPI3", "SPI6"] if col not in sequence_df.columns]
    if missing:
        raise ValueError(f"scenario feature 생성에 필요한 컬럼이 없습니다: {missing}")

    current_rate = float(sequence_df["저수율_5일EMA"].iloc[-1])
    recent_rain = sequence_df["강수량"].tail(forecast_horizon).to_numpy(dtype=np.float64)
    future_rain = np.clip(recent_rain * rainfall_multiplier, 0.0, None)

    if rainfall_sum_30d is not None:
        future_rainfall_sum = float(rainfall_sum_30d)
        future_rainfall_mean = future_rainfall_sum / forecast_horizon
    else:
        future_rainfall_sum = float(future_rain.sum())
        future_rainfall_mean = float(future_rain.mean())

    if rainy_days_30d is not None:
        future_rainy_days = float(np.clip(rainy_days_30d, 0, forecast_horizon))
    else:
        future_rainy_days = float((future_rain > 0).sum())
    future_no_rain_days = float(forecast_horizon - future_rainy_days)

    recent_temp = sequence_df["평균기온(℃)"].tail(forecast_horizon).to_numpy(dtype=np.float64)
    future_temp = recent_temp + temperature_delta
    basis_spi3 = float(sequence_df["SPI3"].iloc[-1])
    basis_spi6 = float(sequence_df["SPI6"].iloc[-1])
    future_spi3_end = basis_spi3 + spi3_delta
    future_spi6_end = basis_spi6 + spi6_delta
    future_spi3_min = min(basis_spi3, future_spi3_end)
    future_spi6_min = min(basis_spi6, future_spi6_end)

    target_date = pd.Timestamp(as_of_date) + pd.Timedelta(days=forecast_horizon)
    scenario = {
        "future_rainfall_sum_30d": future_rainfall_sum,
        "future_rainfall_mean_30d": future_rainfall_mean,
        "future_rainy_days_30d": future_rainy_days,
        "future_no_rain_days_30d": future_no_rain_days,
        "future_temp_mean_30d": float(future_temp.mean()),
        "future_temp_max_30d": float(future_temp.max()),
        "future_spi3_delta_30d": float(spi3_delta),
        "future_spi6_delta_30d": float(spi6_delta),
        "future_spi3_min_30d": float(future_spi3_min),
        "future_spi6_min_30d": float(future_spi6_min),
        "target_month_sin": float(np.sin(2 * np.pi * target_date.month / 12)),
        "target_month_cos": float(np.cos(2 * np.pi * target_date.month / 12)),
        "target_season_sin": float(np.sin(2 * np.pi * target_date.dayofyear / 365.25)),
        "target_season_cos": float(np.cos(2 * np.pi * target_date.dayofyear / 365.25)),
        "current_rate": current_rate,
        "rainfall_multiplier": float(rainfall_multiplier),
        "rainfall_sum_30d_input": rainfall_sum_30d,
        "rainy_days_30d_input": rainy_days_30d,
        "temperature_delta": float(temperature_delta),
        "spi3_delta": float(spi3_delta),
        "spi6_delta": float(spi6_delta),
    }

    missing_scenario = [column for column in SCENARIO_COLUMNS if column not in scenario]
    if missing_scenario:
        raise ValueError(f"scenario feature가 누락되었습니다: {missing_scenario}")
    return scenario


def build_model_inputs(
    sequence_df: pd.DataFrame,
    scenario_features: dict,
    feature_cols: list[str],
    scenario_cols: list[str],
    x_scaler,
    scenario_scaler,
) -> tuple[torch.Tensor, torch.Tensor]:
    missing_features = [column for column in feature_cols if column not in sequence_df.columns]
    if missing_features:
        raise ValueError(f"sequence 입력 피처가 누락되었습니다: {missing_features}")
    missing_scenario = [column for column in scenario_cols if column not in scenario_features]
    if missing_scenario:
        raise ValueError(f"scenario 입력 피처가 누락되었습니다: {missing_scenario}")

    X_seq = sequence_df[feature_cols].to_numpy(dtype=np.float32).reshape(1, len(sequence_df), len(feature_cols))
    X_seq_scaled = x_scaler.transform(X_seq.reshape(-1, X_seq.shape[-1])).reshape(X_seq.shape)

    X_scenario = np.asarray([[scenario_features[column] for column in scenario_cols]], dtype=np.float32)
    X_scenario_scaled = scenario_scaler.transform(X_scenario)

    return (
        torch.as_tensor(X_seq_scaled, dtype=torch.float32),
        torch.as_tensor(X_scenario_scaled, dtype=torch.float32),
    )


def classify_minrisk(
    predicted_min_rate: float,
    danger_threshold: float = 40,
    warning_threshold: float = 60,
) -> dict:
    if predicted_min_rate < danger_threshold:
        return {
            "risk_code": "danger",
            "risk_label": "위험",
            "risk_message": "향후 30일 안에 위험 저수율 아래로 내려갈 가능성이 큽니다.",
        }
    if predicted_min_rate < warning_threshold:
        return {
            "risk_code": "caution",
            "risk_label": "주의",
            "risk_message": "향후 30일 최저 저수율이 주의 구간으로 예상됩니다.",
        }
    return {
        "risk_code": "stable",
        "risk_label": "안정",
        "risk_message": "향후 30일 최저 저수율이 안정 기준 이상으로 예상됩니다.",
    }


def _clip_rate(value: float) -> float:
    return float(np.clip(value, 0.0, 100.0))


def predict_minrisk(
    df: pd.DataFrame,
    reservoir_name: str = "오봉",
    as_of_date: Optional[str | pd.Timestamp] = None,
    forecast_horizon: int = 30,
    rainfall_multiplier: float = 1.0,
    rainfall_sum_30d: Optional[float] = None,
    rainy_days_30d: Optional[int] = None,
    temperature_delta: float = 0.0,
    spi3_delta: float = 0.0,
    spi6_delta: float = 0.0,
    danger_threshold: float = 40,
    warning_threshold: float = 60,
    model_dir: Optional[Path | str] = None,
) -> dict:
    model, x_scaler, scenario_scaler, y_scaler, config, residual_stats, device = load_minrisk_model(model_dir)
    input_window = int(config["input_window"])
    selected_horizon = int(forecast_horizon or config.get("forecast_horizon", 30))
    feature_cols = list(config["feature_cols"])
    scenario_cols = list(config.get("scenario_cols", SCENARIO_COLUMNS))

    prepared_df = prepare_prediction_dataframe(df)
    sequence_df, basis_date, current_rate = make_recent_sequence(
        prepared_df,
        reservoir_name=reservoir_name,
        as_of_date=as_of_date,
        input_window=input_window,
    )
    scenario_features = build_scenario_features_from_user_input(
        sequence_df,
        as_of_date=basis_date,
        forecast_horizon=selected_horizon,
        rainfall_multiplier=rainfall_multiplier,
        rainfall_sum_30d=rainfall_sum_30d,
        rainy_days_30d=rainy_days_30d,
        temperature_delta=temperature_delta,
        spi3_delta=spi3_delta,
        spi6_delta=spi6_delta,
    )
    X_seq, X_scenario = build_model_inputs(
        sequence_df,
        scenario_features,
        feature_cols,
        scenario_cols,
        x_scaler,
        scenario_scaler,
    )

    with torch.no_grad():
        delta_scaled, risk_logit = model(X_seq.to(device), X_scenario.to(device))
    predicted_min_delta = float(
        y_scaler.inverse_transform(delta_scaled.cpu().numpy().reshape(-1, 1)).reshape(-1)[0]
    )
    predicted_min_rate = _clip_rate(current_rate + predicted_min_delta)
    risk_probability = float(torch.sigmoid(risk_logit).cpu().numpy().reshape(-1)[0])

    residual_std = float(residual_stats.get("residual_std", 0.0))
    lower_bound = _clip_rate(predicted_min_rate - 1.96 * residual_std)
    upper_bound = _clip_rate(predicted_min_rate + 1.96 * residual_std)
    risk = classify_minrisk(predicted_min_rate, danger_threshold, warning_threshold)

    forecast_start_date = basis_date + pd.Timedelta(days=1)
    forecast_end_date = basis_date + pd.Timedelta(days=selected_horizon)

    return {
        "reservoir_name": reservoir_name,
        "as_of_date": basis_date.date().isoformat(),
        "forecast_horizon": selected_horizon,
        "forecast_start_date": forecast_start_date.date().isoformat(),
        "forecast_end_date": forecast_end_date.date().isoformat(),
        "current_rate": _clip_rate(current_rate),
        "predicted_min_rate": predicted_min_rate,
        "predicted_min_delta": predicted_min_delta,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "risk_code": risk["risk_code"],
        "risk_label": risk["risk_label"],
        "risk_message": risk["risk_message"],
        "risk_probability_reference": risk_probability,
        "scenario": {
            "rainfall_multiplier": rainfall_multiplier,
            "rainfall_sum_30d": rainfall_sum_30d,
            "rainy_days_30d": rainy_days_30d,
            "temperature_delta": temperature_delta,
            "spi3_delta": spi3_delta,
            "spi6_delta": spi6_delta,
            "scenario_features": {column: scenario_features[column] for column in scenario_cols},
        },
        "residual_std": residual_std,
        "model_type": config.get("model_type", "scenario_minrisk_transformer"),
    }


def format_minrisk_result(result: dict) -> str:
    scenario = result["scenario"]
    return (
        f"기준일: {result['as_of_date']}\n"
        f"예측 기간: {result['forecast_start_date']} ~ {result['forecast_end_date']}\n"
        f"현재 저수율: {result['current_rate']:.2f}%\n"
        f"향후 {result['forecast_horizon']}일 예상 최저 저수율: {result['predicted_min_rate']:.2f}%\n"
        f"예상 범위: {result['lower_bound']:.2f}% ~ {result['upper_bound']:.2f}%\n"
        f"위험 판단: {result['risk_label']}\n"
        f"판단 설명: {result['risk_message']}\n"
        "적용 시나리오: "
        f"강수량 배율 {scenario['rainfall_multiplier']:.2f}, "
        f"30일 총강수량 직접입력 {scenario['rainfall_sum_30d']}, "
        f"비 오는 날 직접입력 {scenario['rainy_days_30d']}, "
        f"SPI3 변화량 {scenario['spi3_delta']:+.2f}, "
        f"SPI6 변화량 {scenario['spi6_delta']:+.2f}, "
        f"기온 변화량 {scenario['temperature_delta']:+.2f}℃\n"
        f"classification head risk_probability 참고값: {result['risk_probability_reference']:.3f} "
        "(최종 위험 판단에는 predicted_min_rate threshold를 사용했습니다.)"
    )


if __name__ == "__main__":
    try:
        master_df = load_master_dataset(source="auto")

        print("[기본 시나리오]")
        default_result = predict_minrisk(master_df)
        print(format_minrisk_result(default_result))

        print("\n[무강수 + SPI 악화 시나리오]")
        drought_result = predict_minrisk(
            master_df,
            rainfall_multiplier=0.0,
            spi3_delta=-0.5,
            spi6_delta=-0.3,
        )
        print(format_minrisk_result(drought_result))

        print("\n[보수적 가뭄 시나리오]")
        severe_drought_result = predict_minrisk(
            master_df,
            rainfall_multiplier=0.0,
            spi3_delta=-1.0,
            spi6_delta=-0.7,
            temperature_delta=2.0,
        )
        print(format_minrisk_result(severe_drought_result))
    except Exception as error:
        print("MinRisk 예측 테스트에 실패했습니다.")
        print(error)
