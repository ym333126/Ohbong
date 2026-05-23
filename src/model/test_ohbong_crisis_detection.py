import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.db.supabase_client import load_master_dataset  # noqa: E402
from src.model.predict_minrisk import format_minrisk_result, predict_minrisk  # noqa: E402


OUTPUT_PATH = PROJECT_ROOT / "models" / "ohbong_crisis_detection_test.csv"
DANGER_THRESHOLD = 40

TEST_DATES = [
    "2025-06-30",
    "2025-07-05",
    "2025-07-10",
    "2025-07-15",
    "2025-07-20",
    "2025-07-25",
    "2025-08-01",
]

SCENARIOS = [
    {
        "scenario_name": "기본 시나리오",
        "rainfall_multiplier": 1.0,
        "spi3_delta": 0.0,
        "spi6_delta": 0.0,
        "temperature_delta": 0.0,
    },
    {
        "scenario_name": "무강수 시나리오",
        "rainfall_multiplier": 0.0,
        "spi3_delta": -0.5,
        "spi6_delta": -0.3,
        "temperature_delta": 0.0,
    },
    {
        "scenario_name": "보수적 가뭄 시나리오",
        "rainfall_multiplier": 0.0,
        "spi3_delta": -1.0,
        "spi6_delta": -0.7,
        "temperature_delta": 2.0,
    },
]


def calculate_actual_minimum(
    df: pd.DataFrame,
    as_of_date: str,
    reservoir_name: str = "오봉",
    forecast_horizon: int = 30,
) -> dict:
    ohbong_df = df[df["저수지명"] == reservoir_name].copy()
    if ohbong_df.empty:
        raise ValueError(f"저수지명 == '{reservoir_name}'인 데이터가 없습니다.")

    ohbong_df["날짜"] = pd.to_datetime(ohbong_df["날짜"])
    ohbong_df["저수율_5일EMA"] = pd.to_numeric(ohbong_df["저수율_5일EMA"], errors="coerce")
    start_date = pd.Timestamp(as_of_date) + pd.Timedelta(days=1)
    end_date = pd.Timestamp(as_of_date) + pd.Timedelta(days=forecast_horizon)

    future_df = ohbong_df[
        (ohbong_df["날짜"] >= start_date)
        & (ohbong_df["날짜"] <= end_date)
    ].dropna(subset=["저수율_5일EMA"])

    if future_df.empty:
        raise ValueError(f"{as_of_date} 이후 30일 실제 오봉 데이터가 없습니다.")

    min_idx = future_df["저수율_5일EMA"].idxmin()
    actual_min_rate = float(future_df.loc[min_idx, "저수율_5일EMA"])
    actual_min_date = pd.Timestamp(future_df.loc[min_idx, "날짜"]).date().isoformat()

    return {
        "actual_min_rate_30d": actual_min_rate,
        "actual_min_date": actual_min_date,
        "actual_risk_within_30d": bool(actual_min_rate < DANGER_THRESHOLD),
    }


def _confusion_flags(predicted_risk: bool, actual_risk: bool) -> dict:
    return {
        "TP": bool(predicted_risk and actual_risk),
        "FP": bool(predicted_risk and not actual_risk),
        "TN": bool(not predicted_risk and not actual_risk),
        "FN": bool(not predicted_risk and actual_risk),
    }


def run_crisis_detection_test() -> pd.DataFrame:
    df = load_master_dataset(source="auto")
    rows = []

    for as_of_date in TEST_DATES:
        actual = calculate_actual_minimum(df, as_of_date)
        for scenario in SCENARIOS:
            result = predict_minrisk(
                df,
                reservoir_name="오봉",
                as_of_date=as_of_date,
                rainfall_multiplier=scenario["rainfall_multiplier"],
                spi3_delta=scenario["spi3_delta"],
                spi6_delta=scenario["spi6_delta"],
                temperature_delta=scenario["temperature_delta"],
                danger_threshold=DANGER_THRESHOLD,
            )
            predicted_risk = result["predicted_min_rate"] < DANGER_THRESHOLD
            actual_risk = actual["actual_risk_within_30d"]
            error = actual["actual_min_rate_30d"] - result["predicted_min_rate"]
            flags = _confusion_flags(predicted_risk, actual_risk)

            rows.append(
                {
                    "기준일": result["as_of_date"],
                    "예측시작일": result["forecast_start_date"],
                    "예측종료일": result["forecast_end_date"],
                    "시나리오명": scenario["scenario_name"],
                    "current_rate": result["current_rate"],
                    "predicted_min_rate": result["predicted_min_rate"],
                    "lower_bound": result["lower_bound"],
                    "upper_bound": result["upper_bound"],
                    "risk_label": result["risk_label"],
                    "risk_code": result["risk_code"],
                    "risk_probability_reference": result["risk_probability_reference"],
                    "actual_min_rate_30d": actual["actual_min_rate_30d"],
                    "actual_min_date": actual["actual_min_date"],
                    "actual_risk_within_30d": actual_risk,
                    "prediction_error": error,
                    "abs_error": abs(error),
                    "predicted_risk_by_threshold": predicted_risk,
                    **flags,
                }
            )

    result_df = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    return result_df


def _find_earliest_detection(result_df: pd.DataFrame) -> str:
    detected = result_df[
        (result_df["actual_risk_within_30d"])
        & (result_df["predicted_risk_by_threshold"])
    ].copy()
    if detected.empty:
        return "감지하지 못함"
    detected["기준일_dt"] = pd.to_datetime(detected["기준일"])
    first_row = detected.sort_values(["기준일_dt", "시나리오명"]).iloc[0]
    return f"{first_row['기준일']} ({first_row['시나리오명']})"


def _print_results(result_df: pd.DataFrame) -> None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 180)

    columns = [
        "기준일",
        "예측시작일",
        "예측종료일",
        "시나리오명",
        "current_rate",
        "predicted_min_rate",
        "lower_bound",
        "upper_bound",
        "risk_label",
        "risk_code",
        "risk_probability_reference",
        "actual_min_rate_30d",
        "actual_min_date",
        "actual_risk_within_30d",
        "prediction_error",
        "abs_error",
        "TP",
        "FP",
        "TN",
        "FN",
    ]

    print("=== 전체 결과 테이블 ===")
    print(result_df[columns].to_string(index=False))

    print("\n=== 기준일별 기본 시나리오 결과 ===")
    print(
        result_df[result_df["시나리오명"] == "기본 시나리오"][columns]
        .to_string(index=False)
    )

    print("\n=== 기준일별 무강수 시나리오 결과 ===")
    print(
        result_df[result_df["시나리오명"] == "무강수 시나리오"][columns]
        .to_string(index=False)
    )

    fn_count = int(result_df["FN"].sum())
    earliest_detection = _find_earliest_detection(result_df)

    print(f"\n실제 위험을 놓친 FN 개수: {fn_count}")
    print(f"2025년 8월 사태를 사전에 감지한 가장 이른 기준일: {earliest_detection}")
    print(f"저장 경로: {OUTPUT_PATH}")
    print("\n참고: classification head risk_probability는 참고값이며, 최종 위험 판단은 predicted_min_rate < 40 기준입니다.")


if __name__ == "__main__":
    try:
        crisis_results = run_crisis_detection_test()
        _print_results(crisis_results)
    except Exception as error:
        print("오봉 crisis detection 테스트에 실패했습니다.")
        print(error)
