import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


PREDICT_PATH = PROJECT_ROOT / "src" / "model" / "predict_minrisk.py"
CRISIS_TEST_PATH = PROJECT_ROOT / "src" / "model" / "test_ohbong_crisis_detection.py"
CONFIG_PATH = PROJECT_ROOT / "models" / "minrisk_model_config.json"

EVALUATION_COLUMNS = [
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

TARGET_LEAKAGE_COLUMNS = [
    "future_min_rate",
    "future_end_rate",
    "actual_delta",
    "risk_within_30d",
    "actual_min_rate_30d",
    "actual_min_date",
    "actual_risk_within_30d",
]


def _read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"점검 대상 파일이 없습니다: {path}")
    return path.read_text(encoding="utf-8")


def _load_model_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def _contains_all(text: str, fragments: list[str]) -> bool:
    return all(fragment in text for fragment in fragments)


def inspect_sources() -> dict:
    predict_source = _read_text(PREDICT_PATH)
    crisis_source = _read_text(CRISIS_TEST_PATH)
    config = _load_model_config()

    checks = {
        "sequence_filters_as_of_date": _contains_all(
            predict_source,
            [
                "reservoir_df[\"날짜\"] <= selected_date",
                "reservoir_df.tail(input_window)",
            ],
        ),
        "scenario_uses_recent_sequence": _contains_all(
            predict_source,
            [
                "recent_rain = sequence_df[\"강수량\"].tail(forecast_horizon)",
                "future_rain = np.clip(recent_rain * rainfall_multiplier",
                "recent_temp = sequence_df[\"평균기온(℃)\"].tail(forecast_horizon)",
            ],
        ),
        "scenario_uses_user_inputs": _contains_all(
            predict_source,
            [
                "rainfall_sum_30d is not None",
                "rainy_days_30d is not None",
                "spi3_delta",
                "spi6_delta",
                "temperature_delta",
            ],
        ),
        "actual_values_only_in_test_file": (
            "actual_min_rate_30d" not in predict_source
            and "actual_risk_within_30d" not in predict_source
            and "calculate_actual_minimum" in crisis_source
        ),
        "prediction_called_without_actuals": _contains_all(
            crisis_source,
            [
                "result = predict_minrisk(",
                "actual = calculate_actual_minimum",
                "actual[\"actual_min_rate_30d\"] - result[\"predicted_min_rate\"]",
            ],
        ),
        "threshold_risk_used": _contains_all(
            predict_source,
            [
                "risk = classify_minrisk(predicted_min_rate",
                "risk_probability_reference",
            ],
        ),
        "preprocess_full_dataframe_before_date_filter": _contains_all(
            predict_source,
            [
                "prepared_df = prepare_prediction_dataframe(df)",
                "make_recent_sequence(",
            ],
        ),
        "cleaning_uses_backward_fill": _source_file_contains(
            PROJECT_ROOT / "src" / "preprocessing" / "prepare_timeseries.py",
            ".ffill().bfill()",
        ),
    }

    return {
        "checks": checks,
        "config": config,
    }


def _source_file_contains(path: Path, fragment: str) -> bool:
    if not path.exists():
        return False
    return fragment in path.read_text(encoding="utf-8")


def determine_leakage_risk(checks: dict) -> tuple[str, list[str], list[str]]:
    issues = []
    suggestions = []

    if not checks["sequence_filters_as_of_date"]:
        issues.append("예측 sequence가 as_of_date 이하로 제한되는지 정적 점검에서 확인하지 못했습니다.")
        suggestions.append("make_recent_sequence()에서 날짜 필터를 먼저 적용한 뒤 tail(input_window)를 호출하세요.")

    if not checks["scenario_uses_recent_sequence"]:
        issues.append("scenario feature가 최근 sequence 또는 사용자 입력만 사용하는지 확인이 필요합니다.")
        suggestions.append("미래 관측값을 직접 읽는 로직이 없는지 build_scenario_features_from_user_input()을 유지 점검하세요.")

    if not checks["actual_values_only_in_test_file"]:
        issues.append("평가용 actual 컬럼이 예측 모듈에 섞였을 가능성이 있습니다.")
        suggestions.append("actual_min_rate_30d 계열 컬럼은 test_ohbong_crisis_detection.py에서만 계산하세요.")

    if checks["preprocess_full_dataframe_before_date_filter"] and checks["cleaning_uses_backward_fill"]:
        issues.append(
            "잠재적 누수: predict_minrisk()가 전체 df를 clean_master_dataset()으로 먼저 정리한 뒤 as_of_date를 필터링합니다. "
            "clean_master_dataset()은 저수지별 ffill 후 bfill을 하므로, 결측치가 있으면 as_of_date 이후 값이 과거 결측치에 들어갈 수 있습니다."
        )
        suggestions.append(
            "retrospective 예측에서는 as_of_date 이하로 먼저 자른 뒤 clean/add_time_features를 적용하거나, "
            "예측용 cleaning에서는 bfill을 끄고 ffill/중앙값만 사용하세요."
        )

    if any("잠재적 누수" in issue for issue in issues):
        risk = "MEDIUM"
    elif issues:
        risk = "HIGH"
    else:
        risk = "LOW"

    return risk, issues, suggestions


def _print_section(title: str) -> None:
    print(f"\n=== {title} ===")


def print_report() -> None:
    inspection = inspect_sources()
    checks = inspection["checks"]
    config = inspection["config"]
    leakage_risk, issues, suggestions = determine_leakage_risk(checks)

    feature_cols = config.get("feature_cols", [])
    scenario_cols = config.get("scenario_cols", [])
    prediction_input_columns = {
        "sequence_feature_cols": feature_cols,
        "scenario_cols": scenario_cols,
    }

    print(f"leakage_risk: {leakage_risk}")

    _print_section("함수별 날짜 범위 사용")
    print(
        "predict_minrisk.prepare_prediction_dataframe(df): 전체 df에 clean_master_dataset()와 add_time_features()를 먼저 적용합니다."
    )
    print(
        "predict_minrisk.make_recent_sequence(df, as_of_date): 특정 저수지를 선택한 뒤, as_of_date가 있으면 날짜 <= as_of_date만 남기고 마지막 input_window일을 사용합니다."
    )
    print(
        "predict_minrisk.build_scenario_features_from_user_input(): sequence_df의 최근 강수/기온/SPI 패턴과 사용자 입력값만 사용해 미래 30일 scenario feature를 만듭니다."
    )
    print(
        "test_ohbong_crisis_detection.calculate_actual_minimum(): as_of_date 다음날부터 30일 실제값을 계산하지만, predict_minrisk() 호출 인자로 전달하지 않고 평가용 컬럼에만 사용합니다."
    )

    _print_section("정적 점검 결과")
    for name, passed in checks.items():
        print(f"- {name}: {'PASS' if passed else 'CHECK'}")

    _print_section("예측 입력 컬럼 목록")
    print("sequence feature columns:")
    for column in prediction_input_columns["sequence_feature_cols"]:
        print(f"- {column}")
    print("\nscenario feature columns:")
    for column in prediction_input_columns["scenario_cols"]:
        print(f"- {column}")

    _print_section("평가용 실제값 컬럼 목록")
    for column in EVALUATION_COLUMNS:
        print(f"- {column}")

    _print_section("정답 정보 입력 여부 점검")
    leaked_sequence = [column for column in TARGET_LEAKAGE_COLUMNS if column in feature_cols]
    leaked_scenario = [column for column in TARGET_LEAKAGE_COLUMNS if column in scenario_cols]
    if leaked_sequence or leaked_scenario:
        print("모델 입력 컬럼에 정답/평가용 정보로 보이는 컬럼이 포함되어 있습니다.")
        print(f"sequence 쪽 의심 컬럼: {leaked_sequence}")
        print(f"scenario 쪽 의심 컬럼: {leaked_scenario}")
    else:
        print("feature_cols/scenario_cols에는 future_min_rate, future_end_rate, actual 계열 정답 컬럼이 없습니다.")

    _print_section("문제 및 수정 제안")
    if not issues:
        print("명시적인 데이터 누수 문제는 발견하지 못했습니다.")
    else:
        for index, issue in enumerate(issues, start=1):
            print(f"{index}. {issue}")

    if suggestions:
        print("\n수정 제안:")
        for index, suggestion in enumerate(suggestions, start=1):
            print(f"{index}. {suggestion}")
    else:
        print("추가 수정 제안 없음.")

    _print_section("종합 판단")
    if leakage_risk == "LOW":
        print("현재 구조는 예측 입력과 평가용 실제값이 분리되어 있어 누수 위험이 낮습니다.")
    elif leakage_risk == "MEDIUM":
        print(
            "예측 입력과 평가용 실제값은 분리되어 있으나, 전체 데이터 전처리 후 날짜를 자르는 순서 때문에 "
            "결측치가 있는 경우 미래값 bfill 누수가 발생할 수 있습니다."
        )
    else:
        print("예측 입력에 미래 관측값 또는 평가용 정답 컬럼이 들어갈 가능성이 있어 즉시 수정이 필요합니다.")


if __name__ == "__main__":
    try:
        print_report()
    except Exception as error:
        print("MinRisk leakage 점검에 실패했습니다.")
        print(error)
