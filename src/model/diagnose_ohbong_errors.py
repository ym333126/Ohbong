import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MODELS_DIR = PROJECT_ROOT / "models"
ENSEMBLE_TEST_PATH = MODELS_DIR / "ohbong_ensemble_predictions_test.csv"
MONTHLY_PATH = MODELS_DIR / "ohbong_error_monthly.csv"
LEVEL_PATH = MODELS_DIR / "ohbong_error_by_level.csv"
TOP20_PATH = MODELS_DIR / "ohbong_error_top20.csv"
SUMMARY_PATH = MODELS_DIR / "ohbong_error_diagnosis_summary.json"
PLOT_PATH = MODELS_DIR / "ohbong_2025_predictions.png"


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"필수 진단 파일이 없습니다: {path}. "
            "먼저 `python src/model/tune_ohbong_ensemble.py`를 실행하세요."
        )


def _find_first_existing_column(df: pd.DataFrame, candidates: list[str], role: str) -> str:
    for column in candidates:
        if column in df.columns:
            return column
    raise ValueError(f"{role} 컬럼을 찾을 수 없습니다. 후보: {candidates}")


def load_ohbong_predictions() -> pd.DataFrame:
    """Load ensemble test predictions and normalize flexible column names."""
    _require_file(ENSEMBLE_TEST_PATH)
    df = pd.read_csv(ENSEMBLE_TEST_PATH)

    required_columns = [
        "저수지명",
        "기준날짜",
        "예측대상날짜",
        "current_rate",
        "delta_prediction",
        "persistence_prediction",
    ]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"진단에 필요한 컬럼이 없습니다: {missing}")

    y_true_col = _find_first_existing_column(df, ["y_true", "future_rate"], "실제값")
    ensemble_col = _find_first_existing_column(
        df,
        ["final_prediction", "ensemble_prediction", "ohbong_ensemble_prediction"],
        "앙상블 예측값",
    )

    ohbong_df = df[df["저수지명"] == "오봉"].copy()
    if ohbong_df.empty:
        raise ValueError("오봉 test 예측 데이터가 없습니다.")

    ohbong_df["예측대상날짜"] = pd.to_datetime(ohbong_df["예측대상날짜"])
    ohbong_df["기준날짜"] = pd.to_datetime(ohbong_df["기준날짜"])
    ohbong_df["y_true"] = pd.to_numeric(ohbong_df[y_true_col], errors="coerce")
    ohbong_df["ensemble_prediction"] = pd.to_numeric(
        ohbong_df[ensemble_col],
        errors="coerce",
    )

    numeric_columns = [
        "current_rate",
        "delta_prediction",
        "persistence_prediction",
        "y_true",
        "ensemble_prediction",
    ]
    for column in numeric_columns:
        ohbong_df[column] = pd.to_numeric(ohbong_df[column], errors="coerce")

    ohbong_df = ohbong_df.dropna(subset=numeric_columns).sort_values("예측대상날짜")
    if ohbong_df.empty:
        raise ValueError("숫자 변환 후 남은 오봉 예측 데이터가 없습니다.")

    return ohbong_df.reset_index(drop=True)


def add_error_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["persistence_error"] = result["y_true"] - result["persistence_prediction"]
    result["delta_error"] = result["y_true"] - result["delta_prediction"]
    result["ensemble_error"] = result["y_true"] - result["ensemble_prediction"]
    result["persistence_abs_error"] = result["persistence_error"].abs()
    result["delta_abs_error"] = result["delta_error"].abs()
    result["ensemble_abs_error"] = result["ensemble_error"].abs()
    result["actual_delta"] = result["y_true"] - result["current_rate"]
    result["month"] = result["예측대상날짜"].dt.to_period("M").astype(str)
    return result


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    residual = y_true - y_pred
    safe_denominator = np.where(np.abs(y_true) < 1e-6, np.nan, np.abs(y_true))
    mape_values = np.abs(residual) / safe_denominator

    return {
        "MAE": float(np.mean(np.abs(residual))),
        "RMSE": float(np.sqrt(np.mean(residual**2))),
        "MAPE": float(np.nanmean(mape_values) * 100)
        if not np.all(np.isnan(mape_values))
        else 0.0,
    }


def _metric_row(group: pd.DataFrame, model_name: str, prediction_col: str) -> dict:
    metrics = calculate_metrics(group["y_true"].to_numpy(), group[prediction_col].to_numpy())
    return {
        "model": model_name,
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "MAPE": metrics["MAPE"],
        "n_samples": int(len(group)),
        "y_true_mean": float(group["y_true"].mean()),
        "y_true_min": float(group["y_true"].min()),
        "y_true_max": float(group["y_true"].max()),
    }


def calculate_monthly_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    model_columns = [
        ("Persistence", "persistence_prediction"),
        ("Delta Transformer", "delta_prediction"),
        ("Ohbong Ensemble", "ensemble_prediction"),
    ]
    for month, group in df.groupby("month"):
        for model_name, prediction_col in model_columns:
            row = _metric_row(group, model_name, prediction_col)
            row["month"] = month
            rows.append(row)
    return pd.DataFrame(rows)[
        ["month", "model", "MAE", "RMSE", "MAPE", "n_samples", "y_true_mean", "y_true_min", "y_true_max"]
    ]


def calculate_level_metrics(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    bins = [0, 20, 40, 60, 80, 100]
    labels = ["0~20", "20~40", "40~60", "60~80", "80~100"]
    result["level_bin"] = pd.cut(
        result["y_true"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )

    rows = []
    model_columns = [
        ("Persistence", "persistence_prediction"),
        ("Delta Transformer", "delta_prediction"),
        ("Ohbong Ensemble", "ensemble_prediction"),
    ]
    for level_bin, group in result.dropna(subset=["level_bin"]).groupby("level_bin", observed=False):
        for model_name, prediction_col in model_columns:
            row = _metric_row(group, model_name, prediction_col)
            row["level_bin"] = str(level_bin)
            rows.append(row)

    return pd.DataFrame(rows)[
        ["level_bin", "model", "MAE", "RMSE", "MAPE", "n_samples", "y_true_mean", "y_true_min", "y_true_max"]
    ]


def calculate_top_errors(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    columns = [
        "예측대상날짜",
        "current_rate",
        "y_true",
        "persistence_prediction",
        "delta_prediction",
        "ensemble_prediction",
        "persistence_abs_error",
        "delta_abs_error",
        "ensemble_abs_error",
    ]
    top_errors = df.sort_values("ensemble_abs_error", ascending=False).head(n).copy()
    top_errors["예측대상날짜"] = top_errors["예측대상날짜"].dt.date.astype(str)
    return top_errors[columns]


def summarize_diagnosis(
    df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    level_df: pd.DataFrame,
) -> dict:
    persistence_better_count = int(
        (df["persistence_abs_error"] < df["ensemble_abs_error"]).sum()
    )
    ensemble_better_count = int(
        (df["ensemble_abs_error"] < df["persistence_abs_error"]).sum()
    )
    total_count = int(len(df))

    ensemble_monthly = monthly_df[monthly_df["model"] == "Ohbong Ensemble"]
    worst_month_row = ensemble_monthly.sort_values("MAE", ascending=False).iloc[0]

    ensemble_level = level_df[level_df["model"] == "Ohbong Ensemble"]
    worst_level_row = ensemble_level.sort_values("MAE", ascending=False).iloc[0]

    actual_delta = df["actual_delta"]
    summary = {
        "n_samples": total_count,
        "worst_month": str(worst_month_row["month"]),
        "worst_month_mae": float(worst_month_row["MAE"]),
        "worst_level_bin": str(worst_level_row["level_bin"]),
        "worst_level_mae": float(worst_level_row["MAE"]),
        "persistence_better_count": persistence_better_count,
        "persistence_better_ratio": persistence_better_count / total_count,
        "ensemble_better_count": ensemble_better_count,
        "ensemble_better_ratio": ensemble_better_count / total_count,
        "actual_delta_mean": float(actual_delta.mean()),
        "actual_delta_std": float(actual_delta.std(ddof=1)),
        "actual_delta_min": float(actual_delta.min()),
        "actual_delta_max": float(actual_delta.max()),
        "actual_delta_drop_20_count": int((actual_delta <= -20).sum()),
        "actual_delta_rise_20_count": int((actual_delta >= 20).sum()),
    }
    return summary


def save_plot(df: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(14, 6))
        plt.plot(df["예측대상날짜"], df["y_true"], label="y_true", linewidth=2)
        plt.plot(
            df["예측대상날짜"],
            df["persistence_prediction"],
            label="persistence_prediction",
            alpha=0.8,
        )
        plt.plot(
            df["예측대상날짜"],
            df["delta_prediction"],
            label="delta_prediction",
            alpha=0.8,
        )
        plt.plot(
            df["예측대상날짜"],
            df["ensemble_prediction"],
            label="ensemble_prediction",
            alpha=0.8,
        )
        plt.xlabel("Prediction target date")
        plt.ylabel("Storage rate (%)")
        plt.title("Ohbong 2025 Predictions")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOT_PATH, dpi=150)
        plt.close()
    except Exception as error:
        print(f"그래프 저장에 실패했지만 진단은 계속 진행합니다: {error}")


def diagnose_ohbong_errors() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    df = add_error_columns(load_ohbong_predictions())
    monthly_df = calculate_monthly_metrics(df)
    level_df = calculate_level_metrics(df)
    top20_df = calculate_top_errors(df)
    summary = summarize_diagnosis(df, monthly_df, level_df)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    monthly_df.to_csv(MONTHLY_PATH, index=False, encoding="utf-8-sig")
    level_df.to_csv(LEVEL_PATH, index=False, encoding="utf-8-sig")
    top20_df.to_csv(TOP20_PATH, index=False, encoding="utf-8-sig")
    with SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    save_plot(df)
    return monthly_df, level_df, top20_df, summary


def _print_results(
    monthly_df: pd.DataFrame,
    level_df: pd.DataFrame,
    top20_df: pd.DataFrame,
    summary: dict,
) -> None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)

    print("=== 월별 성능 ===")
    print(monthly_df.to_string(index=False))

    print("\n=== 저수율 구간별 성능 ===")
    print(level_df.to_string(index=False))

    print("\n=== Ensemble 기준 큰 오차 상위 20개 날짜 ===")
    print(top20_df.to_string(index=False))

    print("\n=== 실제 30일 변화량(actual_delta) 요약 ===")
    print(f"평균: {summary['actual_delta_mean']:.4f}")
    print(f"표준편차: {summary['actual_delta_std']:.4f}")
    print(f"최소: {summary['actual_delta_min']:.4f}")
    print(f"최대: {summary['actual_delta_max']:.4f}")
    print(f"actual_delta <= -20 급락 구간 개수: {summary['actual_delta_drop_20_count']}")
    print(f"actual_delta >= +20 급등 구간 개수: {summary['actual_delta_rise_20_count']}")

    print(f"\n저장 경로: {MONTHLY_PATH}")
    print(f"저장 경로: {LEVEL_PATH}")
    print(f"저장 경로: {TOP20_PATH}")
    print(f"저장 경로: {SUMMARY_PATH}")
    if PLOT_PATH.exists():
        print(f"저장 경로: {PLOT_PATH}")

    print("\n=== 핵심 진단 요약 ===")
    print(f"가장 오차가 큰 월: {summary['worst_month']} (MAE {summary['worst_month_mae']:.4f})")
    print(
        f"가장 오차가 큰 저수율 구간: {summary['worst_level_bin']} "
        f"(MAE {summary['worst_level_mae']:.4f})"
    )
    print(
        "Persistence가 더 좋은 날짜 비율: "
        f"{summary['persistence_better_ratio']:.2%} "
        f"({summary['persistence_better_count']}/{summary['n_samples']})"
    )
    print(
        "Ensemble이 더 좋은 날짜 비율: "
        f"{summary['ensemble_better_ratio']:.2%} "
        f"({summary['ensemble_better_count']}/{summary['n_samples']})"
    )
    print(
        "다음 개선 방향: 오차가 큰 월과 저수율 구간을 중심으로 오봉 전용 validation split을 재설계하고, "
        "급락/급등 구간을 설명할 수 있는 강수 누적, SPI 변화율, 계절별 feature를 추가해 fine-tuning을 검토하세요."
    )


if __name__ == "__main__":
    try:
        monthly_metrics, level_metrics, top_errors, diagnosis_summary = diagnose_ohbong_errors()
        _print_results(monthly_metrics, level_metrics, top_errors, diagnosis_summary)
    except Exception as error:
        print("오봉 예측 오차 진단에 실패했습니다.")
        print(error)
