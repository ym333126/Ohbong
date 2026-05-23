import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.prepare_timeseries import prepare_train_test_data  # noqa: E402


TARGET_COLUMN = "저수율_5일EMA"
MODELS_DIR = PROJECT_ROOT / "models"
BASELINE_METRICS_PATH = MODELS_DIR / "baseline_metrics.csv"
BASELINE_BY_RESERVOIR_PATH = MODELS_DIR / "baseline_metrics_by_reservoir.csv"


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate MAE, RMSE, and safe MAPE."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    residual = y_true - y_pred

    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))

    safe_denominator = np.where(np.abs(y_true) < 1e-6, np.nan, np.abs(y_true))
    mape_values = np.abs(residual) / safe_denominator
    mape = float(np.nanmean(mape_values) * 100) if not np.all(np.isnan(mape_values)) else 0.0

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
    }


def _inverse_scale_sequences(X_scaled: np.ndarray, scaler) -> np.ndarray:
    if X_scaled.ndim != 3:
        raise ValueError("X_test는 3차원 배열이어야 합니다.")

    original_shape = X_scaled.shape
    X_2d = X_scaled.reshape(-1, original_shape[-1])
    X_original = scaler.inverse_transform(X_2d).reshape(original_shape)
    return X_original.astype(np.float32)


def _make_baseline_predictions(
    X_test_original: np.ndarray,
    target_feature_index: int,
    forecast_horizon: int,
) -> dict[str, np.ndarray]:
    target_history = X_test_original[:, :, target_feature_index]

    persistence = target_history[:, -1]
    recent_7day_mean = target_history[:, -7:].mean(axis=1)
    recent_30day_mean = target_history[:, -30:].mean(axis=1)

    daily_change = (target_history[:, -1] - target_history[:, 0]) / (
        target_history.shape[1] - 1
    )
    recent_trend = target_history[:, -1] + daily_change * forecast_horizon
    recent_trend = np.clip(recent_trend, 0.0, 100.0)

    return {
        "Persistence": persistence,
        "Recent 7-day mean": recent_7day_mean,
        "Recent 30-day mean": recent_30day_mean,
        "Recent trend": recent_trend,
    }


def _metrics_row(scope: str, model_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    metrics = calculate_metrics(y_true, y_pred)
    return {
        "scope": scope,
        "model": model_name,
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "MAPE": metrics["MAPE"],
        "n_samples": int(len(y_true)),
    }


def _reservoir_metrics_row(
    reservoir_name: str,
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    metrics = calculate_metrics(y_true, y_pred)
    return {
        "저수지명": reservoir_name,
        "model": model_name,
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "MAPE": metrics["MAPE"],
        "n_samples": int(len(y_true)),
    }


def evaluate_baselines(
    source: str = "auto",
    input_window: int = 60,
    forecast_horizon: int = 30,
    target_col: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate simple baseline models on all reservoirs and Ohbong separately."""
    prepared = prepare_train_test_data(
        source=source,
        input_window=input_window,
        forecast_horizon=forecast_horizon,
        target_col=target_col,
    )

    X_test_scaled = prepared["X_test"]
    y_test = prepared["y_test"]
    meta_test = prepared["meta_test"]
    feature_cols = prepared["feature_cols"]
    scaler = prepared["scaler"]

    if target_col not in feature_cols:
        raise ValueError(
            f"feature_cols에서 target_col '{target_col}'을 찾을 수 없습니다. "
            f"feature_cols: {feature_cols}"
        )

    target_feature_index = feature_cols.index(target_col)
    X_test_original = _inverse_scale_sequences(X_test_scaled, scaler)
    predictions = _make_baseline_predictions(
        X_test_original,
        target_feature_index,
        forecast_horizon,
    )

    summary_rows = []
    reservoir_rows = []

    ohbong_mask = (meta_test["저수지명"] == "오봉").to_numpy()

    for model_name, y_pred in predictions.items():
        summary_rows.append(_metrics_row("all_reservoirs", model_name, y_test, y_pred))

        if ohbong_mask.any():
            summary_rows.append(
                _metrics_row(
                    "ohbong_only",
                    model_name,
                    y_test[ohbong_mask],
                    y_pred[ohbong_mask],
                )
            )

        for reservoir_name in sorted(meta_test["저수지명"].unique()):
            reservoir_mask = (meta_test["저수지명"] == reservoir_name).to_numpy()
            reservoir_rows.append(
                _reservoir_metrics_row(
                    reservoir_name,
                    model_name,
                    y_test[reservoir_mask],
                    y_pred[reservoir_mask],
                )
            )

    summary_df = pd.DataFrame(summary_rows)
    by_reservoir_df = pd.DataFrame(reservoir_rows)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(BASELINE_METRICS_PATH, index=False, encoding="utf-8-sig")
    by_reservoir_df.to_csv(BASELINE_BY_RESERVOIR_PATH, index=False, encoding="utf-8-sig")

    return summary_df, by_reservoir_df


def _print_results(summary_df: pd.DataFrame, by_reservoir_df: pd.DataFrame) -> None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 140)

    print("=== 전체 baseline 성능 ===")
    print(
        summary_df[summary_df["scope"] == "all_reservoirs"]
        .sort_values("MAE")
        .to_string(index=False)
    )

    print("\n=== 오봉 baseline 성능 ===")
    ohbong_summary = summary_df[summary_df["scope"] == "ohbong_only"].sort_values("MAE")
    if ohbong_summary.empty:
        print("오봉 test sample이 없습니다.")
    else:
        print(ohbong_summary.to_string(index=False))

    persistence_df = by_reservoir_df[
        by_reservoir_df["model"] == "Persistence"
    ].sort_values("MAE")

    print("\n=== 저수지별 Persistence baseline 성능 상위 5개 ===")
    print(persistence_df.head(5).to_string(index=False))

    print("\n=== 저수지별 Persistence baseline 성능 하위 5개 ===")
    print(persistence_df.tail(5).sort_values("MAE", ascending=False).to_string(index=False))

    print(f"\n저장 경로: {BASELINE_METRICS_PATH}")
    print(f"저장 경로: {BASELINE_BY_RESERVOIR_PATH}")
    print("\nTransformer 모델의 성능은 이 baseline 결과와 비교하여 해석해야 합니다.")
    print(
        "만약 Transformer MAE가 단순 baseline보다 크다면, "
        "모델 구조 또는 데이터 분할 방식을 개선해야 합니다."
    )


if __name__ == "__main__":
    try:
        summary_metrics, reservoir_metrics = evaluate_baselines()
        _print_results(summary_metrics, reservoir_metrics)
    except Exception as error:
        print("baseline 평가에 실패했습니다.")
        print(error)
