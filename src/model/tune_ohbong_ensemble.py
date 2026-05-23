import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.db.supabase_client import load_master_dataset  # noqa: E402
from src.model.train_transformer import TimeSeriesTransformer  # noqa: E402
from src.model.train_transformer_delta import create_delta_sequences  # noqa: E402
from src.preprocessing.prepare_timeseries import (  # noqa: E402
    add_time_features,
    clean_master_dataset,
)


MODELS_DIR = PROJECT_ROOT / "models"
DELTA_MODEL_PATH = MODELS_DIR / "delta_transformer_ohbong.pt"
DELTA_CONFIG_PATH = MODELS_DIR / "delta_model_config.json"
DELTA_X_SCALER_PATH = MODELS_DIR / "delta_scaler.joblib"
DELTA_Y_SCALER_PATH = MODELS_DIR / "delta_target_scaler.joblib"

ALPHA_RESULTS_PATH = MODELS_DIR / "ohbong_ensemble_alpha_results.csv"
VAL_PREDICTIONS_PATH = MODELS_DIR / "ohbong_ensemble_predictions_val.csv"
TEST_PREDICTIONS_PATH = MODELS_DIR / "ohbong_ensemble_predictions_test.csv"
METRICS_PATH = MODELS_DIR / "ohbong_ensemble_metrics.csv"
CONFIG_PATH = MODELS_DIR / "ohbong_ensemble_config.json"

TARGET_RESERVOIR = "오봉"


def _require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{description} 파일이 없습니다: {path}. "
            "먼저 `python src/model/train_transformer_delta.py`를 실행하세요."
        )


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_delta_model():
    """Load Delta Transformer model, scalers, config, and device."""
    _require_file(DELTA_MODEL_PATH, "Delta Transformer 모델")
    _require_file(DELTA_CONFIG_PATH, "Delta 모델 설정")
    _require_file(DELTA_X_SCALER_PATH, "Delta X scaler")
    _require_file(DELTA_Y_SCALER_PATH, "Delta target scaler")

    config = _load_json(DELTA_CONFIG_PATH)
    x_scaler = joblib.load(DELTA_X_SCALER_PATH)
    y_scaler = joblib.load(DELTA_Y_SCALER_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TimeSeriesTransformer(
        num_features=int(config["num_features"]),
        input_window=int(config["input_window"]),
        d_model=int(config.get("d_model", 64)),
        nhead=int(config.get("nhead", 4)),
        num_layers=int(config.get("num_layers", 2)),
        dim_feedforward=int(config.get("dim_feedforward", 128)),
        dropout=float(config.get("dropout", 0.2)),
    ).to(device)

    state_dict = torch.load(DELTA_MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, x_scaler, y_scaler, config, device


def _scale_x(X: np.ndarray, x_scaler) -> np.ndarray:
    original_shape = X.shape
    X_2d = X.reshape(-1, original_shape[-1])
    return x_scaler.transform(X_2d).reshape(original_shape).astype(np.float32)


def make_delta_predictions_for_split(
    X: np.ndarray,
    metadata: pd.DataFrame,
    model,
    x_scaler,
    y_scaler,
    device: torch.device,
    batch_size: int = 256,
) -> pd.DataFrame:
    """Predict final storage rates with Delta Transformer for one split."""
    X_scaled = _scale_x(X, x_scaler)
    loader = DataLoader(
        torch.as_tensor(X_scaled, dtype=torch.float32),
        batch_size=batch_size,
        shuffle=False,
    )

    predicted_delta_scaled = []
    with torch.no_grad():
        for X_batch in loader:
            X_batch = X_batch.to(device)
            predicted_delta_scaled.append(model(X_batch).cpu().numpy().reshape(-1))

    predicted_delta_scaled = (
        np.concatenate(predicted_delta_scaled)
        if predicted_delta_scaled
        else np.array([], dtype=np.float32)
    )
    predicted_delta = y_scaler.inverse_transform(
        predicted_delta_scaled.reshape(-1, 1)
    ).reshape(-1)

    current_rate = metadata["current_rate"].to_numpy(dtype=np.float64)
    future_rate = metadata["future_rate"].to_numpy(dtype=np.float64)
    delta_prediction = np.clip(current_rate + predicted_delta, 0.0, 100.0)

    return pd.DataFrame(
        {
            "저수지명": metadata["저수지명"].to_numpy(),
            "기준날짜": pd.to_datetime(metadata["기준날짜"]).dt.date.astype(str),
            "예측대상날짜": pd.to_datetime(metadata["예측대상날짜"]).dt.date.astype(str),
            "current_rate": current_rate,
            "future_rate": future_rate,
            "delta_prediction": delta_prediction,
            "persistence_prediction": current_rate,
            "y_true": future_rate,
        }
    )


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

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def find_best_alpha(validation_df: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """Find alpha minimizing Ohbong validation MAE."""
    if validation_df.empty:
        raise ValueError("alpha 튜닝에 사용할 오봉 validation 데이터가 없습니다.")

    rows = []
    y_true = validation_df["y_true"].to_numpy(dtype=np.float64)
    delta_pred = validation_df["delta_prediction"].to_numpy(dtype=np.float64)
    persistence_pred = validation_df["persistence_prediction"].to_numpy(dtype=np.float64)

    for alpha_int in range(0, 101):
        alpha = alpha_int / 100
        ensemble_pred = np.clip(
            alpha * delta_pred + (1 - alpha) * persistence_pred,
            0.0,
            100.0,
        )
        metrics = calculate_metrics(y_true, ensemble_pred)
        rows.append(
            {
                "alpha": alpha,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "MAPE": metrics["MAPE"],
                "n_samples": int(len(validation_df)),
            }
        )

    alpha_results_df = pd.DataFrame(rows).sort_values(["MAE", "alpha"]).reset_index(drop=True)
    best_alpha = float(alpha_results_df.iloc[0]["alpha"])
    return best_alpha, alpha_results_df


def apply_ensemble(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Apply delta-persistence weighted ensemble and add error columns."""
    result = df.copy()
    result["final_prediction"] = np.clip(
        alpha * result["delta_prediction"]
        + (1 - alpha) * result["persistence_prediction"],
        0.0,
        100.0,
    )
    result["residual"] = result["y_true"] - result["final_prediction"]
    result["abs_error"] = result["residual"].abs()
    return result


def _split_validation_test(
    X: np.ndarray,
    metadata: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    target_dates = pd.to_datetime(metadata["예측대상날짜"])
    val_mask = (target_dates >= pd.Timestamp("2024-01-01")) & (
        target_dates <= pd.Timestamp("2024-12-31")
    )
    test_mask = (target_dates >= pd.Timestamp("2025-01-01")) & (
        target_dates <= pd.Timestamp("2025-12-31")
    )

    if not val_mask.any():
        raise ValueError("2024년 validation 데이터가 없습니다.")
    if not test_mask.any():
        raise ValueError("2025년 test 데이터가 없습니다.")

    return (
        X[val_mask.to_numpy()],
        X[test_mask.to_numpy()],
        metadata.loc[val_mask].reset_index(drop=True),
        metadata.loc[test_mask].reset_index(drop=True),
    )


def _model_metric_row(scope: str, split: str, model_name: str, y_true, y_pred) -> dict:
    metrics = calculate_metrics(y_true, y_pred)
    return {
        "scope": scope,
        "split": split,
        "model": model_name,
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "MAPE": metrics["MAPE"],
        "n_samples": int(len(y_true)),
    }


def _build_comparison_metrics(
    ohbong_val_df: pd.DataFrame,
    ohbong_test_df: pd.DataFrame,
    all_test_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    targets = [
        ("ohbong_only", "validation", ohbong_val_df),
        ("ohbong_only", "test", ohbong_test_df),
        ("all_reservoirs", "test", all_test_df),
    ]

    for scope, split, df in targets:
        y_true = df["y_true"].to_numpy(dtype=np.float64)
        rows.append(
            _model_metric_row(
                scope,
                split,
                "Persistence",
                y_true,
                df["persistence_prediction"].to_numpy(dtype=np.float64),
            )
        )
        rows.append(
            _model_metric_row(
                scope,
                split,
                "Delta Transformer",
                y_true,
                df["delta_prediction"].to_numpy(dtype=np.float64),
            )
        )
        rows.append(
            _model_metric_row(
                scope,
                split,
                "Ohbong Ensemble",
                y_true,
                df["final_prediction"].to_numpy(dtype=np.float64),
            )
        )

    return pd.DataFrame(rows)


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def tune_ohbong_ensemble() -> tuple[float, pd.DataFrame, pd.DataFrame]:
    model, x_scaler, y_scaler, config, device = load_delta_model()

    raw_df = load_master_dataset(source="auto")
    prepared_df = add_time_features(clean_master_dataset(raw_df))
    X, _, metadata, _ = create_delta_sequences(
        prepared_df,
        input_window=int(config["input_window"]),
        forecast_horizon=int(config["forecast_horizon"]),
        target_col=config.get("target_col", "저수율_5일EMA"),
        feature_cols=list(config["feature_cols"]),
    )
    X_val, X_test, meta_val, meta_test = _split_validation_test(X, metadata)

    val_predictions = make_delta_predictions_for_split(
        X_val, meta_val, model, x_scaler, y_scaler, device
    )
    test_predictions = make_delta_predictions_for_split(
        X_test, meta_test, model, x_scaler, y_scaler, device
    )

    ohbong_val_base = val_predictions[val_predictions["저수지명"] == TARGET_RESERVOIR].copy()
    best_alpha, alpha_results_df = find_best_alpha(ohbong_val_base)

    val_ensemble = apply_ensemble(val_predictions, best_alpha)
    test_ensemble = apply_ensemble(test_predictions, best_alpha)

    ohbong_val = val_ensemble[val_ensemble["저수지명"] == TARGET_RESERVOIR].copy()
    ohbong_test = test_ensemble[test_ensemble["저수지명"] == TARGET_RESERVOIR].copy()
    if ohbong_test.empty:
        raise ValueError("오봉 test 데이터가 없습니다.")

    metrics_df = _build_comparison_metrics(ohbong_val, ohbong_test, test_ensemble)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    alpha_results_df.to_csv(ALPHA_RESULTS_PATH, index=False, encoding="utf-8-sig")
    val_ensemble.to_csv(VAL_PREDICTIONS_PATH, index=False, encoding="utf-8-sig")
    test_ensemble.to_csv(TEST_PREDICTIONS_PATH, index=False, encoding="utf-8-sig")
    metrics_df.to_csv(METRICS_PATH, index=False, encoding="utf-8-sig")

    ohbong_validation_metrics = metrics_df[
        (metrics_df["scope"] == "ohbong_only")
        & (metrics_df["split"] == "validation")
    ].to_dict(orient="records")
    ohbong_test_metrics = metrics_df[
        (metrics_df["scope"] == "ohbong_only")
        & (metrics_df["split"] == "test")
    ].to_dict(orient="records")

    ensemble_config = {
        "model_type": "ohbong_delta_persistence_ensemble",
        "best_alpha": best_alpha,
        "formula": "final_prediction = alpha * delta_prediction + (1 - alpha) * persistence_prediction",
        "validation_period": "2024-01-01 ~ 2024-12-31",
        "test_period": "2025-01-01 ~ 2025-12-31",
        "target_reservoir": TARGET_RESERVOIR,
        "input_window": int(config["input_window"]),
        "forecast_horizon": int(config["forecast_horizon"]),
        "target_col": config.get("target_col", "저수율_5일EMA"),
        "ohbong_validation_metrics": ohbong_validation_metrics,
        "ohbong_test_metrics": ohbong_test_metrics,
    }
    _save_json(CONFIG_PATH, ensemble_config)

    _print_results(best_alpha, alpha_results_df, metrics_df)
    return best_alpha, alpha_results_df, metrics_df


def _print_scope(metrics_df: pd.DataFrame, title: str, scope: str, split: str) -> None:
    print(f"\n=== {title} ===")
    selected = metrics_df[(metrics_df["scope"] == scope) & (metrics_df["split"] == split)]
    print(selected.sort_values("MAE").to_string(index=False))


def _metric(metrics_df: pd.DataFrame, scope: str, split: str, model_name: str) -> float:
    selected = metrics_df[
        (metrics_df["scope"] == scope)
        & (metrics_df["split"] == split)
        & (metrics_df["model"] == model_name)
    ]
    if selected.empty:
        return float("nan")
    return float(selected.iloc[0]["MAE"])


def _print_results(
    best_alpha: float,
    alpha_results_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
) -> None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 140)

    print(f"validation에서 찾은 best_alpha: {best_alpha:.2f}")
    _print_scope(metrics_df, "오봉 validation 성능 비교", "ohbong_only", "validation")
    _print_scope(metrics_df, "오봉 test 성능 비교", "ohbong_only", "test")
    _print_scope(metrics_df, "전체 test 참고 성능 비교", "all_reservoirs", "test")

    print("\n=== alpha별 validation MAE 상위 10개 ===")
    print(alpha_results_df.head(10).to_string(index=False))

    print(f"\n저장 경로: {ALPHA_RESULTS_PATH}")
    print(f"저장 경로: {VAL_PREDICTIONS_PATH}")
    print(f"저장 경로: {TEST_PREDICTIONS_PATH}")
    print(f"저장 경로: {METRICS_PATH}")
    print(f"저장 경로: {CONFIG_PATH}")

    ohbong_ensemble_mae = _metric(metrics_df, "ohbong_only", "test", "Ohbong Ensemble")
    ohbong_persistence_mae = _metric(metrics_df, "ohbong_only", "test", "Persistence")
    ohbong_delta_mae = _metric(metrics_df, "ohbong_only", "test", "Delta Transformer")

    improved_persistence = ohbong_ensemble_mae < ohbong_persistence_mae
    improved_delta = ohbong_ensemble_mae < ohbong_delta_mae

    if improved_persistence:
        print("오봉 기준 Ensemble 모델이 Persistence baseline보다 개선되었습니다.")
    if improved_delta:
        print("오봉 기준 Ensemble 모델이 Delta Transformer보다 개선되었습니다.")
    if not improved_persistence and not improved_delta:
        print("앙상블 보정만으로는 개선이 부족하므로, 추가 feature 또는 오봉 전용 fine-tuning이 필요합니다.")


if __name__ == "__main__":
    try:
        tune_ohbong_ensemble()
    except Exception as error:
        print("오봉 앙상블 튜닝에 실패했습니다.")
        print(error)
