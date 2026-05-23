import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.train_transformer import TimeSeriesTransformer  # noqa: E402
from src.preprocessing.prepare_timeseries import prepare_train_test_data  # noqa: E402


MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "transformer_ohbong.pt"
CONFIG_PATH = MODELS_DIR / "model_config.json"
PREDICTIONS_PATH = MODELS_DIR / "transformer_test_predictions.csv"
METRICS_PATH = MODELS_DIR / "transformer_metrics.csv"
METRICS_BY_RESERVOIR_PATH = MODELS_DIR / "transformer_metrics_by_reservoir.csv"
BASELINE_METRICS_PATH = MODELS_DIR / "baseline_metrics.csv"
BASELINE_BY_RESERVOIR_PATH = MODELS_DIR / "baseline_metrics_by_reservoir.csv"


def _require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{description} 파일이 없습니다: {path}. "
            "먼저 `python src/model/train_transformer.py`를 실행하세요."
        )


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_model_for_evaluation() -> tuple[TimeSeriesTransformer, dict, torch.device]:
    """Restore the trained Transformer model for evaluation."""
    _require_file(MODEL_PATH, "모델")
    _require_file(CONFIG_PATH, "모델 설정")

    config = _load_json(CONFIG_PATH)
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

    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, config, device


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


def predict_test_set(batch_size: int = 256) -> tuple[pd.DataFrame, dict]:
    """Run trained Transformer predictions on the full test set."""
    model, config, device = load_model_for_evaluation()
    prepared = prepare_train_test_data(
        source="auto",
        input_window=int(config.get("input_window", 60)),
        forecast_horizon=int(config.get("forecast_horizon", 30)),
        target_col=config.get("target_col", "저수율_5일EMA"),
    )

    X_test = prepared["X_test"]
    y_test = prepared["y_test"]
    meta_test = prepared["meta_test"].copy()

    X_tensor = torch.as_tensor(X_test, dtype=torch.float32)
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    predictions = []
    with torch.no_grad():
        for (X_batch,) in dataloader:
            X_batch = X_batch.to(device)
            batch_pred = model(X_batch).cpu().numpy().reshape(-1)
            predictions.append(batch_pred)

    y_pred = np.concatenate(predictions) if predictions else np.array([], dtype=np.float32)
    y_pred = np.clip(y_pred, 0.0, 100.0)

    result_df = pd.DataFrame(
        {
            "저수지명": meta_test["저수지명"].to_numpy(),
            "기준날짜": pd.to_datetime(meta_test["기준날짜"]).dt.date.astype(str),
            "예측대상날짜": pd.to_datetime(meta_test["예측대상날짜"]).dt.date.astype(str),
            "y_true": y_test.astype(float),
            "y_pred": y_pred.astype(float),
        }
    )
    result_df["residual"] = result_df["y_true"] - result_df["y_pred"]
    result_df["abs_error"] = result_df["residual"].abs()

    return result_df, config


def _summary_row(scope: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    metrics = calculate_metrics(y_true, y_pred)
    return {
        "scope": scope,
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "MAPE": metrics["MAPE"],
        "n_samples": int(len(y_true)),
    }


def _build_metrics(
    predictions_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [
        _summary_row(
            "all_reservoirs",
            predictions_df["y_true"].to_numpy(),
            predictions_df["y_pred"].to_numpy(),
        )
    ]

    ohbong_df = predictions_df[predictions_df["저수지명"] == "오봉"]
    if not ohbong_df.empty:
        rows.append(
            _summary_row(
                "ohbong_only",
                ohbong_df["y_true"].to_numpy(),
                ohbong_df["y_pred"].to_numpy(),
            )
        )

    metrics_df = pd.DataFrame(rows)

    reservoir_rows = []
    for reservoir_name, group in predictions_df.groupby("저수지명"):
        metrics = calculate_metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy())
        reservoir_rows.append(
            {
                "저수지명": reservoir_name,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "MAPE": metrics["MAPE"],
                "n_samples": int(len(group)),
            }
        )

    by_reservoir_df = pd.DataFrame(reservoir_rows).sort_values("MAE").reset_index(drop=True)
    return metrics_df, by_reservoir_df


def evaluate_transformer() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate trained Transformer and save prediction/metric CSV files."""
    predictions_df, _ = predict_test_set()
    metrics_df, by_reservoir_df = _build_metrics(predictions_df)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(PREDICTIONS_PATH, index=False, encoding="utf-8-sig")
    metrics_df.to_csv(METRICS_PATH, index=False, encoding="utf-8-sig")
    by_reservoir_df.to_csv(METRICS_BY_RESERVOIR_PATH, index=False, encoding="utf-8-sig")

    return predictions_df, metrics_df, by_reservoir_df


def _get_metric_value(metrics_df: pd.DataFrame, scope: str, metric: str) -> float | None:
    selected = metrics_df[metrics_df["scope"] == scope]
    if selected.empty:
        return None
    return float(selected.iloc[0][metric])


def _load_persistence_baseline(scope: str) -> float | None:
    if not BASELINE_METRICS_PATH.exists():
        return None

    baseline_df = pd.read_csv(BASELINE_METRICS_PATH)
    selected = baseline_df[
        (baseline_df["scope"] == scope) & (baseline_df["model"] == "Persistence")
    ]
    if selected.empty:
        return None
    return float(selected.iloc[0]["MAE"])


def _print_comparison(metrics_df: pd.DataFrame) -> None:
    all_transformer_mae = _get_metric_value(metrics_df, "all_reservoirs", "MAE")
    ohbong_transformer_mae = _get_metric_value(metrics_df, "ohbong_only", "MAE")
    all_persistence_mae = _load_persistence_baseline("all_reservoirs")
    ohbong_persistence_mae = _load_persistence_baseline("ohbong_only")

    if all_persistence_mae is None and ohbong_persistence_mae is None:
        print("\nbaseline 결과 파일이 없어 Persistence 비교를 생략합니다.")
        print(f"필요 파일: {BASELINE_METRICS_PATH}")
        return

    print("\n=== Persistence baseline 비교 ===")
    if all_transformer_mae is not None and all_persistence_mae is not None:
        print(
            "전체 기준 Transformer MAE vs Persistence MAE: "
            f"{all_transformer_mae:.4f} vs {all_persistence_mae:.4f}"
        )
    if ohbong_transformer_mae is not None and ohbong_persistence_mae is not None:
        print(
            "오봉 기준 Transformer MAE vs 오봉 Persistence MAE: "
            f"{ohbong_transformer_mae:.4f} vs {ohbong_persistence_mae:.4f}"
        )

    comparable_pairs = []
    if all_transformer_mae is not None and all_persistence_mae is not None:
        comparable_pairs.append(all_transformer_mae < all_persistence_mae)
    if ohbong_transformer_mae is not None and ohbong_persistence_mae is not None:
        comparable_pairs.append(ohbong_transformer_mae < ohbong_persistence_mae)

    if comparable_pairs and all(comparable_pairs):
        print("Transformer가 Persistence baseline보다 개선되었습니다.")
    else:
        print("현재 Transformer는 baseline보다 성능이 낮으므로 모델 개선이 필요합니다.")


def _print_results(metrics_df: pd.DataFrame, by_reservoir_df: pd.DataFrame) -> None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 140)

    print("=== 전체 Transformer 성능 ===")
    print(metrics_df[metrics_df["scope"] == "all_reservoirs"].to_string(index=False))

    print("\n=== 오봉 Transformer 성능 ===")
    ohbong_metrics = metrics_df[metrics_df["scope"] == "ohbong_only"]
    if ohbong_metrics.empty:
        print("오봉 test sample이 없습니다.")
    else:
        print(ohbong_metrics.to_string(index=False))

    print("\n=== 저수지별 Transformer 성능 상위 5개 ===")
    print(by_reservoir_df.head(5).to_string(index=False))

    print("\n=== 저수지별 Transformer 성능 하위 5개 ===")
    print(by_reservoir_df.tail(5).sort_values("MAE", ascending=False).to_string(index=False))

    print(f"\n저장 경로: {PREDICTIONS_PATH}")
    print(f"저장 경로: {METRICS_PATH}")
    print(f"저장 경로: {METRICS_BY_RESERVOIR_PATH}")

    if BASELINE_BY_RESERVOIR_PATH.exists():
        print(f"저수지별 baseline 비교 파일 확인됨: {BASELINE_BY_RESERVOIR_PATH}")

    _print_comparison(metrics_df)


if __name__ == "__main__":
    try:
        _, transformer_metrics, transformer_by_reservoir = evaluate_transformer()
        _print_results(transformer_metrics, transformer_by_reservoir)
    except Exception as error:
        print("Transformer 평가에 실패했습니다.")
        print(error)
