import json
import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.db.supabase_client import load_master_dataset  # noqa: E402
from src.preprocessing.prepare_timeseries import (  # noqa: E402
    DEFAULT_FEATURE_COLUMNS,
    add_time_features,
    clean_master_dataset,
)

try:
    from src.model.train_transformer import TimeSeriesTransformer
except Exception as exc:
    raise ImportError(
        "TimeSeriesTransformer import에 실패했습니다. "
        "프로젝트 루트에서 `python src/model/train_transformer_delta.py`로 실행하세요."
    ) from exc


MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "delta_transformer_ohbong.pt"
CONFIG_PATH = MODELS_DIR / "delta_model_config.json"
X_SCALER_PATH = MODELS_DIR / "delta_scaler.joblib"
Y_SCALER_PATH = MODELS_DIR / "delta_target_scaler.joblib"
RESIDUAL_STATS_PATH = MODELS_DIR / "delta_residual_stats.json"
PREDICTIONS_PATH = MODELS_DIR / "delta_test_predictions.csv"
METRICS_PATH = MODELS_DIR / "delta_metrics.csv"
METRICS_BY_RESERVOIR_PATH = MODELS_DIR / "delta_metrics_by_reservoir.csv"
BASELINE_METRICS_PATH = MODELS_DIR / "baseline_metrics.csv"


class ReservoirDataset(Dataset):
    def __init__(self, X: np.ndarray, y_scaled: np.ndarray) -> None:
        if len(X) != len(y_scaled):
            raise ValueError("X와 y_scaled의 sample 수가 서로 다릅니다.")
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y_scaled, dtype=torch.float32).reshape(-1, 1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[index], self.y[index]


def create_delta_sequences(
    df: pd.DataFrame,
    input_window: int = 60,
    forecast_horizon: int = 30,
    target_col: str = "저수율_5일EMA",
    feature_cols: Optional[list[str]] = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str]]:
    if input_window <= 0:
        raise ValueError("input_window는 1 이상의 정수여야 합니다.")
    if forecast_horizon <= 0:
        raise ValueError("forecast_horizon은 1 이상의 정수여야 합니다.")

    selected_features = feature_cols or DEFAULT_FEATURE_COLUMNS
    required_columns = ["저수지명", "날짜", target_col, *selected_features]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"delta sequence 생성에 필요한 컬럼이 없습니다: {missing}")

    X_values = []
    y_delta_values = []
    metadata_rows = []

    sorted_df = df.sort_values(["저수지명", "날짜"]).reset_index(drop=True)
    for reservoir_name, group in sorted_df.groupby("저수지명", sort=False):
        group = group.sort_values("날짜").reset_index(drop=True)
        features = group[selected_features].to_numpy(dtype=np.float32)
        target = group[target_col].to_numpy(dtype=np.float32)
        dates = pd.to_datetime(group["날짜"]).to_numpy()

        max_start = len(group) - input_window - forecast_horizon + 1
        if max_start <= 0:
            continue

        for start_idx in range(max_start):
            end_idx = start_idx + input_window
            target_idx = end_idx + forecast_horizon - 1
            current_rate = float(target[end_idx - 1])
            future_rate = float(target[target_idx])
            delta_target = future_rate - current_rate

            X_values.append(features[start_idx:end_idx])
            y_delta_values.append(delta_target)
            metadata_rows.append(
                {
                    "저수지명": reservoir_name,
                    "기준날짜": pd.Timestamp(dates[end_idx - 1]),
                    "예측대상날짜": pd.Timestamp(dates[target_idx]),
                    "current_rate": current_rate,
                    "future_rate": future_rate,
                }
            )

    if not X_values:
        raise ValueError("생성된 delta sequence가 없습니다. window와 horizon을 확인하세요.")

    return (
        np.stack(X_values).astype(np.float32),
        np.asarray(y_delta_values, dtype=np.float32),
        pd.DataFrame(metadata_rows),
        selected_features,
    )


def split_by_date_delta(
    X: np.ndarray,
    y_delta: np.ndarray,
    metadata: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_dates = pd.to_datetime(metadata["예측대상날짜"])

    train_mask = target_dates <= pd.Timestamp("2023-12-31")
    val_mask = (target_dates >= pd.Timestamp("2024-01-01")) & (
        target_dates <= pd.Timestamp("2024-12-31")
    )
    test_mask = (target_dates >= pd.Timestamp("2025-01-01")) & (
        target_dates <= pd.Timestamp("2025-12-31")
    )

    if not train_mask.any():
        raise ValueError("train 데이터가 없습니다. 날짜 분할 조건을 확인하세요.")
    if not val_mask.any():
        raise ValueError("validation 데이터가 없습니다. 날짜 분할 조건을 확인하세요.")
    if not test_mask.any():
        raise ValueError("test 데이터가 없습니다. 날짜 분할 조건을 확인하세요.")

    return (
        X[train_mask.to_numpy()],
        X[val_mask.to_numpy()],
        X[test_mask.to_numpy()],
        y_delta[train_mask.to_numpy()],
        y_delta[val_mask.to_numpy()],
        y_delta[test_mask.to_numpy()],
        metadata.loc[train_mask].reset_index(drop=True),
        metadata.loc[val_mask].reset_index(drop=True),
        metadata.loc[test_mask].reset_index(drop=True),
    )


def _scale_3d(
    scaler: StandardScaler,
    X: np.ndarray,
    fit: bool = False,
) -> np.ndarray:
    original_shape = X.shape
    X_2d = X.reshape(-1, original_shape[-1])
    scaled = scaler.fit_transform(X_2d) if fit else scaler.transform(X_2d)
    return scaled.reshape(original_shape).astype(np.float32)


def scale_x_and_y(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, StandardScaler]:
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_scaled = _scale_3d(x_scaler, X_train, fit=True)
    X_val_scaled = _scale_3d(x_scaler, X_val)
    X_test_scaled = _scale_3d(x_scaler, X_test)

    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).astype(np.float32)
    y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).astype(np.float32)
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).astype(np.float32)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(x_scaler, X_SCALER_PATH)
    joblib.dump(y_scaler, Y_SCALER_PATH)

    return (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        y_train_scaled,
        y_val_scaled,
        y_test_scaled,
        x_scaler,
        y_scaler,
    )


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    residual = y_true - y_pred

    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    safe_denominator = np.where(np.abs(y_true) < 1e-6, np.nan, np.abs(y_true))
    mape_values = np.abs(residual) / safe_denominator
    mape = float(np.nanmean(mape_values) * 100) if not np.all(np.isnan(mape_values)) else 0.0

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        total_samples += X_batch.size(0)

    return total_loss / max(total_samples, 1)


def evaluate_loss(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            loss = criterion(model(X_batch), y_batch)
            total_loss += loss.item() * X_batch.size(0)
            total_samples += X_batch.size(0)
    return total_loss / max(total_samples, 1)


def predict_delta_scaled(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    model.eval()
    loader = DataLoader(
        torch.as_tensor(X, dtype=torch.float32),
        batch_size=batch_size,
        shuffle=False,
    )
    predictions = []
    with torch.no_grad():
        for X_batch in loader:
            X_batch = X_batch.to(device)
            predictions.append(model(X_batch).cpu().numpy().reshape(-1))
    return np.concatenate(predictions) if predictions else np.array([], dtype=np.float32)


def _build_prediction_frame(
    meta_test: pd.DataFrame,
    predicted_delta: np.ndarray,
) -> pd.DataFrame:
    y_true = meta_test["future_rate"].to_numpy(dtype=np.float64)
    current_rate = meta_test["current_rate"].to_numpy(dtype=np.float64)
    y_pred = np.clip(current_rate + predicted_delta, 0.0, 100.0)

    result = pd.DataFrame(
        {
            "저수지명": meta_test["저수지명"].to_numpy(),
            "기준날짜": pd.to_datetime(meta_test["기준날짜"]).dt.date.astype(str),
            "예측대상날짜": pd.to_datetime(meta_test["예측대상날짜"]).dt.date.astype(str),
            "current_rate": current_rate,
            "future_rate": y_true,
            "predicted_delta": predicted_delta.astype(float),
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    result["residual"] = result["y_true"] - result["y_pred"]
    result["abs_error"] = result["residual"].abs()
    return result


def _metrics_row(scope: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    metrics = calculate_metrics(y_true, y_pred)
    return {
        "scope": scope,
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "MAPE": metrics["MAPE"],
        "n_samples": int(len(y_true)),
    }


def _build_metrics(predictions_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [
        _metrics_row(
            "all_reservoirs",
            predictions_df["y_true"].to_numpy(),
            predictions_df["y_pred"].to_numpy(),
        )
    ]

    ohbong_df = predictions_df[predictions_df["저수지명"] == "오봉"]
    if not ohbong_df.empty:
        rows.append(
            _metrics_row(
                "ohbong_only",
                ohbong_df["y_true"].to_numpy(),
                ohbong_df["y_pred"].to_numpy(),
            )
        )

    by_reservoir_rows = []
    for reservoir_name, group in predictions_df.groupby("저수지명"):
        metrics = calculate_metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy())
        by_reservoir_rows.append(
            {
                "저수지명": reservoir_name,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "MAPE": metrics["MAPE"],
                "n_samples": int(len(group)),
            }
        )

    return (
        pd.DataFrame(rows),
        pd.DataFrame(by_reservoir_rows).sort_values("MAE").reset_index(drop=True),
    )


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def _load_persistence_mae(scope: str) -> float | None:
    if not BASELINE_METRICS_PATH.exists():
        return None
    baseline_df = pd.read_csv(BASELINE_METRICS_PATH)
    selected = baseline_df[
        (baseline_df["scope"] == scope) & (baseline_df["model"] == "Persistence")
    ]
    if selected.empty:
        return None
    return float(selected.iloc[0]["MAE"])


def _metric_value(metrics_df: pd.DataFrame, scope: str, metric: str) -> float | None:
    selected = metrics_df[metrics_df["scope"] == scope]
    if selected.empty:
        return None
    return float(selected.iloc[0][metric])


def train_delta_model(
    source: str = "auto",
    input_window: int = 60,
    forecast_horizon: int = 30,
    target_col: str = "저수율_5일EMA",
    batch_size: int = 64,
    epochs: int = 80,
    learning_rate: float = 0.0005,
    weight_decay: float = 1e-4,
    patience: int = 12,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 128,
    dropout: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_df = load_master_dataset(source=source)
    prepared_df = add_time_features(clean_master_dataset(raw_df))
    X, y_delta, metadata, feature_cols = create_delta_sequences(
        prepared_df,
        input_window=input_window,
        forecast_horizon=forecast_horizon,
        target_col=target_col,
    )
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        meta_train,
        meta_val,
        meta_test,
    ) = split_by_date_delta(X, y_delta, metadata)

    (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        y_train_scaled,
        y_val_scaled,
        _,
        _,
        y_scaler,
    ) = scale_x_and_y(X_train, X_val, X_test, y_train, y_val, y_test)

    train_loader = DataLoader(
        ReservoirDataset(X_train_scaled, y_train_scaled),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        ReservoirDataset(X_val_scaled, y_val_scaled),
        batch_size=batch_size,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 device: {device}")

    model = TimeSeriesTransformer(
        num_features=len(feature_cols),
        input_window=input_window,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    ).to(device)
    criterion = nn.HuberLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        is_best = val_loss < best_val_loss

        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train loss: {train_loss:.6f} | "
            f"validation loss: {val_loss:.6f} "
            f"{'BEST' if is_best else ''}"
        )

        if epochs_without_improvement >= patience:
            print(f"Early stopping: {patience} epoch 동안 개선이 없어 중단합니다.")
            break

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    predicted_delta_scaled = predict_delta_scaled(model, X_test_scaled, device)
    predicted_delta = y_scaler.inverse_transform(
        predicted_delta_scaled.reshape(-1, 1)
    ).reshape(-1)

    predictions_df = _build_prediction_frame(meta_test, predicted_delta)
    metrics_df, by_reservoir_df = _build_metrics(predictions_df)

    all_metrics = metrics_df[metrics_df["scope"] == "all_reservoirs"].iloc[0].to_dict()
    ohbong_metrics_df = metrics_df[metrics_df["scope"] == "ohbong_only"]
    ohbong_metrics = (
        ohbong_metrics_df.iloc[0].to_dict()
        if not ohbong_metrics_df.empty
        else {"MAE": None, "RMSE": None, "MAPE": None}
    )

    residuals = predictions_df["residual"].to_numpy(dtype=np.float64)
    config = {
        "model_type": "delta_transformer",
        "input_window": input_window,
        "forecast_horizon": forecast_horizon,
        "target_col": target_col,
        "feature_cols": feature_cols,
        "num_features": len(feature_cols),
        "d_model": d_model,
        "nhead": nhead,
        "num_layers": num_layers,
        "dim_feedforward": dim_feedforward,
        "dropout": dropout,
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
    }
    residual_stats = {
        "mae": float(all_metrics["MAE"]),
        "rmse": float(all_metrics["RMSE"]),
        "mape": float(all_metrics["MAPE"]),
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0,
        "ohbong_mae": None if ohbong_metrics["MAE"] is None else float(ohbong_metrics["MAE"]),
        "ohbong_rmse": None if ohbong_metrics["RMSE"] is None else float(ohbong_metrics["RMSE"]),
        "ohbong_mape": None if ohbong_metrics["MAPE"] is None else float(ohbong_metrics["MAPE"]),
    }

    predictions_df.to_csv(PREDICTIONS_PATH, index=False, encoding="utf-8-sig")
    metrics_df.to_csv(METRICS_PATH, index=False, encoding="utf-8-sig")
    by_reservoir_df.to_csv(METRICS_BY_RESERVOIR_PATH, index=False, encoding="utf-8-sig")
    _save_json(CONFIG_PATH, config)
    _save_json(RESIDUAL_STATS_PATH, residual_stats)

    print("\n=== Delta Transformer 학습 완료 ===")
    print(f"best epoch: {best_epoch}")
    print(f"best validation loss: {best_val_loss:.6f}")
    _print_results(metrics_df, by_reservoir_df)
    _print_baseline_comparison(metrics_df)
    print(f"\n저장된 모델 경로: {MODEL_PATH}")
    print(f"저장된 config 경로: {CONFIG_PATH}")
    print(f"저장된 residual_stats 경로: {RESIDUAL_STATS_PATH}")

    return predictions_df, metrics_df, by_reservoir_df


def _print_results(metrics_df: pd.DataFrame, by_reservoir_df: pd.DataFrame) -> None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 140)

    print("\n=== 전체 test 성능 ===")
    print(metrics_df[metrics_df["scope"] == "all_reservoirs"].to_string(index=False))

    print("\n=== 오봉 test 성능 ===")
    ohbong_metrics = metrics_df[metrics_df["scope"] == "ohbong_only"]
    print(
        "오봉 test sample이 없습니다."
        if ohbong_metrics.empty
        else ohbong_metrics.to_string(index=False)
    )

    print("\n=== 저수지별 성능 상위 5개 ===")
    print(by_reservoir_df.head(5).to_string(index=False))

    print("\n=== 저수지별 성능 하위 5개 ===")
    print(by_reservoir_df.tail(5).sort_values("MAE", ascending=False).to_string(index=False))


def _print_baseline_comparison(metrics_df: pd.DataFrame) -> None:
    all_delta_mae = _metric_value(metrics_df, "all_reservoirs", "MAE")
    ohbong_delta_mae = _metric_value(metrics_df, "ohbong_only", "MAE")
    all_persistence_mae = _load_persistence_mae("all_reservoirs")
    ohbong_persistence_mae = _load_persistence_mae("ohbong_only")

    print("\n=== Persistence baseline 비교 ===")
    if all_persistence_mae is None or ohbong_persistence_mae is None:
        print(f"baseline 파일이 없거나 필요한 행이 없습니다: {BASELINE_METRICS_PATH}")
        print("Persistence baseline과의 비교는 baseline 평가 후 다시 확인하세요.")
        return

    print(
        "전체 기준 Delta Transformer MAE vs Persistence MAE: "
        f"{all_delta_mae:.4f} vs {all_persistence_mae:.4f}"
    )
    if ohbong_delta_mae is not None:
        print(
            "오봉 기준 Delta Transformer MAE vs 오봉 Persistence MAE: "
            f"{ohbong_delta_mae:.4f} vs {ohbong_persistence_mae:.4f}"
        )

    if all_delta_mae < all_persistence_mae:
        print("전체 기준 Delta Transformer MAE가 Persistence보다 낮아 개선 성공입니다.")
    else:
        print("전체 기준 Delta Transformer는 추가 튜닝이 필요합니다.")

    if ohbong_delta_mae is not None and ohbong_delta_mae < ohbong_persistence_mae:
        print("오봉 기준 Delta Transformer MAE가 오봉 Persistence보다 낮아 오봉 예측 개선 성공입니다.")
    else:
        print("오봉 기준 Delta Transformer는 추가 튜닝이 필요합니다.")


if __name__ == "__main__":
    try:
        train_delta_model()
    except Exception as error:
        print("Delta Transformer 학습에 실패했습니다.")
        print(error)
