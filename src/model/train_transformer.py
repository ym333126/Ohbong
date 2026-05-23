import json
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.prepare_timeseries import prepare_train_test_data  # noqa: E402


MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "transformer_ohbong.pt"
CONFIG_PATH = MODEL_DIR / "model_config.json"
RESIDUAL_STATS_PATH = MODEL_DIR / "residual_stats.json"


class ReservoirDataset(Dataset):
    """PyTorch Dataset for reservoir sequence data."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        if len(X) != len(y):
            raise ValueError("X와 y의 sample 수가 서로 다릅니다.")

        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[index], self.y[index]


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for batch_first Transformer inputs."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TimeSeriesTransformer(nn.Module):
    """Lightweight Transformer encoder for 30-day reservoir-rate prediction."""

    def __init__(
        self,
        num_features: int,
        input_window: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(num_features, d_model)
        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=input_window,
            dropout=dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.regression_head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        encoded = self.transformer_encoder(x)
        last_hidden_state = encoded[:, -1, :]
        return self.regression_head(last_hidden_state)


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
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()

        batch_size = X_batch.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    predictions_list = []
    targets_list = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)

            batch_size = X_batch.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            predictions_list.append(predictions.cpu().numpy().reshape(-1))
            targets_list.append(y_batch.cpu().numpy().reshape(-1))

    y_pred = np.concatenate(predictions_list) if predictions_list else np.array([])
    y_true = np.concatenate(targets_list) if targets_list else np.array([])
    average_loss = total_loss / max(total_samples, 1)

    return average_loss, y_pred, y_true


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    residual = y_true - y_pred

    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))

    safe_denominator = np.where(np.abs(y_true) < 1e-6, np.nan, np.abs(y_true))
    mape_values = np.abs(residual) / safe_denominator
    mape = float(np.nanmean(mape_values) * 100) if not np.all(np.isnan(mape_values)) else 0.0

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
    }


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def _split_train_validation(dataset: Dataset, validation_ratio: float = 0.2) -> tuple[Subset, Subset]:
    dataset_size = len(dataset)
    validation_size = max(1, int(dataset_size * validation_ratio))
    train_size = dataset_size - validation_size

    if train_size <= 0:
        raise ValueError("validation 분할 후 train 데이터가 없습니다.")

    train_indices = list(range(0, train_size))
    validation_indices = list(range(train_size, dataset_size))
    return Subset(dataset, train_indices), Subset(dataset, validation_indices)


def train_model(
    source: str = "auto",
    input_window: int = 60,
    forecast_horizon: int = 30,
    target_col: str = "저수율_5일EMA",
    batch_size: int = 64,
    epochs: int = 30,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
    patience: int = 7,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 128,
    dropout: float = 0.1,
    device: Optional[str] = None,
) -> dict:
    data = prepare_train_test_data(
        source=source,
        input_window=input_window,
        forecast_horizon=forecast_horizon,
        target_col=target_col,
    )

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    feature_cols = data["feature_cols"]
    num_features = len(feature_cols)

    train_dataset_full = ReservoirDataset(X_train, y_train)
    test_dataset = ReservoirDataset(X_test, y_test)
    train_dataset, validation_dataset = _split_train_validation(train_dataset_full)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    selected_device = torch.device(
        device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"사용 device: {selected_device}")

    model = TimeSeriesTransformer(
        num_features=num_features,
        input_window=input_window,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    ).to(selected_device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            selected_device,
        )
        validation_loss, _, _ = evaluate(
            model,
            validation_loader,
            criterion,
            selected_device,
        )

        is_best = validation_loss < best_validation_loss
        if is_best:
            best_validation_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            epochs_without_improvement += 1

        best_label = "BEST" if is_best else ""
        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train loss: {train_loss:.6f} | "
            f"validation loss: {validation_loss:.6f} {best_label}"
        )

        if epochs_without_improvement >= patience:
            print(
                f"Early stopping: {patience} epoch 동안 validation loss 개선이 없어 중단합니다."
            )
            break

    model.load_state_dict(torch.load(MODEL_PATH, map_location=selected_device))
    test_loss, y_pred, y_true = evaluate(model, test_loader, criterion, selected_device)
    metrics = calculate_metrics(y_true, y_pred)
    residuals = y_true - y_pred

    config = {
        "input_window": input_window,
        "forecast_horizon": forecast_horizon,
        "target_col": target_col,
        "feature_cols": feature_cols,
        "num_features": num_features,
        "d_model": d_model,
        "nhead": nhead,
        "num_layers": num_layers,
        "dim_feedforward": dim_feedforward,
        "dropout": dropout,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
    }
    residual_stats = {
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "mape": metrics["mape"],
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0,
    }

    _save_json(CONFIG_PATH, config)
    _save_json(RESIDUAL_STATS_PATH, residual_stats)

    print("\n=== 학습 완료 ===")
    print(f"best epoch: {best_epoch}")
    print(f"best validation loss: {best_validation_loss:.6f}")
    print(f"test loss: {test_loss:.6f}")
    print(f"test MAE: {metrics['mae']:.4f}")
    print(f"test RMSE: {metrics['rmse']:.4f}")
    print(f"test MAPE: {metrics['mape']:.2f}%")
    print(f"residual_std: {residual_stats['residual_std']:.4f}")
    print(f"저장된 모델 경로: {MODEL_PATH}")
    print(f"저장된 config 경로: {CONFIG_PATH}")
    print(f"저장된 residual_stats 경로: {RESIDUAL_STATS_PATH}")

    return {
        "model": model,
        "config": config,
        "residual_stats": residual_stats,
        "best_validation_loss": best_validation_loss,
        "test_loss": test_loss,
        "y_true": y_true,
        "y_pred": y_pred,
    }


if __name__ == "__main__":
    try:
        train_model()
    except Exception as error:
        print("Transformer 학습에 실패했습니다.")
        print(error)
