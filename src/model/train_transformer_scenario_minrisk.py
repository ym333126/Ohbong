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
from src.model.train_transformer import PositionalEncoding  # noqa: E402
from src.preprocessing.prepare_timeseries import (  # noqa: E402
    DEFAULT_FEATURE_COLUMNS,
    add_time_features,
    clean_master_dataset,
)


MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "minrisk_transformer_ohbong.pt"
CONFIG_PATH = MODELS_DIR / "minrisk_model_config.json"
X_SCALER_PATH = MODELS_DIR / "minrisk_x_scaler.joblib"
SCENARIO_SCALER_PATH = MODELS_DIR / "minrisk_scenario_scaler.joblib"
Y_SCALER_PATH = MODELS_DIR / "minrisk_y_scaler.joblib"
RESIDUAL_STATS_PATH = MODELS_DIR / "minrisk_residual_stats.json"
PREDICTIONS_PATH = MODELS_DIR / "minrisk_test_predictions.csv"
METRICS_PATH = MODELS_DIR / "minrisk_metrics.csv"
METRICS_BY_RESERVOIR_PATH = MODELS_DIR / "minrisk_metrics_by_reservoir.csv"
OHBONG_AUGUST_PATH = MODELS_DIR / "minrisk_ohbong_2025_diagnosis.csv"

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


def create_minrisk_sequences(
    df: pd.DataFrame,
    input_window: int = 60,
    forecast_horizon: int = 30,
    target_col: str = "저수율_5일EMA",
    danger_threshold: float = 40,
    feature_cols: Optional[list[str]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, list[str], list[str]]:
    selected_features = feature_cols or DEFAULT_FEATURE_COLUMNS
    required = [
        "저수지명", "날짜", target_col, "강수량", "평균기온(℃)", "SPI3", "SPI6",
        *selected_features,
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"min-risk sequence 생성에 필요한 컬럼이 없습니다: {missing}")

    X_seq_values, X_scenario_values, y_delta_values, y_risk_values, meta_rows = [], [], [], [], []
    sorted_df = df.sort_values(["저수지명", "날짜"]).reset_index(drop=True)

    for reservoir_name, group in sorted_df.groupby("저수지명", sort=False):
        group = group.sort_values("날짜").reset_index(drop=True)
        features = group[selected_features].to_numpy(dtype=np.float32)
        target = group[target_col].to_numpy(dtype=np.float32)
        rainfall = group["강수량"].to_numpy(dtype=np.float32)
        temp = group["평균기온(℃)"].to_numpy(dtype=np.float32)
        spi3 = group["SPI3"].to_numpy(dtype=np.float32)
        spi6 = group["SPI6"].to_numpy(dtype=np.float32)
        dates = pd.to_datetime(group["날짜"])

        max_start = len(group) - input_window - forecast_horizon + 1
        if max_start <= 0:
            continue

        for start_idx in range(max_start):
            end_idx = start_idx + input_window
            future_start = end_idx
            future_end = end_idx + forecast_horizon
            current_idx = end_idx - 1

            future_target = target[future_start:future_end]
            min_offset = int(np.argmin(future_target))
            current_rate = float(target[current_idx])
            future_min_rate = float(future_target[min_offset])
            future_end_rate = float(future_target[-1])
            min_delta = future_min_rate - current_rate
            risk = float(future_min_rate < danger_threshold)
            target_date = dates.iloc[future_end - 1]

            scenario = [
                float(rainfall[future_start:future_end].sum()),
                float(rainfall[future_start:future_end].mean()),
                float((rainfall[future_start:future_end] > 0).sum()),
                float((rainfall[future_start:future_end] <= 0).sum()),
                float(temp[future_start:future_end].mean()),
                float(temp[future_start:future_end].max()),
                float(spi3[future_end - 1] - spi3[current_idx]),
                float(spi6[future_end - 1] - spi6[current_idx]),
                float(spi3[future_start:future_end].min()),
                float(spi6[future_start:future_end].min()),
                float(np.sin(2 * np.pi * target_date.month / 12)),
                float(np.cos(2 * np.pi * target_date.month / 12)),
                float(np.sin(2 * np.pi * target_date.dayofyear / 365.25)),
                float(np.cos(2 * np.pi * target_date.dayofyear / 365.25)),
                current_rate,
            ]

            X_seq_values.append(features[start_idx:end_idx])
            X_scenario_values.append(scenario)
            y_delta_values.append(min_delta)
            y_risk_values.append(risk)
            meta_rows.append(
                {
                    "저수지명": reservoir_name,
                    "기준날짜": dates.iloc[current_idx],
                    "예측시작날짜": dates.iloc[future_start],
                    "예측종료날짜": dates.iloc[future_end - 1],
                    "current_rate": current_rate,
                    "future_min_rate": future_min_rate,
                    "future_end_rate": future_end_rate,
                    "min_delta": min_delta,
                    "risk_within_30d": bool(risk),
                    "days_to_min": min_offset + 1,
                    "min_rate_date": dates.iloc[future_start + min_offset],
                }
            )

    if not X_seq_values:
        raise ValueError("생성된 min-risk sequence가 없습니다. window와 horizon을 확인하세요.")

    return (
        np.stack(X_seq_values).astype(np.float32),
        np.asarray(X_scenario_values, dtype=np.float32),
        np.asarray(y_delta_values, dtype=np.float32),
        np.asarray(y_risk_values, dtype=np.float32),
        pd.DataFrame(meta_rows),
        selected_features,
        SCENARIO_COLUMNS,
    )


def split_by_date_minrisk(X_seq, X_scenario, y_min_delta, y_risk, metadata):
    dates = pd.to_datetime(metadata["예측종료날짜"])
    train_mask = dates <= pd.Timestamp("2023-12-31")
    val_mask = (dates >= pd.Timestamp("2024-01-01")) & (dates <= pd.Timestamp("2024-12-31"))
    test_mask = (dates >= pd.Timestamp("2025-01-01")) & (dates <= pd.Timestamp("2025-12-31"))
    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        raise ValueError("train/validation/test 중 비어 있는 분할이 있습니다.")

    def take(mask):
        arr_mask = mask.to_numpy()
        return (
            X_seq[arr_mask],
            X_scenario[arr_mask],
            y_min_delta[arr_mask],
            y_risk[arr_mask],
            metadata.loc[mask].reset_index(drop=True),
        )

    return (*take(train_mask), *take(val_mask), *take(test_mask))


def _scale_3d(scaler, X, fit=False):
    shape = X.shape
    X_2d = X.reshape(-1, shape[-1])
    scaled = scaler.fit_transform(X_2d) if fit else scaler.transform(X_2d)
    return scaled.reshape(shape).astype(np.float32)


def scale_minrisk_data(X_train, S_train, y_train, X_val, S_val, y_val, X_test, S_test, y_test):
    x_scaler, scenario_scaler, y_scaler = StandardScaler(), StandardScaler(), StandardScaler()
    X_train_s = _scale_3d(x_scaler, X_train, fit=True)
    X_val_s = _scale_3d(x_scaler, X_val)
    X_test_s = _scale_3d(x_scaler, X_test)
    S_train_s = scenario_scaler.fit_transform(S_train).astype(np.float32)
    S_val_s = scenario_scaler.transform(S_val).astype(np.float32)
    S_test_s = scenario_scaler.transform(S_test).astype(np.float32)
    y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).astype(np.float32)
    y_val_s = y_scaler.transform(y_val.reshape(-1, 1)).astype(np.float32)
    y_test_s = y_scaler.transform(y_test.reshape(-1, 1)).astype(np.float32)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(x_scaler, X_SCALER_PATH)
    joblib.dump(scenario_scaler, SCENARIO_SCALER_PATH)
    joblib.dump(y_scaler, Y_SCALER_PATH)
    return X_train_s, S_train_s, y_train_s, X_val_s, S_val_s, y_val_s, X_test_s, S_test_s, y_test_s, y_scaler


class MinRiskReservoirDataset(Dataset):
    def __init__(self, X_seq, X_scenario, y_delta_scaled, y_risk):
        self.X_seq = torch.as_tensor(X_seq, dtype=torch.float32)
        self.X_scenario = torch.as_tensor(X_scenario, dtype=torch.float32)
        self.y_delta = torch.as_tensor(y_delta_scaled, dtype=torch.float32).reshape(-1, 1)
        self.y_risk = torch.as_tensor(y_risk, dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.X_seq)

    def __getitem__(self, idx):
        return self.X_seq[idx], self.X_scenario[idx], self.y_delta[idx], self.y_risk[idx]


class ScenarioMinRiskTransformer(nn.Module):
    def __init__(
        self,
        num_seq_features,
        num_scenario_features,
        input_window=60,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.2,
        scenario_hidden=32,
        fusion_hidden=64,
    ):
        super().__init__()
        self.seq_projection = nn.Linear(num_seq_features, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=input_window, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.scenario_branch = nn.Sequential(
            nn.Linear(num_scenario_features, scenario_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(scenario_hidden, scenario_hidden),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(d_model + scenario_hidden, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.regression_head = nn.Linear(fusion_hidden, 1)
        self.classification_head = nn.Linear(fusion_hidden, 1)

    def forward(self, x_seq, x_scenario):
        seq = self.seq_projection(x_seq)
        seq = self.positional_encoding(seq)
        seq_hidden = self.transformer_encoder(seq)[:, -1, :]
        scenario_hidden = self.scenario_branch(x_scenario)
        fused = self.fusion(torch.cat([seq_hidden, scenario_hidden], dim=1))
        return self.regression_head(fused), self.classification_head(fused)


def calculate_regression_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)
    residual = y_true - y_pred
    denom = np.where(np.abs(y_true) < 1e-6, np.nan, np.abs(y_true))
    mape = np.abs(residual) / denom
    return {
        "MAE": float(np.mean(np.abs(residual))),
        "RMSE": float(np.sqrt(np.mean(residual**2))),
        "MAPE": float(np.nanmean(mape) * 100) if not np.all(np.isnan(mape)) else 0.0,
    }


def calculate_classification_metrics(y_true, y_prob, y_pred):
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / max(len(y_true), 1)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "TP": tp, "FP": fp, "TN": tn, "FN": fn}


def _run_epoch(model, loader, reg_loss_fn, cls_loss_fn, optimizer, device, classification_weight=0.05):
    model.train()
    total = reg_total = cls_total = samples = 0.0
    for x_seq, x_scenario, y_delta, y_risk in loader:
        x_seq, x_scenario = x_seq.to(device), x_scenario.to(device)
        y_delta, y_risk = y_delta.to(device), y_risk.to(device)
        optimizer.zero_grad()
        pred_delta, risk_logit = model(x_seq, x_scenario)
        reg_loss = reg_loss_fn(pred_delta, y_delta)
        cls_loss = cls_loss_fn(risk_logit, y_risk)
        loss = reg_loss + classification_weight * cls_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        batch = x_seq.size(0)
        total += loss.item() * batch
        reg_total += reg_loss.item() * batch
        cls_total += cls_loss.item() * batch
        samples += batch
    return total / samples, reg_total / samples, cls_total / samples


def _eval_loss(model, loader, reg_loss_fn, cls_loss_fn, device, classification_weight=0.05):
    model.eval()
    total = reg_total = cls_total = samples = 0.0
    with torch.no_grad():
        for x_seq, x_scenario, y_delta, y_risk in loader:
            x_seq, x_scenario = x_seq.to(device), x_scenario.to(device)
            y_delta, y_risk = y_delta.to(device), y_risk.to(device)
            pred_delta, risk_logit = model(x_seq, x_scenario)
            reg_loss = reg_loss_fn(pred_delta, y_delta)
            cls_loss = cls_loss_fn(risk_logit, y_risk)
            loss = reg_loss + classification_weight * cls_loss
            batch = x_seq.size(0)
            total += loss.item() * batch
            reg_total += reg_loss.item() * batch
            cls_total += cls_loss.item() * batch
            samples += batch
    return total / samples, reg_total / samples, cls_total / samples


def _predict(model, X_seq, X_scenario, y_scaler, metadata, device, danger_threshold=40, batch_size=256):
    model.eval()
    loader = DataLoader(
        torch.utils.data.TensorDataset(
            torch.as_tensor(X_seq, dtype=torch.float32),
            torch.as_tensor(X_scenario, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    deltas_s, logits = [], []
    with torch.no_grad():
        for x_seq, x_scenario in loader:
            pred_delta, risk_logit = model(x_seq.to(device), x_scenario.to(device))
            deltas_s.append(pred_delta.cpu().numpy().reshape(-1))
            logits.append(risk_logit.cpu().numpy().reshape(-1))
    pred_delta_scaled = np.concatenate(deltas_s)
    risk_logit = np.concatenate(logits)
    pred_delta = y_scaler.inverse_transform(pred_delta_scaled.reshape(-1, 1)).reshape(-1)
    current = metadata["current_rate"].to_numpy(dtype=np.float64)
    pred_min = np.clip(current + pred_delta, 0, 100)
    risk_prob = 1 / (1 + np.exp(-risk_logit))
    result = pd.DataFrame({
        "저수지명": metadata["저수지명"].to_numpy(),
        "기준날짜": pd.to_datetime(metadata["기준날짜"]).dt.date.astype(str),
        "예측시작날짜": pd.to_datetime(metadata["예측시작날짜"]).dt.date.astype(str),
        "예측종료날짜": pd.to_datetime(metadata["예측종료날짜"]).dt.date.astype(str),
        "current_rate": current,
        "future_min_rate": metadata["future_min_rate"].to_numpy(dtype=np.float64),
        "future_end_rate": metadata["future_end_rate"].to_numpy(dtype=np.float64),
        "predicted_min_rate": pred_min,
        "predicted_min_delta": pred_delta,
        "risk_within_30d": metadata["risk_within_30d"].astype(bool).to_numpy(),
        "risk_probability": risk_prob,
        "risk_pred": risk_prob >= 0.5,
        "risk_pred_by_threshold": pred_min < danger_threshold,
        "days_to_min": metadata["days_to_min"].to_numpy(),
        "min_rate_date": pd.to_datetime(metadata["min_rate_date"]).dt.date.astype(str),
    })
    result["residual"] = result["future_min_rate"] - result["predicted_min_rate"]
    result["abs_error"] = result["residual"].abs()
    return result


def _scope_metrics(scope, df):
    reg = calculate_regression_metrics(df["future_min_rate"], df["predicted_min_rate"])
    cls_head = calculate_classification_metrics(df["risk_within_30d"], df["risk_probability"], df["risk_pred"])
    cls_threshold = calculate_classification_metrics(
        df["risk_within_30d"],
        df["risk_probability"],
        df["risk_pred_by_threshold"],
    )
    return {
        "scope": scope,
        **reg,
        **{f"risk_head_{k}": v for k, v in cls_head.items()},
        **{f"risk_threshold_{k}": v for k, v in cls_threshold.items()},
        "n_samples": int(len(df)),
    }


def _save_json(path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def train_minrisk_model(
    source="auto",
    input_window=60,
    forecast_horizon=30,
    target_col="저수율_5일EMA",
    danger_threshold=40,
    warning_threshold=60,
    batch_size=64,
    epochs=100,
    learning_rate=0.0001,
    weight_decay=1e-4,
    patience=15,
    classification_weight=0.05,
    d_model=64,
    nhead=4,
    num_layers=2,
    dim_feedforward=128,
    dropout=0.2,
    scenario_hidden=32,
    fusion_hidden=64,
):
    df = add_time_features(clean_master_dataset(load_master_dataset(source=source)))
    X_seq, X_scenario, y_delta, y_risk, metadata, feature_cols, scenario_cols = create_minrisk_sequences(
        df, input_window, forecast_horizon, target_col, danger_threshold
    )
    split = split_by_date_minrisk(X_seq, X_scenario, y_delta, y_risk, metadata)
    Xtr, Str, ytr, rtr, mtr, Xv, Sv, yv, rv, mv, Xte, Ste, yte, rte, mte = split
    Xtr_s, Str_s, ytr_s, Xv_s, Sv_s, yv_s, Xte_s, Ste_s, _, y_scaler = scale_minrisk_data(
        Xtr, Str, ytr, Xv, Sv, yv, Xte, Ste, yte
    )

    train_loader = DataLoader(MinRiskReservoirDataset(Xtr_s, Str_s, ytr_s, rtr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(MinRiskReservoirDataset(Xv_s, Sv_s, yv_s, rv), batch_size=batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 device: {device}")

    model = ScenarioMinRiskTransformer(
        len(feature_cols), len(scenario_cols), input_window, d_model, nhead, num_layers,
        dim_feedforward, dropout, scenario_hidden, fusion_hidden
    ).to(device)
    pos = float(rtr.sum())
    neg = float(len(rtr) - rtr.sum())
    raw_pos_weight = neg / pos if pos > 0 else 1.0
    clipped_pos_weight = min(raw_pos_weight, 10.0)
    pos_weight = torch.tensor([clipped_pos_weight], dtype=torch.float32, device=device)
    reg_loss_fn = nn.HuberLoss()
    cls_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )

    best_reg_loss, best_epoch, stale = float("inf"), 0, 0
    for epoch in range(1, epochs + 1):
        train_loss, _, _ = _run_epoch(model, train_loader, reg_loss_fn, cls_loss_fn, optimizer, device, classification_weight)
        val_loss, val_reg, val_cls = _eval_loss(model, val_loader, reg_loss_fn, cls_loss_fn, device, classification_weight)
        scheduler.step(val_reg)
        current_lr = optimizer.param_groups[0]["lr"]
        is_best = val_reg < best_reg_loss
        if is_best:
            best_reg_loss, best_epoch, stale = val_reg, epoch, 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            stale += 1
        print(f"Epoch {epoch:03d}/{epochs} | train total loss: {train_loss:.6f} | validation total loss: {val_loss:.6f} | validation regression loss: {val_reg:.6f} | validation classification loss: {val_cls:.6f} | lr: {current_lr:.8f} {'BEST' if is_best else ''}")
        if stale >= patience:
            print(f"Early stopping: {patience} epoch 동안 개선이 없어 중단합니다.")
            break

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    pred_df = _predict(model, Xte_s, Ste_s, y_scaler, mte, device, danger_threshold=danger_threshold)
    pred_df.to_csv(PREDICTIONS_PATH, index=False, encoding="utf-8-sig")

    ohbong_df = pred_df[pred_df["저수지명"] == "오봉"].copy()
    august_df = pred_df[
        (pred_df["저수지명"] == "오봉")
        & (
            pd.to_datetime(pred_df["예측종료날짜"]).dt.strftime("%Y-%m").eq("2025-08")
            | pd.to_datetime(pred_df["min_rate_date"]).dt.strftime("%Y-%m").eq("2025-08")
        )
    ].copy()
    august_df.to_csv(OHBONG_AUGUST_PATH, index=False, encoding="utf-8-sig")

    rows = [_scope_metrics("all_reservoirs", pred_df)]
    if not ohbong_df.empty:
        rows.append(_scope_metrics("ohbong_only", ohbong_df))
    if not august_df.empty:
        rows.append(_scope_metrics("ohbong_august_2025", august_df))
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(METRICS_PATH, index=False, encoding="utf-8-sig")

    by_rows = []
    for name, group in pred_df.groupby("저수지명"):
        row = _scope_metrics(name, group)
        row["저수지명"] = name
        by_rows.append(row)
    by_reservoir = pd.DataFrame(by_rows).rename(columns={"risk_accuracy": "risk_accuracy", "risk_precision": "risk_precision", "risk_recall": "risk_recall", "risk_f1": "risk_f1"})
    by_reservoir = by_reservoir[["저수지명", "MAE", "RMSE", "MAPE", "risk_head_accuracy", "risk_head_precision", "risk_head_recall", "risk_head_f1", "risk_head_TP", "risk_head_FP", "risk_head_TN", "risk_head_FN", "risk_threshold_accuracy", "risk_threshold_precision", "risk_threshold_recall", "risk_threshold_f1", "risk_threshold_TP", "risk_threshold_FP", "risk_threshold_TN", "risk_threshold_FN", "n_samples"]]
    by_reservoir.to_csv(METRICS_BY_RESERVOIR_PATH, index=False, encoding="utf-8-sig")

    config = {
        "model_type": "scenario_minrisk_transformer",
        "input_window": input_window,
        "forecast_horizon": forecast_horizon,
        "target_col": target_col,
        "danger_threshold": danger_threshold,
        "warning_threshold": warning_threshold,
        "feature_cols": feature_cols,
        "scenario_cols": scenario_cols,
        "num_seq_features": len(feature_cols),
        "num_scenario_features": len(scenario_cols),
        "d_model": d_model,
        "nhead": nhead,
        "num_layers": num_layers,
        "dim_feedforward": dim_feedforward,
        "dropout": dropout,
        "scenario_hidden": scenario_hidden,
        "fusion_hidden": fusion_hidden,
        "classification_weight": classification_weight,
        "train_samples": int(len(Xtr)),
        "val_samples": int(len(Xv)),
        "test_samples": int(len(Xte)),
    }
    _save_json(CONFIG_PATH, config)

    all_reg = calculate_regression_metrics(pred_df["future_min_rate"], pred_df["predicted_min_rate"])
    oh_reg = calculate_regression_metrics(ohbong_df["future_min_rate"], ohbong_df["predicted_min_rate"]) if not ohbong_df.empty else {"MAE": None, "RMSE": None, "MAPE": None}
    oh_cls_head = calculate_classification_metrics(ohbong_df["risk_within_30d"], ohbong_df["risk_probability"], ohbong_df["risk_pred"]) if not ohbong_df.empty else {"recall": None, "f1": None}
    oh_cls_threshold = calculate_classification_metrics(ohbong_df["risk_within_30d"], ohbong_df["risk_probability"], ohbong_df["risk_pred_by_threshold"]) if not ohbong_df.empty else {"recall": None, "f1": None}
    residuals = pred_df["residual"].to_numpy()
    residual_stats = {
        "mae": all_reg["MAE"],
        "rmse": all_reg["RMSE"],
        "mape": all_reg["MAPE"],
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals, ddof=1)),
        "ohbong_mae": oh_reg["MAE"],
        "ohbong_rmse": oh_reg["RMSE"],
        "ohbong_mape": oh_reg["MAPE"],
        "ohbong_risk_recall": oh_cls_threshold["recall"],
        "ohbong_risk_f1": oh_cls_threshold["f1"],
        "ohbong_risk_head_recall": oh_cls_head["recall"],
        "ohbong_risk_head_f1": oh_cls_head["f1"],
        "ohbong_risk_threshold_recall": oh_cls_threshold["recall"],
        "ohbong_risk_threshold_f1": oh_cls_threshold["f1"],
    }
    _save_json(RESIDUAL_STATS_PATH, residual_stats)

    _print_report(metrics_df, by_reservoir, august_df, residual_stats, best_epoch, best_reg_loss)
    return pred_df, metrics_df, by_reservoir


def _print_report(metrics_df, by_reservoir, august_df, residual_stats, best_epoch, best_loss):
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    print("\n=== Scenario-aware Min-Risk Transformer 학습 완료 ===")
    print(f"best epoch: {best_epoch}")
    print(f"best validation regression loss: {best_loss:.6f}")
    print("\n=== 전체/오봉 metrics ===")
    print(metrics_df.to_string(index=False))
    print("\n=== 오봉 위험 탐지 recall 비교 ===")
    print(f"오봉 risk recall by classification head: {residual_stats['ohbong_risk_head_recall']}")
    print(f"오봉 risk recall by predicted_min_rate threshold: {residual_stats['ohbong_risk_threshold_recall']}")
    print("\n=== 저수지별 성능 상위 5개 ===")
    print(by_reservoir.sort_values("MAE").head(5).to_string(index=False))
    print("\n=== 저수지별 성능 하위 5개 ===")
    print(by_reservoir.sort_values("MAE", ascending=False).head(5).to_string(index=False))
    if not august_df.empty:
        reg = calculate_regression_metrics(august_df["future_min_rate"], august_df["predicted_min_rate"])
        cls_head = calculate_classification_metrics(august_df["risk_within_30d"], august_df["risk_probability"], august_df["risk_pred"])
        cls_threshold = calculate_classification_metrics(august_df["risk_within_30d"], august_df["risk_probability"], august_df["risk_pred_by_threshold"])
        print("\n=== 2025년 8월 오봉 구간 진단 ===")
        print(f"sample 수: {len(august_df)}")
        print(f"실제 future_min_rate 평균/최소: {august_df['future_min_rate'].mean():.4f} / {august_df['future_min_rate'].min():.4f}")
        print(f"predicted_min_rate 평균/최소: {august_df['predicted_min_rate'].mean():.4f} / {august_df['predicted_min_rate'].min():.4f}")
        print(f"MAE: {reg['MAE']:.4f}")
        print(f"RMSE: {reg['RMSE']:.4f}")
        print(f"위험 recall by classification head: {cls_head['recall']:.4f}")
        print(f"위험 recall by predicted_min_rate threshold: {cls_threshold['recall']:.4f}")
        print(f"위험 FN 개수 by classification head: {cls_head['FN']}")
        print(f"위험 FN 개수 by predicted_min_rate threshold: {cls_threshold['FN']}")
        if cls_threshold["FN"] > 0:
            print("위험 상황을 놓치는 FN이 있어 threshold 조정이나 위험 sample 가중치 조정이 필요합니다.")
    print("\n기존 30일 뒤 예측 모델은 '정확히 30일 뒤 저수율' 기준이라, 이번 모델의 '향후 30일 최저 저수율'과 직접 비교 대상이 아닙니다.")
    print("대신 프로젝트 목적상 이번 모델은 위험 탐지에 더 적합합니다.")
    print("이 모델은 정확히 30일 뒤 저수율이 아니라 향후 30일 동안의 최저 저수율과 위험 발생 여부를 예측합니다.")
    print("따라서 2025년 8월 오봉 저수지 물 부족 사태 예방 목적에는 기존 30일 뒤 저수율 예측보다 더 직접적으로 부합합니다.")
    recall = residual_stats["ohbong_risk_recall"]
    if recall is not None and recall >= 0.7:
        print("오봉 위험 탐지 recall이 높아 위험 사전 경보 모델로 활용 가능성이 있습니다.")
    else:
        print("위험 구간을 놓치는 경우가 있으므로 위험 sample 가중치 조정 또는 threshold 조정이 필요합니다.")
    print(f"\n저장된 모델 경로: {MODEL_PATH}")
    print(f"저장된 예측 경로: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    try:
        train_minrisk_model()
    except Exception as error:
        print("Scenario-aware Min-Risk Transformer 학습에 실패했습니다.")
        print(error)
