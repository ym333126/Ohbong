import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.db.supabase_client import load_master_dataset  # noqa: E402


TARGET_COLUMN = "저수율_5일EMA"

BASE_COLUMNS = [
    "저수지명",
    "날짜",
    "저수율",
    "저수율_5일EMA",
    "평균기온(℃)",
    "기온_7일평균",
    "강수량",
    "강수량_3일누적",
    "강수량_7일누적",
    "SPI3",
    "SPI6",
    "위도",
    "경도",
    "유효저수량(천m3)",
]

NUMERIC_COLUMNS = [
    "저수율",
    "저수율_5일EMA",
    "평균기온(℃)",
    "기온_7일평균",
    "강수량",
    "강수량_3일누적",
    "강수량_7일누적",
    "SPI3",
    "SPI6",
    "위도",
    "경도",
    "유효저수량(천m3)",
]

DEFAULT_FEATURE_COLUMNS = [
    "저수율",
    "저수율_5일EMA",
    "평균기온(℃)",
    "기온_7일평균",
    "강수량",
    "강수량_3일누적",
    "강수량_7일누적",
    "SPI3",
    "SPI6",
    "위도",
    "경도",
    "유효저수량(천m3)",
    "month_sin",
    "month_cos",
    "season_sin",
    "season_cos",
]


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            "전처리에 필요한 컬럼이 누락되었습니다. "
            f"누락 컬럼: {missing_columns}. "
            "한글 컬럼명을 임의로 변경하지 말고 원본 그대로 유지하세요."
        )


def clean_master_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean master_dataset while preserving original Korean column names."""
    _require_columns(df, BASE_COLUMNS)

    cleaned = df.copy()
    cleaned["날짜"] = pd.to_datetime(cleaned["날짜"], errors="coerce")

    if cleaned["날짜"].isna().any():
        invalid_count = int(cleaned["날짜"].isna().sum())
        raise ValueError(
            f"날짜 컬럼을 datetime으로 변환하지 못한 행이 {invalid_count}개 있습니다. "
            "원본 날짜 값을 확인하세요."
        )

    cleaned = cleaned.sort_values(["저수지명", "날짜"]).reset_index(drop=True)

    for column in NUMERIC_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned[NUMERIC_COLUMNS] = cleaned.groupby("저수지명", group_keys=False)[
        NUMERIC_COLUMNS
    ].apply(lambda group: group.ffill().bfill())

    medians = cleaned[NUMERIC_COLUMNS].median(numeric_only=True)
    cleaned[NUMERIC_COLUMNS] = cleaned[NUMERIC_COLUMNS].fillna(medians)
    cleaned[NUMERIC_COLUMNS] = cleaned[NUMERIC_COLUMNS].fillna(0)

    return cleaned


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add date-derived cyclic features for Transformer inputs."""
    _require_columns(df, ["날짜"])

    featured = df.copy()
    featured["날짜"] = pd.to_datetime(featured["날짜"], errors="coerce")

    if featured["날짜"].isna().any():
        invalid_count = int(featured["날짜"].isna().sum())
        raise ValueError(
            f"시간 피처 생성 중 날짜 변환 실패 행이 {invalid_count}개 있습니다."
        )

    featured["month"] = featured["날짜"].dt.month
    featured["dayofyear"] = featured["날짜"].dt.dayofyear
    featured["month_sin"] = np.sin(2 * np.pi * featured["month"] / 12)
    featured["month_cos"] = np.cos(2 * np.pi * featured["month"] / 12)
    featured["season_sin"] = np.sin(2 * np.pi * featured["dayofyear"] / 365.25)
    featured["season_cos"] = np.cos(2 * np.pi * featured["dayofyear"] / 365.25)

    return featured


def create_sequences(
    df: pd.DataFrame,
    input_window: int = 60,
    forecast_horizon: int = 30,
    target_col: str = TARGET_COLUMN,
    feature_cols: Optional[list[str]] = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str]]:
    """Create reservoir-wise Transformer sequences."""
    if input_window <= 0:
        raise ValueError("input_window는 1 이상의 정수여야 합니다.")
    if forecast_horizon <= 0:
        raise ValueError("forecast_horizon은 1 이상의 정수여야 합니다.")

    selected_features = feature_cols or DEFAULT_FEATURE_COLUMNS
    required_columns = ["저수지명", "날짜", target_col, *selected_features]
    _require_columns(df, required_columns)

    X_values = []
    y_values = []
    metadata_rows = []

    sorted_df = df.sort_values(["저수지명", "날짜"]).reset_index(drop=True)

    for reservoir_name, group in sorted_df.groupby("저수지명", sort=False):
        group = group.sort_values("날짜").reset_index(drop=True)
        feature_array = group[selected_features].to_numpy(dtype=np.float32)
        target_array = group[target_col].to_numpy(dtype=np.float32)
        date_array = group["날짜"].to_numpy()

        max_start = len(group) - input_window - forecast_horizon + 1
        if max_start <= 0:
            continue

        for start_idx in range(max_start):
            end_idx = start_idx + input_window
            target_idx = end_idx + forecast_horizon - 1

            X_values.append(feature_array[start_idx:end_idx])
            y_values.append(target_array[target_idx])
            metadata_rows.append(
                {
                    "저수지명": reservoir_name,
                    "기준날짜": pd.Timestamp(date_array[end_idx - 1]),
                    "예측대상날짜": pd.Timestamp(date_array[target_idx]),
                }
            )

    if not X_values:
        raise ValueError(
            "생성된 sequence가 없습니다. "
            "데이터 길이, input_window, forecast_horizon 값을 확인하세요."
        )

    X = np.stack(X_values).astype(np.float32)
    y = np.asarray(y_values, dtype=np.float32)
    metadata = pd.DataFrame(metadata_rows)

    return X, y, metadata, selected_features


def split_sequences_by_date(
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    train_end_date: str = "2024-12-31",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Split sequences by prediction target date."""
    _require_columns(metadata, ["예측대상날짜"])

    target_dates = pd.to_datetime(metadata["예측대상날짜"], errors="coerce")
    if target_dates.isna().any():
        raise ValueError("metadata의 예측대상날짜를 datetime으로 변환할 수 없습니다.")

    train_end = pd.Timestamp(train_end_date)
    train_mask = target_dates <= train_end
    test_mask = target_dates > train_end

    if not train_mask.any():
        raise ValueError("train 데이터가 없습니다. train_end_date를 확인하세요.")
    if not test_mask.any():
        raise ValueError("test 데이터가 없습니다. train_end_date를 확인하세요.")

    return (
        X[train_mask.to_numpy()],
        X[test_mask.to_numpy()],
        y[train_mask.to_numpy()],
        y[test_mask.to_numpy()],
        metadata.loc[train_mask].reset_index(drop=True),
        metadata.loc[test_mask].reset_index(drop=True),
    )


def scale_sequences(
    X_train: np.ndarray,
    X_test: np.ndarray,
    scaler_path: Optional[Path | str] = None,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Fit StandardScaler on train sequences only and transform train/test."""
    if X_train.ndim != 3 or X_test.ndim != 3:
        raise ValueError("X_train과 X_test는 3차원 배열이어야 합니다.")

    scaler = StandardScaler()
    train_shape = X_train.shape
    test_shape = X_test.shape

    X_train_2d = X_train.reshape(-1, train_shape[-1])
    X_test_2d = X_test.reshape(-1, test_shape[-1])

    X_train_scaled = scaler.fit_transform(X_train_2d).reshape(train_shape)
    X_test_scaled = scaler.transform(X_test_2d).reshape(test_shape)

    selected_scaler_path = Path(scaler_path) if scaler_path else PROJECT_ROOT / "models" / "scaler.joblib"
    selected_scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, selected_scaler_path)

    return (
        X_train_scaled.astype(np.float32),
        X_test_scaled.astype(np.float32),
        scaler,
    )


def prepare_train_test_data(
    source: str = "auto",
    input_window: int = 60,
    forecast_horizon: int = 30,
    target_col: str = TARGET_COLUMN,
) -> dict:
    """Run the full data preparation pipeline for Transformer training."""
    raw_df = load_master_dataset(source=source)
    cleaned_df = clean_master_dataset(raw_df)
    featured_df = add_time_features(cleaned_df)
    X, y, metadata, feature_cols = create_sequences(
        featured_df,
        input_window=input_window,
        forecast_horizon=forecast_horizon,
        target_col=target_col,
    )
    X_train, X_test, y_train, y_test, meta_train, meta_test = split_sequences_by_date(
        X, y, metadata
    )
    X_train_scaled, X_test_scaled, scaler = scale_sequences(X_train, X_test)

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "meta_train": meta_train,
        "meta_test": meta_test,
        "feature_cols": feature_cols,
        "scaler": scaler,
    }


def _format_date_range(metadata: pd.DataFrame) -> str:
    dates = pd.to_datetime(metadata["예측대상날짜"])
    return f"{dates.min().date()} ~ {dates.max().date()}"


if __name__ == "__main__":
    try:
        raw_dataset = load_master_dataset(source="auto")
        cleaned_dataset = clean_master_dataset(raw_dataset)
        featured_dataset = add_time_features(cleaned_dataset)
        X_all, y_all, meta_all, features = create_sequences(featured_dataset)
        X_train_raw, X_test_raw, y_train, y_test, meta_train, meta_test = (
            split_sequences_by_date(X_all, y_all, meta_all)
        )
        X_train, X_test, _ = scale_sequences(X_train_raw, X_test_raw)

        ohbong_sample_count = int((meta_all["저수지명"] == "오봉").sum())

        print("=== Ohbong Transformer 전처리 테스트 ===")
        print(f"원본 데이터 shape: {raw_dataset.shape}")
        print(f"정리 후 데이터 shape: {featured_dataset.shape}")
        print(f"생성된 sequence X shape: {X_all.shape}")
        print(f"생성된 y shape: {y_all.shape}")
        print(f"train sample 수: {len(X_train):,}")
        print(f"test sample 수: {len(X_test):,}")
        print(f"feature 개수: {len(features)}")
        print(f"feature 목록: {features}")
        print(f"train 날짜 범위: {_format_date_range(meta_train)}")
        print(f"test 날짜 범위: {_format_date_range(meta_test)}")
        print(f"오봉 샘플 개수: {ohbong_sample_count:,}")
    except Exception as error:
        print("전처리 테스트에 실패했습니다.")
        print(error)
