from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from pulp import LpMinimize, LpProblem, LpStatus, LpVariable, PULP_CBC_CMD, lpSum, value


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.db.supabase_client import load_master_dataset, load_reservoir_distances_from_supabase  # noqa: E402
from src.utils.risk_policy import classify_reservoir_stage, is_supply_allowed  # noqa: E402


OBONG_NAME = "오봉"
DEFAULT_TARGET_RATE = 50.0
DEFAULT_SAFETY_RATE = 40.0
DEFAULT_MAX_DISTANCE_KM = 100.0

PENALTY_UNMET = 1_000_000.0
PENALTY_OUT_OF_REGION = 50.0

DEFAULT_DISTANCE_CSV_PATH = PROJECT_ROOT / "data" / "reservoir_distances.csv"


def validate_distance_data(distance_df: pd.DataFrame) -> None:
    required = ["source_reservoir", "target_reservoir", "distance_km"]
    missing = [column for column in required if column not in distance_df.columns]
    if missing:
        raise ValueError(f"거리 데이터 표준 컬럼이 누락되었습니다: {missing}")


def normalize_distance_columns(distance_df: pd.DataFrame) -> pd.DataFrame:
    df = distance_df.copy()
    if {"reservoir_from", "reservoir_to", "distance"}.issubset(df.columns):
        df["source_reservoir"] = df["reservoir_from"]
        df["target_reservoir"] = df["reservoir_to"]
        df["distance_km"] = pd.to_numeric(df["distance"], errors="coerce")
    elif {"source_reservoir", "target_reservoir", "distance_km"}.issubset(df.columns):
        df["distance_km"] = pd.to_numeric(df["distance_km"], errors="coerce")
    else:
        raise ValueError("거리 데이터는 reservoir_from/reservoir_to/distance 또는 source_reservoir/target_reservoir/distance_km 컬럼이 필요합니다.")
    validate_distance_data(df)
    return df.dropna(subset=["source_reservoir", "target_reservoir", "distance_km"])


def load_distance_data(source: str = "supabase", csv_path: Optional[Path | str] = None) -> pd.DataFrame:
    normalized = source.lower().strip()
    if normalized == "supabase":
        distance_df = load_reservoir_distances_from_supabase()
        print("Supabase reservoir_distances 거리 테이블 사용 중")
        return distance_df
    if normalized == "csv":
        selected_path = Path(csv_path) if csv_path else DEFAULT_DISTANCE_CSV_PATH
        if not selected_path.exists():
            raise FileNotFoundError(f"거리 CSV 파일을 찾을 수 없습니다: {selected_path}")
        return normalize_distance_columns(pd.read_csv(selected_path))
    if normalized == "auto":
        try:
            return load_distance_data(source="supabase")
        except Exception as supabase_error:
            print(f"[안내] Supabase 거리 테이블 로딩 실패. CSV fallback 시도: {supabase_error}")
            return load_distance_data(source="csv", csv_path=csv_path)
    raise ValueError("source는 'supabase', 'csv', 'auto' 중 하나여야 합니다.")


def get_distance_between_reservoirs(source_name: str, target_name: str, distance_df: pd.DataFrame) -> Optional[float]:
    validate_distance_data(distance_df)
    direct = distance_df[
        (distance_df["source_reservoir"] == source_name)
        & (distance_df["target_reservoir"] == target_name)
    ]
    if not direct.empty:
        return float(direct["distance_km"].iloc[0])
    reverse = distance_df[
        (distance_df["source_reservoir"] == target_name)
        & (distance_df["target_reservoir"] == source_name)
    ]
    if not reverse.empty:
        return float(reverse["distance_km"].iloc[0])
    return None


def load_reservoir_info_from_master_dataset(master_df: pd.DataFrame) -> pd.DataFrame:
    required = ["저수지명", "시군", "유효저수량(천m3)"]
    missing = [column for column in required if column not in master_df.columns]
    if missing:
        raise ValueError(f"저수지 정보 생성에 필요한 컬럼이 없습니다: {missing}")
    optional = [column for column in ["위도", "경도"] if column in master_df.columns]
    info = master_df[required + optional].drop_duplicates(subset=["저수지명"]).copy()
    info["capacity_1000m3"] = pd.to_numeric(info["유효저수량(천m3)"], errors="coerce").fillna(0.0)
    return info


def _normalize_prediction_columns(predictions_df: pd.DataFrame) -> pd.DataFrame:
    df = predictions_df.copy()
    if "저수지명" not in df.columns:
        if "reservoir_name" in df.columns:
            df["저수지명"] = df["reservoir_name"]
        elif "source_reservoir" in df.columns:
            df["저수지명"] = df["source_reservoir"]
        else:
            raise ValueError("예측 결과에는 저수지명 또는 reservoir_name 컬럼이 필요합니다.")
    if "predicted_min_rate" not in df.columns:
        if "예측저수율" in df.columns:
            df["predicted_min_rate"] = df["예측저수율"]
        elif "predicted_rate" in df.columns:
            df["predicted_min_rate"] = df["predicted_rate"]
        else:
            raise ValueError("예측 결과에 predicted_min_rate 컬럼이 필요합니다.")
    df["predicted_min_rate"] = pd.to_numeric(df["predicted_min_rate"], errors="coerce")
    return df.dropna(subset=["저수지명", "predicted_min_rate"])


def _normalize_reservoir_info(reservoir_info_df: pd.DataFrame) -> pd.DataFrame:
    df = reservoir_info_df.copy()
    if "capacity_1000m3" not in df.columns:
        if "유효저수량(천m3)" in df.columns:
            df["capacity_1000m3"] = pd.to_numeric(df["유효저수량(천m3)"], errors="coerce")
        elif "총저수량_m3" in df.columns:
            df["capacity_1000m3"] = pd.to_numeric(df["총저수량_m3"], errors="coerce") / 1000.0
        else:
            raise ValueError("reservoir_info_df에는 capacity_1000m3 또는 유효저수량(천m3) 컬럼이 필요합니다.")
    if "시군" not in df.columns:
        df["시군"] = ""
    return df.dropna(subset=["저수지명", "capacity_1000m3"])


def _stage_for_prediction(row: pd.Series) -> dict:
    if pd.notna(row.get("stage_code")):
        return {
            "stage_code": row.get("stage_code"),
            "stage_label": row.get("stage_label"),
            "basis": row.get("stage_basis"),
            "basis_value": row.get("stage_basis_value"),
            "normal_rate": row.get("normal_rate_at_min"),
            "normal_ratio": row.get("normal_ratio_at_min"),
            "threshold_description": row.get("threshold_description"),
        }
    return classify_reservoir_stage(
        row.get("predicted_min_rate"),
        city=row.get("시군"),
        normal_rate=row.get("normal_rate_at_min"),
    )


def _supply_min_rate(city: str, normal_rate: Optional[float]) -> Optional[float]:
    if city == "강릉시":
        return 30.0
    numeric_normal = pd.to_numeric(pd.Series([normal_rate]), errors="coerce").iloc[0]
    if pd.isna(numeric_normal) or numeric_normal <= 0:
        return None
    return float(numeric_normal) * 0.60


def build_candidate_supplies(
    predictions_df: pd.DataFrame,
    reservoir_info_df: pd.DataFrame,
    distance_df: pd.DataFrame,
    target_reservoir: str = OBONG_NAME,
    max_distance_km: float = DEFAULT_MAX_DISTANCE_KM,
) -> pd.DataFrame:
    predictions = _normalize_prediction_columns(predictions_df)
    info = _normalize_reservoir_info(reservoir_info_df)
    distance_df = normalize_distance_columns(distance_df)

    supply_df = (
        predictions[predictions["저수지명"] != target_reservoir]
        .merge(info[["저수지명", "시군", "capacity_1000m3"]], on="저수지명", how="inner", suffixes=("", "_info"))
        .copy()
    )
    if "시군_info" in supply_df.columns:
        supply_df["시군"] = supply_df["시군"].fillna(supply_df["시군_info"])

    rows = []
    for _, row in supply_df.iterrows():
        stage = _stage_for_prediction(row)
        supply_allowed = is_supply_allowed(stage.get("stage_code"), min_allowed_stage="attention")
        supply_min_rate = _supply_min_rate(row.get("시군"), row.get("normal_rate_at_min"))
        if supply_min_rate is None:
            available_supply = 0.0
            supply_allowed = False
        else:
            available_supply = max(
                0.0,
                float(row["capacity_1000m3"]) * (float(row["predicted_min_rate"]) - supply_min_rate) / 100.0,
            )
        if not supply_allowed or available_supply <= 0:
            continue

        distance = get_distance_between_reservoirs(row["저수지명"], target_reservoir, distance_df)
        if distance is None or distance > max_distance_km:
            continue

        candidate = row.to_dict()
        candidate.update(
            {
                "source_reservoir": row["저수지명"],
                "target_reservoir": target_reservoir,
                "distance_km": distance,
                "supply_allowed": supply_allowed,
                "supply_min_rate": supply_min_rate,
                "available_supply_1000m3": available_supply,
                "stage_code": stage.get("stage_code"),
                "stage_label": stage.get("stage_label"),
                "stage_basis": stage.get("basis"),
                "normal_rate_at_min": stage.get("normal_rate"),
                "normal_ratio_at_min": stage.get("normal_ratio"),
            }
        )
        rows.append(candidate)

    return pd.DataFrame(rows)


def _candidate_records(candidates_df: pd.DataFrame) -> list[dict]:
    if candidates_df.empty:
        return []
    ordered = candidates_df.sort_values(["distance_km", "available_supply_1000m3"], ascending=[True, False])
    records = []
    for row in ordered.itertuples(index=False):
        records.append(
            {
                "source_reservoir": row.source_reservoir,
                "target_reservoir": row.target_reservoir,
                "distance_km": round(float(row.distance_km), 3),
                "available_supply_1000m3": round(float(row.available_supply_1000m3), 4),
                "predicted_min_rate": round(float(row.predicted_min_rate), 4),
                "stage_label": row.stage_label,
                "stage_basis": row.stage_basis,
                "normal_rate_at_min": None if pd.isna(row.normal_rate_at_min) else round(float(row.normal_rate_at_min), 4),
                "normal_ratio_at_min": None if pd.isna(row.normal_ratio_at_min) else round(float(row.normal_ratio_at_min), 4),
                "region": "강릉시" if row.시군 == "강릉시" else "타지역",
            }
        )
    return records


def optimize_water_transfer(
    obong_predicted_min_rate: float,
    candidate_predictions_df: pd.DataFrame,
    reservoir_info_df: pd.DataFrame,
    distance_df: Optional[pd.DataFrame] = None,
    target_rate: float = DEFAULT_TARGET_RATE,
    safety_rate: float = DEFAULT_SAFETY_RATE,
    max_distance_km: float = DEFAULT_MAX_DISTANCE_KM,
    target_reservoir: str = OBONG_NAME,
    prefer_region: str = "강릉시",
) -> dict:
    if distance_df is None:
        distance_df = load_distance_data(source="supabase")
    else:
        distance_df = normalize_distance_columns(distance_df)

    predictions = _normalize_prediction_columns(candidate_predictions_df)
    info = _normalize_reservoir_info(reservoir_info_df)
    obong_info = info[info["저수지명"] == target_reservoir]
    if obong_info.empty:
        raise ValueError(f"reservoir_info_df에 '{target_reservoir}' 저수지 정보가 없습니다.")

    obong_capacity = float(obong_info["capacity_1000m3"].iloc[0])
    required_water = max(0.0, obong_capacity * (target_rate - obong_predicted_min_rate) / 100.0)
    if required_water <= 0:
        return {
            "status": "SAFE",
            "required_water_1000m3": 0.0,
            "total_supplied_1000m3": 0.0,
            "total_available_supply_1000m3": 0.0,
            "feasible": True,
            "unmet_water_1000m3": 0.0,
            "candidate_count_100km": 0,
            "transfers": [],
            "candidates": [],
        }

    candidates_df = build_candidate_supplies(
        predictions_df=predictions,
        reservoir_info_df=info,
        distance_df=distance_df,
        target_reservoir=target_reservoir,
        max_distance_km=max_distance_km,
    )
    if candidates_df.empty:
        return {
            "status": "DANGER",
            "required_water_1000m3": round(required_water, 4),
            "total_supplied_1000m3": 0.0,
            "total_available_supply_1000m3": 0.0,
            "feasible": False,
            "unmet_water_1000m3": round(required_water, 4),
            "candidate_count_100km": 0,
            "transfers": [],
            "candidates": [],
        }

    problem = LpProblem("ohbong_water_transfer", LpMinimize)
    total_available_supply = float(candidates_df["available_supply_1000m3"].sum())
    sources = candidates_df["source_reservoir"].tolist()
    x = {source: LpVariable(f"x_{source}", lowBound=0) for source in sources}
    unmet = LpVariable("unmet_water_1000m3", lowBound=0)

    problem += (
        lpSum(
            row.distance_km * x[row.source_reservoir]
            + (PENALTY_OUT_OF_REGION * x[row.source_reservoir] if row.시군 != prefer_region else 0)
            for row in candidates_df.itertuples(index=False)
        )
        + PENALTY_UNMET * unmet
    )
    for row in candidates_df.itertuples(index=False):
        source = row.source_reservoir
        problem += x[source] <= row.available_supply_1000m3, f"supply_cap_{source}"
    problem += lpSum(x[source] for source in sources) + unmet == required_water, "meet_demand"
    problem.solve(PULP_CBC_CMD(msg=0))

    if LpStatus[problem.status] not in {"Optimal", "Not Solved"}:
        raise RuntimeError(f"MILP 솔버 비정상 종료: {LpStatus[problem.status]}")

    candidate_map = {row.source_reservoir: row for row in candidates_df.itertuples(index=False)}
    transfers = []
    total_supplied = 0.0
    for source in sources:
        amount = float(value(x[source]) or 0.0)
        if amount <= 1e-8:
            continue
        row = candidate_map[source]
        remaining_rate = row.predicted_min_rate - amount / row.capacity_1000m3 * 100.0
        remaining_stage = classify_reservoir_stage(
            remaining_rate,
            city=row.시군,
            normal_rate=row.normal_rate_at_min,
        )
        total_supplied += amount
        transfers.append(
            {
                "source_reservoir": source,
                "target_reservoir": target_reservoir,
                "transfer_amount_1000m3": round(amount, 4),
                "distance_km": round(float(row.distance_km), 3),
                "available_supply_1000m3": round(float(row.available_supply_1000m3), 4),
                "predicted_min_rate": round(float(row.predicted_min_rate), 4),
                "stage_label": row.stage_label,
                "stage_basis": row.stage_basis,
                "normal_rate_at_min": None if pd.isna(row.normal_rate_at_min) else round(float(row.normal_rate_at_min), 4),
                "normal_ratio_at_min": None if pd.isna(row.normal_ratio_at_min) else round(float(row.normal_ratio_at_min), 4),
                "remaining_rate_after_transfer": round(float(remaining_rate), 4),
                "remaining_stage_code": remaining_stage["stage_code"],
                "remaining_stage_label": remaining_stage["stage_label"],
                "region": "강릉시" if row.시군 == prefer_region else "타지역",
            }
        )

    unmet_value = float(value(unmet) or 0.0)
    return {
        "status": "DANGER",
        "required_water_1000m3": round(required_water, 4),
        "total_supplied_1000m3": round(total_supplied, 4),
        "total_available_supply_1000m3": round(total_available_supply, 4),
        "feasible": unmet_value <= 1e-6,
        "unmet_water_1000m3": round(unmet_value, 4),
        "candidate_count_100km": int(len(candidates_df)),
        "transfers": transfers,
        "candidates": _candidate_records(candidates_df),
    }


def optimize_from_seq2seq_summary(
    min_rate_summary_df: pd.DataFrame,
    reservoir_info_df: pd.DataFrame,
    distance_df: Optional[pd.DataFrame] = None,
    target_rate: float = DEFAULT_TARGET_RATE,
    safety_rate: float = DEFAULT_SAFETY_RATE,
    max_distance_km: float = DEFAULT_MAX_DISTANCE_KM,
) -> dict:
    if distance_df is None:
        distance_df = load_distance_data(source="supabase")
    summary = _normalize_prediction_columns(min_rate_summary_df)
    obong_row = summary[summary["저수지명"] == OBONG_NAME]
    if obong_row.empty:
        raise ValueError("seq2seq 요약 결과에 오봉 저수지가 없습니다.")
    obong_rate = float(obong_row["predicted_min_rate"].iloc[0])
    return optimize_water_transfer(
        obong_predicted_min_rate=obong_rate,
        candidate_predictions_df=summary,
        reservoir_info_df=reservoir_info_df,
        distance_df=distance_df,
        target_rate=target_rate,
        safety_rate=safety_rate,
        max_distance_km=max_distance_km,
    )


def _print_distance_summary(distance_df: pd.DataFrame) -> None:
    connected = distance_df[(distance_df["source_reservoir"] == OBONG_NAME) | (distance_df["target_reservoir"] == OBONG_NAME)]
    within_100km = connected[connected["distance_km"] <= DEFAULT_MAX_DISTANCE_KM]
    print(f"reservoir_distances 로딩 행 수: {len(distance_df):,}")
    print(f"거리 테이블 컬럼 목록: {list(distance_df.columns)}")
    print(f"오봉과 연결된 거리 데이터 개수: {len(connected):,}")
    print(f"100km 이내 후보 개수: {len(within_100km):,}")


if __name__ == "__main__":
    try:
        distances = load_distance_data(source="supabase")
        _print_distance_summary(distances)
        print("transport_optimizer.py 로딩 테스트 완료")
    except Exception as error:
        print("transport_optimizer.py 실행 테스트에 실패했습니다.")
        print(error)
