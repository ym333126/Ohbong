import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.db.supabase_client import load_master_dataset, load_reservoir_distances_from_supabase  # noqa: E402
from src.model.predict_seq2seq import predict_seq2seq_forecasts, summarize_min_rates  # noqa: E402
from src.optimization.transport_optimizer import (  # noqa: E402
    DEFAULT_MAX_DISTANCE_KM,
    OBONG_NAME,
    build_candidate_supplies,
    load_reservoir_info_from_master_dataset,
    optimize_from_seq2seq_summary,
)


def run_prediction_optimization(as_of_date: str | None = None) -> dict:
    print("1. Supabase master_dataset 로드")
    master_df = load_master_dataset(source="supabase")

    print("2. seq2seq 모델로 22개 저수지 predicted_min_rate 및 단계 생성")
    daily_forecast_df = predict_seq2seq_forecasts(master_df, as_of_date=as_of_date)
    min_rate_summary_df = summarize_min_rates(daily_forecast_df)

    print("3. Supabase reservoir_distances 로드")
    distance_df = load_reservoir_distances_from_supabase()

    connected = distance_df[(distance_df["source_reservoir"] == OBONG_NAME) | (distance_df["target_reservoir"] == OBONG_NAME)]
    within_100km = connected[connected["distance_km"] <= DEFAULT_MAX_DISTANCE_KM]
    print(f"reservoir_distances 로딩 행 수: {len(distance_df):,}")
    print(f"거리 테이블 컬럼 목록: {list(distance_df.columns)}")
    print(f"오봉과 연결된 거리 데이터 개수: {len(connected):,}")
    print(f"100km 이내 후보 개수: {len(within_100km):,}")

    print("4. 정상 또는 관심 단계 공급 후보 필터링")
    reservoir_info_df = load_reservoir_info_from_master_dataset(master_df)
    candidates_df = build_candidate_supplies(
        predictions_df=min_rate_summary_df,
        reservoir_info_df=reservoir_info_df,
        distance_df=distance_df,
    )
    print(f"공급 가능 후보 개수: {len(candidates_df):,}")

    print("5. PuLP 최적화 실행")
    optimization_result = optimize_from_seq2seq_summary(
        min_rate_summary_df=min_rate_summary_df,
        reservoir_info_df=reservoir_info_df,
        distance_df=distance_df,
    )

    print("6. 결과 출력")
    print(f"필요 물량(천m3): {optimization_result['required_water_1000m3']}")
    print(f"공급 물량(천m3): {optimization_result['total_supplied_1000m3']}")
    print(f"실현 가능 여부: {optimization_result['feasible']}")
    print(f"미충족 물량(천m3): {optimization_result['unmet_water_1000m3']}")
    for transfer in optimization_result["transfers"]:
        print(
            f"- {transfer['source_reservoir']} -> {transfer['target_reservoir']}: "
            f"{transfer['transfer_amount_1000m3']}천m3, "
            f"{transfer['distance_km']}km, "
            f"현재 단계 {transfer.get('stage_label')}, "
            f"공급 후 단계 {transfer.get('remaining_stage_label')}"
        )

    return {
        "daily_forecast": daily_forecast_df,
        "min_rate_summary": min_rate_summary_df,
        "candidate_supplies": candidates_df,
        "optimization_result": optimization_result,
    }


if __name__ == "__main__":
    try:
        run_prediction_optimization()
    except Exception as error:
        print("prediction -> optimization 파이프라인 실행에 실패했습니다.")
        print(error)
