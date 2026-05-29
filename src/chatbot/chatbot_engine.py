import re
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chatbot.gemini_answer_formatter import format_answer_with_gemini  # noqa: E402
from src.chatbot.gemini_intent_parser import parse_intent_with_gemini  # noqa: E402
from src.db.supabase_client import load_master_dataset, load_reservoir_distances_from_supabase  # noqa: E402
from src.model.predict_seq2seq import predict_seq2seq_forecasts, summarize_min_rates  # noqa: E402
from src.optimization.transport_optimizer import (  # noqa: E402
    OBONG_NAME,
    load_reservoir_info_from_master_dataset,
    optimize_from_seq2seq_summary,
)
from src.utils.normal_rate import get_normal_rate_for_date  # noqa: E402
from src.utils.risk_policy import classify_reservoir_stage  # noqa: E402


RESERVOIR_COL = "저수지명"
DATE_COL = "날짜"
CITY_COL = "시군"
OBSERVED_RATE_COL = "저수율"
EMA_RATE_COL = "저수율_5일EMA"
OBSERVATION_COLUMNS = [RESERVOIR_COL, DATE_COL, OBSERVED_RATE_COL, EMA_RATE_COL, "강수량", "SPI3", "SPI6"]
PREDICTION_TABLE_COLUMNS = [
    RESERVOIR_COL,
    "예측 기간",
    "예상 최저 저수율(%)",
    "최저 예상일",
    "평균 예측 저수율(%)",
    "단계",
    "판단 기준",
    "평년 저수율(%)",
    "평년 대비 비율(%)",
]
STAGE_LIST_COLUMNS = [
    RESERVOIR_COL,
    "예상 최저 저수율(%)",
    "최저 예상일",
    "하한값(%)",
    "단계",
    "판단 기준",
    "평년 저수율(%)",
    "평년 대비 비율(%)",
]


def _fmt_date(value: Any) -> str:
    if value is None or pd.isna(value):
        return "-"
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _fmt_rate(value: Any) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "확인 불가" if pd.isna(number) else f"{float(number):.2f}%"


class OhbongChatbot:
    def __init__(self) -> None:
        self.master_df: pd.DataFrame | None = None
        self.distance_df: pd.DataFrame | None = None
        self.last_daily_forecast: pd.DataFrame | None = None
        self.last_prediction_summary: pd.DataFrame | None = None
        self.last_optimization_result: dict | None = None
        self.last_intent_plan: dict[str, Any] | None = None
        self.last_optimization_plan: dict[str, Any] | None = None

        try:
            self.master_df = load_master_dataset(source="supabase")
            self.master_df[DATE_COL] = pd.to_datetime(self.master_df[DATE_COL], errors="coerce")
        except Exception as exc:
            raise RuntimeError("Supabase master_dataset 로딩에 실패했습니다. .env 설정과 RLS SELECT 정책을 확인하세요.") from exc

        try:
            self.distance_df = load_reservoir_distances_from_supabase()
        except Exception as exc:
            print(f"[안내] reservoir_distances는 최적화 요청 시 다시 로딩합니다. 현재 오류: {exc}")
            self.distance_df = None

    def _response(
        self,
        answer: str,
        intent: str,
        table: Optional[pd.DataFrame] = None,
        prediction_summary: Optional[pd.DataFrame] = None,
        optimization_result: Optional[dict] = None,
        intent_plan: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        return {
            "answer": answer,
            "intent": intent,
            "table": table,
            "prediction_summary": prediction_summary,
            "optimization_result": optimization_result,
            "intent_plan": intent_plan,
        }

    def classify_intent(self, message: str) -> str:
        text = message.strip().lower()
        if not text:
            return "unknown"
        if any(keyword in text for keyword in ["도움", "사용법", "help", "예시", "뭘 물어", "무엇을 물어"]):
            return "help"
        optimization_keywords = ["어디서", "얼마", "가져와", "운송", "최적화", "공급", "회복"]
        pipeline_keywords = ["최적화까지", "운송 계획까지", "예측하고", "예측 후", "위험하면"]
        prediction_keywords = ["예측", "다음달", "다음 달", "향후", "앞으로", "30일", "한 달", "물 부족", "최저", "위험"]
        table_keywords = ["표", "목록", "비교", "22개", "전체", "여러", "랑", ",", "，"]
        observation_keywords = ["저수율", "알려줘", "조회", "현재", "오늘", "관측", "실제"]
        has_optimization = any(keyword in text for keyword in optimization_keywords)
        has_prediction = any(keyword in text for keyword in prediction_keywords) or self._has_future_month_request(text)
        has_table = any(keyword in text for keyword in table_keywords) or len(self.extract_reservoir_names(message)) >= 2
        has_observation = any(keyword in text for keyword in observation_keywords)
        if has_optimization and (has_prediction or any(keyword in text for keyword in pipeline_keywords)):
            return "pipeline"
        if has_table and (has_prediction or "단계" in text or "예측 결과" in text):
            return "comparison_or_table"
        if has_optimization:
            return "optimization"
        if has_prediction:
            return "prediction"
        if has_table and has_observation:
            return "comparison_or_table"
        if has_observation:
            return "observation_lookup"
        return "unknown"

    @staticmethod
    def _has_future_month_request(text: str) -> bool:
        return bool(re.search(r"\d{1,2}\s*월", text)) and any(keyword in text for keyword in ["예측", "어떻게", "전망", "위험"])

    @staticmethod
    def _requests_prediction(message: str) -> bool:
        text = message.strip().lower()
        return any(keyword in text for keyword in ["예측", "다음달", "다음 달", "향후", "앞으로", "30일", "한 달", "물 부족", "최저", "위험"]) or OhbongChatbot._has_future_month_request(text)

    def extract_reservoir_names(self, message: str) -> list[str]:
        if self.master_df is None:
            return [OBONG_NAME]
        names = sorted(self.master_df[RESERVOIR_COL].dropna().astype(str).unique(), key=len, reverse=True)
        found = [name for name in names if name in message]
        return found or [OBONG_NAME]

    def parse_date_or_period(self, message: str) -> dict[str, Any]:
        latest = pd.Timestamp(self.master_df[DATE_COL].max()) if self.master_df is not None else pd.Timestamp.today().normalize()
        text = message.strip()
        result = {"target_date": None, "month": None, "latest_date": latest, "period_type": None}
        if "오늘" in text or "현재" in text:
            result.update({"target_date": latest, "period_type": "date"})
            return result
        explicit = re.search(r"(20\d{2})[-./년\s]+(\d{1,2})[-./월\s]+(\d{1,2})", text)
        if explicit:
            year, month, day = map(int, explicit.groups())
            result.update({"target_date": pd.Timestamp(year=year, month=month, day=day), "period_type": "date"})
            return result
        day_only = re.search(r"(\d{1,2})\s*월\s*(\d{1,2})\s*일", text)
        if day_only:
            month, day = map(int, day_only.groups())
            result.update({"target_date": pd.Timestamp(year=int(latest.year), month=month, day=day), "period_type": "date"})
            return result
        month_only = re.search(r"(\d{1,2})\s*월", text)
        if month_only:
            result.update({"month": int(month_only.group(1)), "period_type": "month"})
            return result
        if any(keyword in text for keyword in ["다음달", "다음 달", "향후", "앞으로"]):
            result["period_type"] = "next_30_days"
            return result
        result.update({"target_date": latest, "period_type": "date"})
        return result

    def apply_intent_plan_to_period(self, period_info: dict[str, Any], intent_plan: dict[str, Any]) -> dict[str, Any]:
        updated = dict(period_info)
        if intent_plan.get("target_date"):
            parsed = pd.to_datetime(intent_plan["target_date"], errors="coerce")
            if not pd.isna(parsed):
                updated.update({"target_date": pd.Timestamp(parsed), "period_type": "date"})
        if intent_plan.get("target_month") is not None:
            updated.update({"month": int(intent_plan["target_month"]), "period_type": "month"})
        return updated

    def parse_user_scenario(self, message: str) -> dict[str, Any]:
        text = message.strip()
        scenario = {
            "rainfall_multiplier": 1.0,
            "spi3_delta": 0.0,
            "spi6_delta": 0.0,
            "temperature_delta": 0.0,
            "target_rate": None,
            "safety_rate": None,
        }
        target_match = re.search(r"목표\s*저수율\s*(\d+(?:\.\d+)?)", text)
        if target_match:
            scenario["target_rate"] = float(target_match.group(1))
        safety_match = re.search(r"안전\s*저수율\s*(\d+(?:\.\d+)?)", text)
        if safety_match:
            scenario["safety_rate"] = float(safety_match.group(1))
        return scenario

    @staticmethod
    def parse_candidate_limit(message: str, default: int = 5) -> int:
        for pattern in [r"(?:후보|상위|최종)\s*(\d+)\s*개", r"(\d+)\s*개\s*(?:후보|까지|만)"]:
            match = re.search(pattern, message)
            if match:
                return max(1, min(int(match.group(1)), 20))
        return default

    @staticmethod
    def parse_region_filter(message: str) -> Optional[str]:
        text = message.strip()
        asks_for_gangneung = any(keyword in text for keyword in ["강릉시", "강릉 소재", "강릉 저수지", "강릉에 있는", "강릉"])
        excludes_gangneung = any(keyword in text for keyword in ["강릉시가 아닌", "강릉시 아닌", "강릉이 아닌", "강릉 아닌", "타지역", "다른 지역", "섞어서", "포함해서"])
        if asks_for_gangneung and not excludes_gangneung:
            return "강릉시"
        return None

    @staticmethod
    def _mentions_non_gangneung(message: str) -> bool:
        text = message.strip()
        return any(keyword in text for keyword in ["강릉시가 아닌", "강릉시 아닌", "강릉이 아닌", "강릉 아닌", "타지역", "다른 지역", "섞어서", "포함해서", "아닌 곳", "아닌곳"])

    @staticmethod
    def is_optimization_followup(message: str) -> bool:
        text = message.strip()
        followup_keywords = ["그럼", "그러면", "이번엔", "이번에는", "아닌 곳", "아닌곳", "섞어서", "포함해서", "제외", "말고"]
        optimization_context_keywords = ["가져", "후보", "저수지", "강릉", "타지역", "지역", "운송", "얼마"]
        return any(keyword in text for keyword in followup_keywords) and any(keyword in text for keyword in optimization_context_keywords)

    @staticmethod
    def resolve_allowed_region(message: str, previous_region: Optional[str] = None) -> Optional[str]:
        text = message.strip()
        if OhbongChatbot._mentions_non_gangneung(text):
            return None
        if any(keyword in text for keyword in ["강릉시", "강릉 소재", "강릉 저수지", "강릉에 있는", "강릉"]):
            return "강릉시"
        return previous_region

    def build_rule_based_intent_plan(self, message: str, target_rate: float, safety_rate: float, max_distance_km: float) -> dict[str, Any]:
        period_info = self.parse_date_or_period(message)
        scenario = self.parse_user_scenario(message)
        return {
            "intent": self.classify_intent(message),
            "target_reservoirs": self.extract_reservoir_names(message),
            "target_date": _fmt_date(period_info.get("target_date")) if period_info.get("target_date") is not None else None,
            "target_month": period_info.get("month"),
            "forecast_horizon": 30,
            "target_rate": scenario["target_rate"] if scenario["target_rate"] is not None else target_rate,
            "safety_rate": scenario["safety_rate"] if scenario["safety_rate"] is not None else safety_rate,
            "max_distance_km": max_distance_km,
            "allowed_region": self.parse_region_filter(message),
            "top_k": self.parse_candidate_limit(message),
            "need_table": self.classify_intent(message) in {"comparison_or_table", "optimization", "pipeline"},
            "scenario": {
                "rainfall_multiplier": scenario.get("rainfall_multiplier", 1.0),
                "spi3_delta": scenario.get("spi3_delta", 0.0),
                "spi6_delta": scenario.get("spi6_delta", 0.0),
                "temperature_delta": scenario.get("temperature_delta", 0.0),
            },
        }

    def _city_for_reservoir(self, reservoir_name: str) -> Optional[str]:
        if self.master_df is None or CITY_COL not in self.master_df.columns:
            return None
        rows = self.master_df[self.master_df[RESERVOIR_COL] == reservoir_name]
        if rows.empty:
            return None
        return str(rows[CITY_COL].dropna().iloc[-1]) if not rows[CITY_COL].dropna().empty else None

    def run_observation_lookup(self, reservoir_names: list[str], target_date: pd.Timestamp) -> pd.DataFrame:
        if self.master_df is None:
            raise RuntimeError("master_dataset이 로드되지 않았습니다.")
        rows = []
        for reservoir_name in reservoir_names:
            reservoir_df = self.master_df[self.master_df[RESERVOIR_COL] == reservoir_name].sort_values(DATE_COL)
            exact = reservoir_df[reservoir_df[DATE_COL] == target_date]
            if exact.empty:
                rows.append({RESERVOIR_COL: reservoir_name, DATE_COL: _fmt_date(target_date), OBSERVED_RATE_COL: None, EMA_RATE_COL: None, "강수량": None, "SPI3": None, "SPI6": None, "단계": "판단불가", "판단 기준": "데이터 없음"})
                continue
            row = exact.iloc[-1]
            city = row.get(CITY_COL)
            normal_rate = get_normal_rate_for_date(self.master_df, reservoir_name, target_date, target_col=EMA_RATE_COL)
            stage = classify_reservoir_stage(row.get(OBSERVED_RATE_COL), city=city, normal_rate=normal_rate)
            item = {column: row.get(column) for column in OBSERVATION_COLUMNS}
            item.update(
                {
                    "단계": stage["stage_label"],
                    "판단 기준": stage["threshold_description"],
                    "평년 저수율(%)": stage["normal_rate"],
                    "평년 대비 비율(%)": stage["normal_ratio"],
                }
            )
            rows.append(item)
        result = pd.DataFrame(rows)
        if DATE_COL in result.columns:
            result[DATE_COL] = pd.to_datetime(result[DATE_COL], errors="coerce").dt.strftime("%Y-%m-%d")
        return result

    def format_observation_answer(self, table: pd.DataFrame) -> str:
        if table.empty:
            return "조회할 관측 데이터가 없습니다."
        if len(table) == 1:
            row = table.iloc[0]
            if pd.isna(row.get(OBSERVED_RATE_COL)):
                return f"{row.get(DATE_COL)} 기준 {row.get(RESERVOIR_COL)} 저수지 관측 데이터가 없습니다."
            basis = row.get("판단 기준")
            ratio_text = ""
            if pd.notna(row.get("평년 대비 비율(%)")):
                ratio_text = f" 평년 저수율 대비 비율은 {float(row.get('평년 대비 비율(%)')):.2f}%입니다."
            return (
                f"{row.get(DATE_COL)} 기준 {row.get(RESERVOIR_COL)} 저수지의 관측 저수율은 "
                f"{_fmt_rate(row.get(OBSERVED_RATE_COL))}입니다. 참고용 5일 EMA 저수율은 {_fmt_rate(row.get(EMA_RATE_COL))}입니다. "
                f"{basis}으로는 {row.get('단계')} 단계입니다.{ratio_text}"
            )
        return "요청하신 저수지의 관측값과 단계를 아래 표로 정리했습니다."

    def _ensure_prediction(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self.master_df is None:
            raise RuntimeError("master_dataset이 로드되지 않았습니다.")
        daily_forecast = predict_seq2seq_forecasts(self.master_df)
        summary = summarize_min_rates(daily_forecast)
        if daily_forecast.empty or summary.empty:
            raise RuntimeError("seq2seq 예측 결과가 비어 있습니다.")
        self.last_daily_forecast = daily_forecast
        self.last_prediction_summary = summary
        return daily_forecast, summary

    def _filter_forecast_period(self, daily_forecast: pd.DataFrame, period_info: dict[str, Any]) -> tuple[pd.DataFrame, str]:
        forecast_df = daily_forecast.copy()
        forecast_df["forecast_date"] = pd.to_datetime(forecast_df["forecast_date"], errors="coerce")
        start = forecast_df["forecast_date"].min()
        end = forecast_df["forecast_date"].max()
        as_of_date = daily_forecast["as_of_date"].iloc[0]
        if period_info.get("period_type") == "month" and period_info.get("month") is not None:
            month = int(period_info["month"])
            filtered = forecast_df[forecast_df["forecast_date"].dt.month == month].copy()
            if filtered.empty:
                return filtered, f"현재 모델은 {_fmt_date(start)}부터 {_fmt_date(end)}까지만 예측하므로 {month}월 예측값이 없습니다."
            return filtered, f"현재 모델은 {as_of_date} 기준으로 {_fmt_date(end)}까지 예측합니다. 따라서 {month}월 예측은 {_fmt_date(filtered['forecast_date'].min())}부터 {_fmt_date(filtered['forecast_date'].max())}까지를 기준으로 계산했습니다."
        return forecast_df, f"현재 모델은 {as_of_date} 기준으로 {_fmt_date(start)}부터 {_fmt_date(end)}까지 예측합니다."

    def build_prediction_table(self, daily_forecast: pd.DataFrame, reservoir_names: list[str], period_info: dict[str, Any]) -> tuple[pd.DataFrame, str]:
        period_df, period_message = self._filter_forecast_period(daily_forecast, period_info)
        if period_df.empty:
            return pd.DataFrame(columns=PREDICTION_TABLE_COLUMNS), period_message
        selected = period_df[period_df[RESERVOIR_COL].isin(reservoir_names)].copy()
        rows = []
        for reservoir_name, group in selected.groupby(RESERVOIR_COL):
            min_idx = group["predicted_rate"].idxmin()
            min_row = group.loc[min_idx]
            rows.append(
                {
                    RESERVOIR_COL: reservoir_name,
                    "예측 기간": f"{_fmt_date(group['forecast_date'].min())} ~ {_fmt_date(group['forecast_date'].max())}",
                    "예상 최저 저수율(%)": round(float(min_row["predicted_rate"]), 2),
                    "최저 예상일": _fmt_date(min_row["forecast_date"]),
                    "평균 예측 저수율(%)": round(float(group["predicted_rate"].mean()), 2),
                    "단계": min_row.get("stage_label"),
                    "판단 기준": min_row.get("threshold_description"),
                    "평년 저수율(%)": min_row.get("normal_rate"),
                    "평년 대비 비율(%)": min_row.get("normal_ratio"),
                }
            )
        return pd.DataFrame(rows, columns=PREDICTION_TABLE_COLUMNS), period_message

    def format_prediction_answer(self, table: pd.DataFrame, period_message: str) -> str:
        if table.empty:
            return period_message
        if len(table) == 1:
            row = table.iloc[0]
            ratio_text = ""
            if pd.notna(row.get("평년 대비 비율(%)")):
                ratio_text = f" 평년 대비 비율은 {float(row.get('평년 대비 비율(%)')):.2f}%입니다."
            return (
                f"{period_message} {row[RESERVOIR_COL]} 저수지는 예측 기간 중 최저 저수율이 약 "
                f"{row['예상 최저 저수율(%)']:.2f}%로 예상되며, 최저 예상일은 {row['최저 예상일']}입니다. "
                f"{row['판단 기준']}으로는 {row['단계']} 단계입니다.{ratio_text}\n\n"
                "운송 방안까지 보려면 '어디서 얼마나 가져와야 해?'라고 입력해주세요."
            )
        return f"{period_message} 아래 표는 요청하신 저수지들의 예측 비교 결과입니다."

    def build_stage_list_table(self, summary: pd.DataFrame) -> pd.DataFrame:
        table = summary.copy().sort_values(["stage_code", "predicted_min_rate"], ascending=[True, True])
        table = table.rename(
            columns={
                "predicted_min_rate": "예상 최저 저수율(%)",
                "min_forecast_date": "최저 예상일",
                "lower_bound_at_min": "하한값(%)",
                "stage_label": "단계",
                "stage_basis": "판단 기준",
                "normal_rate_at_min": "평년 저수율(%)",
                "normal_ratio_at_min": "평년 대비 비율(%)",
            }
        )
        return table[[column for column in STAGE_LIST_COLUMNS if column in table.columns]].reset_index(drop=True)

    def run_optimization(
        self,
        prediction_summary_df: Optional[pd.DataFrame] = None,
        target_rate: float = 50,
        safety_rate: float = 40,
        max_distance_km: float = 100,
        allowed_region: Optional[str] = None,
    ) -> dict:
        if prediction_summary_df is None:
            prediction_summary_df = self.last_prediction_summary
        if prediction_summary_df is None:
            _, prediction_summary_df = self._ensure_prediction()
        if self.master_df is None:
            raise RuntimeError("master_dataset이 로드되지 않았습니다.")
        if self.distance_df is None:
            self.distance_df = load_reservoir_distances_from_supabase()
        reservoir_info_df = load_reservoir_info_from_master_dataset(self.master_df)
        if allowed_region:
            reservoir_info_df = reservoir_info_df[(reservoir_info_df[RESERVOIR_COL] == OBONG_NAME) | (reservoir_info_df[CITY_COL] == allowed_region)].copy()
        result = optimize_from_seq2seq_summary(
            min_rate_summary_df=prediction_summary_df,
            reservoir_info_df=reservoir_info_df,
            distance_df=self.distance_df,
            target_rate=target_rate,
            safety_rate=safety_rate,
            max_distance_km=max_distance_km,
        )
        result.update({"target_rate": target_rate, "safety_rate": safety_rate, "max_distance_km": max_distance_km, "target_reservoir": OBONG_NAME, "allowed_region": allowed_region})
        obong_row = prediction_summary_df[prediction_summary_df[RESERVOIR_COL] == OBONG_NAME].iloc[0]
        result["obong_predicted_min_rate"] = float(obong_row["predicted_min_rate"])
        self.last_optimization_result = result
        return result

    def format_optimization_answer(self, result: dict, candidate_limit: int = 5) -> tuple[str, pd.DataFrame]:
        transfers = pd.DataFrame(result.get("transfers", []))
        if not transfers.empty:
            transfers = transfers.sort_values(["transfer_amount_1000m3", "distance_km"], ascending=[False, True]).reset_index(drop=True)
            display_transfers = transfers.head(candidate_limit).copy()
        else:
            display_transfers = transfers
        shortage = float(result.get("unmet_water_1000m3", result.get("shortage_1000m3", 0.0)) or 0.0)
        feasible = bool(result.get("feasible"))
        region_text = f"{result.get('allowed_region')} 저수지로만 제한하면 " if result.get("allowed_region") else ""
        candidate_scope = f"100km 이내 {result.get('allowed_region')} 후보 저수지" if result.get("allowed_region") else "100km 이내 후보 저수지"
        answer = (
            f"{region_text}오봉 저수지를 목표 저수율 {result.get('target_rate', 50):.0f}%까지 회복하려면 "
            f"약 {result.get('required_water_1000m3', 0):.2f}천m3가 필요합니다. "
            f"현재 {candidate_scope}에서 정상 또는 관심 단계로 공급 가능한 물량은 약 {result.get('total_supplied_1000m3', 0):.2f}천m3입니다. "
        )
        answer += "따라서 현재 조건에서는 목표 달성이 가능합니다." if feasible else f"약 {shortage:.2f}천m3가 부족하여 현재 조건에서는 목표 달성이 어렵습니다."
        if not display_transfers.empty:
            lines = [
                f"{index + 1}. {row['source_reservoir']}에서 {row['transfer_amount_1000m3']:.2f}천m3 (거리 {row['distance_km']:.2f}km, 현재 단계 {row.get('stage_label')})"
                for index, row in display_transfers.iterrows()
            ]
            answer += f"\n\n최종 후보 {len(display_transfers)}개는 다음과 같습니다.\n" + "\n".join(lines)
        else:
            answer += "\n\n현재 조건에서 표시할 운송 후보가 없습니다."
        return answer, display_transfers

    def _apply_gemini(
        self,
        answer: str,
        intent: str,
        user_message: str,
        use_gemini: bool,
        prediction_result: Optional[dict] = None,
        optimization_result: Optional[dict] = None,
        observation_result: Optional[dict] = None,
        table: Optional[pd.DataFrame] = None,
    ) -> str:
        if not use_gemini:
            return answer
        table_preview = table.head(5).to_dict("records") if table is not None and not table.empty else None
        return format_answer_with_gemini(
            user_message=user_message,
            intent=intent,
            prediction_result=prediction_result,
            optimization_result=optimization_result,
            observation_result=observation_result,
            table_preview=table_preview,
            fallback_answer=answer,
        )

    def handle_message(
        self,
        message: str,
        use_gemini: bool = True,
        target_rate: float = 50,
        safety_rate: float = 40,
        max_distance_km: float = 100,
    ) -> dict[str, Any]:
        fallback_plan = self.build_rule_based_intent_plan(message, target_rate, safety_rate, max_distance_km)
        known_reservoirs = sorted(self.master_df[RESERVOIR_COL].dropna().astype(str).unique().tolist()) if self.master_df is not None else [OBONG_NAME]
        latest_date = _fmt_date(self.master_df[DATE_COL].max()) if self.master_df is not None else None
        intent_plan = parse_intent_with_gemini(message, fallback_plan=fallback_plan, known_reservoirs=known_reservoirs, latest_date=latest_date)
        if self.last_optimization_plan and self.is_optimization_followup(message):
            previous_plan = dict(self.last_optimization_plan)
            previous_plan.update({key: value for key, value in intent_plan.items() if value not in [None, [], "unknown"]})
            previous_plan["intent"] = "optimization"
            if intent_plan.get("top_k") is None:
                previous_plan["top_k"] = self.last_optimization_plan.get("top_k")
            intent_plan = previous_plan
        intent = intent_plan["intent"]
        period_info = self.apply_intent_plan_to_period(self.parse_date_or_period(message), intent_plan)
        reservoir_names = intent_plan.get("target_reservoirs") or [OBONG_NAME]
        candidate_limit = intent_plan.get("top_k") or self.parse_candidate_limit(message)
        target_rate = float(intent_plan.get("target_rate", target_rate))
        safety_rate = float(intent_plan.get("safety_rate", safety_rate))
        max_distance_km = float(intent_plan.get("max_distance_km", max_distance_km))
        previous_region = self.last_optimization_plan.get("allowed_region") if self.last_optimization_plan else None
        planned_region = intent_plan.get("allowed_region")
        allowed_region = planned_region if planned_region is not None else self.resolve_allowed_region(message, previous_region)
        intent_plan["allowed_region"] = allowed_region

        try:
            if intent == "observation_lookup":
                table = self.run_observation_lookup(reservoir_names, period_info["target_date"])
                answer = self.format_observation_answer(table)
                self.last_intent_plan = intent_plan
                return self._response(answer, intent, table if len(table) > 1 or intent_plan.get("need_table") else None, intent_plan=intent_plan)
            if intent == "prediction":
                daily_forecast, summary = self._ensure_prediction()
                table, period_message = self.build_prediction_table(daily_forecast, reservoir_names, period_info)
                answer = self.format_prediction_answer(table, period_message)
                answer = self._apply_gemini(answer, intent, message, use_gemini, prediction_result={"rows": table.head(5).to_dict("records")}, table=table)
                self.last_intent_plan = intent_plan
                return self._response(answer, intent, table if intent_plan.get("need_table") or len(table) > 1 else None, summary, None, intent_plan)
            if intent == "comparison_or_table":
                if not self._requests_prediction(message):
                    table = self.run_observation_lookup(reservoir_names, period_info["target_date"])
                    answer = self.format_observation_answer(table)
                    self.last_intent_plan = intent_plan
                    return self._response(answer, "observation_lookup", table, None, None, intent_plan)
                daily_forecast, summary = self._ensure_prediction()
                if any(keyword in message for keyword in ["목록", "표", "22개", "전체", "단계"]):
                    table = self.build_stage_list_table(summary)
                    answer = "아래 표는 향후 30일 예측 기준 저수지 단계 목록입니다."
                    self.last_intent_plan = intent_plan
                    return self._response(answer, intent, table, summary, None, intent_plan)
                table, period_message = self.build_prediction_table(daily_forecast, reservoir_names, period_info)
                answer = self.format_prediction_answer(table, period_message)
                self.last_intent_plan = intent_plan
                return self._response(answer, intent, table, summary, None, intent_plan)
            if intent == "optimization":
                result = self.run_optimization(target_rate=target_rate, safety_rate=safety_rate, max_distance_km=max_distance_km, allowed_region=allowed_region)
                answer, table = self.format_optimization_answer(result, candidate_limit=candidate_limit)
                result["allocations_display"] = table.to_dict("records") if not table.empty else []
                self.last_intent_plan = intent_plan
                self.last_optimization_plan = intent_plan
                return self._response(answer, intent, table if not table.empty else None, None, result, intent_plan)
            if intent == "pipeline":
                daily_forecast, summary = self._ensure_prediction()
                prediction_table, period_message = self.build_prediction_table(daily_forecast, [OBONG_NAME], period_info)
                prediction_answer = self.format_prediction_answer(prediction_table, period_message)
                result = self.run_optimization(summary, target_rate=target_rate, safety_rate=safety_rate, max_distance_km=max_distance_km, allowed_region=allowed_region)
                optimization_answer, transfer_table = self.format_optimization_answer(result, candidate_limit=candidate_limit)
                result["allocations_display"] = transfer_table.to_dict("records") if not transfer_table.empty else []
                answer = f"{prediction_answer}\n\n{optimization_answer}"
                self.last_intent_plan = intent_plan
                self.last_optimization_plan = intent_plan
                return self._response(answer, intent, transfer_table if not transfer_table.empty else None, summary, result, intent_plan)
            if intent == "help":
                return self._response(self.help_message(), intent, intent_plan=intent_plan)
            self.last_daily_forecast = None
            self.last_prediction_summary = None
            self.last_optimization_result = None
            return self._response(self.unsupported_message(), intent, intent_plan=intent_plan)
        except Exception as exc:
            return self._response(f"요청 처리 중 문제가 발생했습니다.\n{exc}", intent, intent_plan=intent_plan)

    @staticmethod
    def help_message() -> str:
        return (
            "아래처럼 질문할 수 있습니다.\n\n"
            "- 오봉 저수지 5월 26일 저수율을 알려줘\n"
            "- 6월 오봉 저수지 저수율 예측해줘\n"
            "- 22개 저수지 예측 결과 표로 보여줘\n"
            "- 어디서 얼만큼 가져와야하는지 최종 후보 5개까지 알려줘"
        )

    @staticmethod
    def unsupported_message() -> str:
        return "현재 지원하는 질문은 관측 저수율 조회, 향후 30일 예측, 여러 저수지 단계 표, 물 운송 최적화입니다."


if __name__ == "__main__":
    try:
        chatbot = OhbongChatbot()
    except Exception as error:
        print(f"챗봇 초기화 실패: {error}")
        raise SystemExit(1)
    print("Ohbong 챗봇 CLI입니다. 종료하려면 exit 또는 quit을 입력하세요.")
    while True:
        user_input = input("사용자: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        result = chatbot.handle_message(user_input, use_gemini=False)
        print(f"챗봇: {result['answer']}")
        if result.get("table") is not None:
            print(result["table"].to_string(index=False))
