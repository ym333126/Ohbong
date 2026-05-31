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

PREDICTION_CACHE_DIR = PROJECT_ROOT / "models"
BASE_DAILY_CACHE_PATH = PREDICTION_CACHE_DIR / "base_seq2seq_daily_forecast_cache.csv"
BASE_SUMMARY_CACHE_PATH = PREDICTION_CACHE_DIR / "base_seq2seq_summary_cache.csv"


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
        self.base_daily_forecast: pd.DataFrame | None = None
        self.base_prediction_summary: pd.DataFrame | None = None
        self.scenario_prediction_cache: dict[tuple[Any, ...], tuple[pd.DataFrame, pd.DataFrame]] = {}
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

        self._load_base_prediction_cache()

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

    def _latest_master_date(self) -> str | None:
        if self.master_df is None or DATE_COL not in self.master_df.columns:
            return None
        latest = pd.to_datetime(self.master_df[DATE_COL], errors="coerce").max()
        return None if pd.isna(latest) else latest.date().isoformat()

    def _load_base_prediction_cache(self) -> None:
        if not BASE_DAILY_CACHE_PATH.exists() or not BASE_SUMMARY_CACHE_PATH.exists():
            return
        try:
            daily = pd.read_csv(BASE_DAILY_CACHE_PATH)
            summary = pd.read_csv(BASE_SUMMARY_CACHE_PATH)
            latest_date = self._latest_master_date()
            cached_date = str(daily["as_of_date"].iloc[0]) if "as_of_date" in daily.columns and not daily.empty else None
            if latest_date and cached_date != latest_date:
                return
            if daily.empty or summary.empty:
                return
            self.base_daily_forecast = daily
            self.base_prediction_summary = summary
            self.last_daily_forecast = daily
            self.last_prediction_summary = summary
        except Exception as exc:
            print(f"[안내] 기본 예측 캐시를 읽지 못했습니다. 새로 계산합니다: {exc}")

    def _save_base_prediction_cache(self) -> None:
        if self.base_daily_forecast is None or self.base_prediction_summary is None:
            return
        try:
            PREDICTION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            self.base_daily_forecast.to_csv(BASE_DAILY_CACHE_PATH, index=False, encoding="utf-8-sig")
            self.base_prediction_summary.to_csv(BASE_SUMMARY_CACHE_PATH, index=False, encoding="utf-8-sig")
        except Exception as exc:
            print(f"[안내] 기본 예측 캐시 저장에 실패했습니다: {exc}")

    def classify_intent(self, message: str) -> str:
        text = message.strip().lower()
        if not text:
            return "unknown"
        if any(keyword in text for keyword in ["도움", "사용법", "help", "예시", "뭘 물어", "무엇을 물어"]):
            return "help"
        optimization_keywords = ["어디서", "얼마", "얼만큼", "가져와", "가져온", "후보", "운송", "최적화", "공급", "회복", "물량"]
        pipeline_keywords = ["최적화까지", "운송 계획까지", "예측하고", "예측 후", "위험하면", "필요한지", "필요한 지"]
        prediction_keywords = ["예측", "다음달", "다음 달", "향후", "앞으로", "30일", "한 달", "물 부족", "최저", "위험", "어떻", "어떤지"]
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
        return bool(re.search(r"\d{1,2}\s*월", text)) and any(keyword in text for keyword in ["예측", "어떻게", "어떻", "어떤지", "전망", "위험", "저수율", "강수", "장마"])

    @staticmethod
    def _requests_prediction(message: str) -> bool:
        text = message.strip().lower()
        return any(keyword in text for keyword in ["예측", "다음달", "다음 달", "향후", "앞으로", "30일", "한 달", "물 부족", "최저", "위험", "어떻", "어떤지"]) or OhbongChatbot._has_future_month_request(text)

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
            "rainfall_sum_30d": None,
            "spi3_delta": 0.0,
            "spi6_delta": 0.0,
            "temperature_delta": 0.0,
            "target_rate": None,
            "safety_rate": None,
        }
        if any(keyword in text for keyword in ["비 안", "비가 안", "무강수", "강수 없음"]):
            scenario["rainfall_multiplier"] = 0.0
        if any(keyword in text for keyword in ["강수 절반", "비 절반"]):
            scenario["rainfall_multiplier"] = 0.5
        rainfall_range = re.search(r"(?:강수량|강수|비|장마).*?(\d+(?:\.\d+)?)\s*(?:~|-|에서)\s*(\d+(?:\.\d+)?)\s*mm", text, flags=re.IGNORECASE)
        rainfall_single = re.search(r"(?:강수량|강수|비|장마).*?(\d+(?:\.\d+)?)\s*mm", text, flags=re.IGNORECASE)
        if rainfall_range:
            low, high = map(float, rainfall_range.groups())
            scenario["rainfall_sum_30d"] = (low + high) / 2.0
        elif rainfall_single:
            scenario["rainfall_sum_30d"] = float(rainfall_single.group(1))
        spi3_match = re.search(r"SPI3\s*(?:가|은|는)?\s*([+-]?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
        if spi3_match:
            scenario["spi3_delta"] = float(spi3_match.group(1))
        spi6_match = re.search(r"SPI6\s*(?:가|은|는)?\s*([+-]?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
        if spi6_match:
            scenario["spi6_delta"] = float(spi6_match.group(1))
        temp_match = re.search(r"(?:기온|온도).*?([+-]?\d+(?:\.\d+)?)\s*도", text)
        if temp_match:
            scenario["temperature_delta"] = float(temp_match.group(1))
        target_match = re.search(r"목표\s*저수율\s*(\d+(?:\.\d+)?)", text)
        if target_match:
            scenario["target_rate"] = float(target_match.group(1))
        safety_match = re.search(r"안전\s*저수율\s*(\d+(?:\.\d+)?)", text)
        if safety_match:
            scenario["safety_rate"] = float(safety_match.group(1))
        return scenario

    @staticmethod
    def parse_candidate_limit_optional(message: str) -> Optional[int]:
        for pattern in [r"(?:후보|상위|최종)\s*(\d+)\s*개", r"(\d+)\s*개\s*(?:후보|까지|만)"]:
            match = re.search(pattern, message)
            if match:
                return max(1, min(int(match.group(1)), 20))
        return None

    @staticmethod
    def parse_candidate_limit(message: str, default: int = 5) -> int:
        parsed = OhbongChatbot.parse_candidate_limit_optional(message)
        if parsed is not None:
            return parsed
        return default

    @staticmethod
    def parse_region_filter(message: str) -> Optional[str]:
        text = message.strip()
        asks_for_gangneung = any(keyword in text for keyword in ["강릉시", "강릉 소재", "강릉 저수지", "강릉에 있는", "강릉"])
        excludes_gangneung = any(
            keyword in text
            for keyword in [
                "강릉시가 아닌",
                "강릉시 아닌",
                "강릉시 이외",
                "강릉시 외",
                "강릉시 제외",
                "강릉이 아닌",
                "강릉 아닌",
                "강릉 이외",
                "강릉 외",
                "강릉 제외",
                "타지역",
                "타 지역",
                "다른 지역",
                "섞어서",
                "포함해서",
            ]
        )
        if asks_for_gangneung and not excludes_gangneung:
            return "강릉시"
        return None

    @staticmethod
    def _mentions_non_gangneung(message: str) -> bool:
        text = message.strip()
        return any(
            keyword in text
            for keyword in [
                "강릉시가 아닌",
                "강릉시 아닌",
                "강릉시 이외",
                "강릉시 외",
                "강릉시 제외",
                "강릉이 아닌",
                "강릉 아닌",
                "강릉 이외",
                "강릉 외",
                "강릉 제외",
                "타지역",
                "다른 지역",
                "타 지역",
                "아닌 곳",
                "아닌곳",
            ]
        )

    @staticmethod
    def _mentions_all_regions(message: str) -> bool:
        text = message.strip()
        return any(
            keyword in text
            for keyword in [
                "섞어서",
                "포함해서",
                "전체",
                "모든 지역",
                "전부",
                "타지역도",
                "타 지역도",
                "타지역 포함",
                "타 지역 포함",
                "다른 지역도",
                "다른 지역 포함",
            ]
        )

    @staticmethod
    def _region_scope_from_message(message: str) -> Optional[str]:
        """Return explicit region scope from the current message only.

        Values:
        - all: include Gangneung and non-Gangneung candidates
        - only_gangneung: only Gangneung candidates
        - exclude_gangneung: non-Gangneung candidates only
        """
        text = message.strip()
        if OhbongChatbot._mentions_all_regions(text):
            return "all"
        if OhbongChatbot._mentions_non_gangneung(text):
            return "exclude_gangneung"
        if any(keyword in text for keyword in ["강릉시로만", "강릉으로만", "강릉시 저수지", "강릉 소재", "강릉에 있는"]):
            return "only_gangneung"
        return None

    @staticmethod
    def is_optimization_followup(message: str) -> bool:
        text = message.strip()
        if OhbongChatbot._requests_prediction(text):
            return False
        followup_keywords = ["그럼", "그러면", "이번엔", "이번에는", "만약", "라면", "다면", "이외", "외", "아닌 곳", "아닌곳", "섞어서", "포함해서", "제외", "말고"]
        optimization_context_keywords = ["가져", "후보", "저수지", "강릉", "타지역", "지역", "운송", "얼마"]
        return any(keyword in text for keyword in followup_keywords) and any(keyword in text for keyword in optimization_context_keywords)

    def is_prediction_explanation_followup(self, message: str) -> bool:
        text = message.strip()
        if self.last_daily_forecast is None:
            return False
        explanation_keywords = ["왜", "이유", "낮아", "낮아졌", "떨어", "감소", "차이", "비해", "대비", "설명"]
        prediction_context_keywords = ["6월", "예측", "저수율", "5월", "이전", "아까", "방금", "결과"]
        return any(keyword in text for keyword in explanation_keywords) and any(keyword in text for keyword in prediction_context_keywords)

    @staticmethod
    def resolve_allowed_region(message: str, previous_region: Optional[str] = None) -> Optional[str]:
        text = message.strip()
        if OhbongChatbot._mentions_non_gangneung(text) or OhbongChatbot._mentions_all_regions(text):
            return None
        if any(keyword in text for keyword in ["강릉시", "강릉 소재", "강릉 저수지", "강릉에 있는", "강릉"]):
            return "강릉시"
        return previous_region

    @staticmethod
    def resolve_excluded_region(message: str) -> Optional[str]:
        if OhbongChatbot._mentions_all_regions(message):
            return None
        if OhbongChatbot._mentions_non_gangneung(message):
            return "강릉시"
        return None

    @staticmethod
    def _is_empty_slot(value: Any) -> bool:
        return value is None or value == [] or value == "unknown"

    @staticmethod
    def _is_default_scenario(scenario: Optional[dict[str, Any]]) -> bool:
        scenario = scenario or {}
        return (
            scenario.get("rainfall_sum_30d") is None
            and float(scenario.get("rainfall_multiplier", 1.0) or 1.0) == 1.0
            and float(scenario.get("spi3_delta", 0.0) or 0.0) == 0.0
            and float(scenario.get("spi6_delta", 0.0) or 0.0) == 0.0
            and float(scenario.get("temperature_delta", 0.0) or 0.0) == 0.0
        )

    @staticmethod
    def _requests_prediction_and_optimization(message: str) -> bool:
        text = message.strip().lower()
        has_prediction_context = OhbongChatbot._requests_prediction(text) or any(
            keyword in text for keyword in ["강수", "장마", "spi", "저수량", "저수율"]
        )
        has_optimization_context = any(
            keyword in text for keyword in ["운송", "최적화", "가져와", "공급", "회복", "필요한지", "필요한 지"]
        )
        return has_prediction_context and has_optimization_context

    def build_contextual_execution_plan(
        self,
        message: str,
        parsed_plan: dict[str, Any],
        target_rate: float,
        safety_rate: float,
        max_distance_km: float,
    ) -> dict[str, Any]:
        """Resolve LLM/rule output into an executable plan with conversation state.

        The important rule is that only compatible follow-up questions inherit
        previous slots. Explicit phrases in the current message always override
        stale context.
        """
        plan = dict(parsed_plan)
        current_intent = plan.get("intent", "unknown")
        optimization_followup = self.last_optimization_plan is not None and self.is_optimization_followup(message)
        prediction_explanation_followup = self.is_prediction_explanation_followup(message)

        if self._requests_prediction_and_optimization(message):
            plan["intent"] = "pipeline"
            optimization_followup = False
            prediction_explanation_followup = False

        if optimization_followup:
            previous = dict(self.last_optimization_plan or {})
            for key, value in plan.items():
                if key == "scenario" and self._is_default_scenario(value):
                    continue
                if not self._is_empty_slot(value):
                    previous[key] = value
            previous["intent"] = "optimization"
            plan = previous
        elif prediction_explanation_followup:
            previous = dict(self.last_intent_plan or {})
            for key, value in plan.items():
                if key == "scenario" and self._is_default_scenario(value):
                    continue
                if not self._is_empty_slot(value):
                    previous[key] = value
            previous["intent"] = "prediction_explanation"
            plan = previous
        elif current_intent == "unknown" and self.last_intent_plan:
            plan["intent"] = "unknown"

        region_scope = self._region_scope_from_message(message)
        plan["region_scope"] = region_scope
        if region_scope == "all":
            plan["allowed_region"] = None
            plan["excluded_region"] = None
        elif region_scope == "only_gangneung":
            plan["allowed_region"] = "강릉시"
            plan["excluded_region"] = None
        elif region_scope == "exclude_gangneung":
            plan["allowed_region"] = None
            plan["excluded_region"] = "강릉시"
        elif optimization_followup:
            plan["allowed_region"] = plan.get("allowed_region")
            plan["excluded_region"] = plan.get("excluded_region")
        else:
            plan["allowed_region"] = None
            plan["excluded_region"] = None

        explicit_top_k = self.parse_candidate_limit_optional(message)
        if explicit_top_k is not None:
            plan["top_k"] = explicit_top_k
        elif optimization_followup:
            plan["top_k"] = plan.get("top_k") or (self.last_optimization_plan or {}).get("top_k")

        if plan.get("target_rate") is None:
            plan["target_rate"] = target_rate
        if plan.get("safety_rate") is None:
            plan["safety_rate"] = safety_rate
        if plan.get("max_distance_km") is None:
            plan["max_distance_km"] = max_distance_km

        scenario = plan.get("scenario") or {}
        if not prediction_explanation_followup and not optimization_followup and self._is_default_scenario(scenario):
            plan["scenario"] = {
                "rainfall_multiplier": 1.0,
                "rainfall_sum_30d": None,
                "spi3_delta": 0.0,
                "spi6_delta": 0.0,
                "temperature_delta": 0.0,
            }

        plan["conversation_mode"] = (
            "optimization_followup"
            if optimization_followup
            else "prediction_explanation_followup"
            if prediction_explanation_followup
            else "new_request"
        )
        return plan

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
            "top_k": self.parse_candidate_limit_optional(message),
            "need_table": self.classify_intent(message) in {"comparison_or_table", "optimization", "pipeline"},
            "scenario": {
                "rainfall_multiplier": scenario.get("rainfall_multiplier", 1.0),
                "rainfall_sum_30d": scenario.get("rainfall_sum_30d"),
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
        if target_date is None or pd.isna(target_date):
            target_date = pd.Timestamp(self.master_df[DATE_COL].max())
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

    def explain_prediction_change(self, message: str) -> tuple[str, pd.DataFrame]:
        if self.master_df is None or self.last_daily_forecast is None:
            return "직전 예측 결과가 없어 설명할 수 없습니다. 먼저 '6월 오봉 저수지 예측해줘'처럼 예측을 요청해 주세요.", pd.DataFrame()

        reservoirs = self.extract_reservoir_names(message)
        reservoir_name = reservoirs[0] if reservoirs else OBONG_NAME
        period_info = self.parse_date_or_period(message)
        if period_info.get("period_type") != "month":
            last_month = None
            if self.last_intent_plan and self.last_intent_plan.get("target_month"):
                last_month = self.last_intent_plan.get("target_month")
            period_info = {"month": int(last_month or 6), "period_type": "month", "target_date": None}

        prediction_table, period_message = self.build_prediction_table(self.last_daily_forecast, [reservoir_name], period_info)
        if prediction_table.empty:
            return f"{reservoir_name} 저수지의 직전 예측 결과를 찾지 못했습니다.", pd.DataFrame()

        latest_date = pd.Timestamp(self.master_df[DATE_COL].max())
        reservoir_df = self.master_df[self.master_df[RESERVOIR_COL] == reservoir_name].sort_values(DATE_COL)
        latest_row_df = reservoir_df[reservoir_df[DATE_COL] == latest_date]
        latest_row = latest_row_df.iloc[-1] if not latest_row_df.empty else reservoir_df.iloc[-1]
        observed_rate = pd.to_numeric(latest_row.get(OBSERVED_RATE_COL), errors="coerce")
        observed_ema = pd.to_numeric(latest_row.get(EMA_RATE_COL), errors="coerce")
        observed_date = pd.Timestamp(latest_row.get(DATE_COL))

        row = prediction_table.iloc[0]
        predicted_min = float(row["예상 최저 저수율(%)"])
        predicted_avg = float(row["평균 예측 저수율(%)"])
        min_date = row["최저 예상일"]
        stage = row["단계"]

        compare_rate = observed_ema if pd.notna(observed_ema) else observed_rate
        drop_to_min = compare_rate - predicted_min if pd.notna(compare_rate) else None
        drop_to_avg = compare_rate - predicted_avg if pd.notna(compare_rate) else None

        explanation_rows = [
            {
                "항목": "기준 관측일",
                "값": _fmt_date(observed_date),
                "설명": "예측을 시작하기 직전 DB의 최신 관측일입니다.",
            },
            {
                "항목": "기준 관측 저수율",
                "값": _fmt_rate(observed_rate),
                "설명": "실제 관측 저수율입니다.",
            },
            {
                "항목": "기준 5일 EMA 저수율",
                "값": _fmt_rate(observed_ema),
                "설명": "모델이 참고하는 완만한 저수율 흐름입니다.",
            },
            {
                "항목": "예측 기간 평균",
                "값": f"{predicted_avg:.2f}%",
                "설명": "해당 기간의 일별 예측 평균입니다.",
            },
            {
                "항목": "예상 최저 저수율",
                "값": f"{predicted_min:.2f}%",
                "설명": f"{min_date}에 가장 낮을 것으로 예측했습니다.",
            },
            {
                "항목": "저수율 단계",
                "값": str(stage),
                "설명": str(row["판단 기준"]),
            },
        ]
        table = pd.DataFrame(explanation_rows)

        drop_text = ""
        if drop_to_min is not None and pd.notna(drop_to_min):
            drop_text = f" 기준 5일 EMA 저수율 {_fmt_rate(compare_rate)}와 비교하면 예상 최저치는 약 {drop_to_min:.2f}%p 낮습니다."
        avg_text = ""
        if drop_to_avg is not None and pd.notna(drop_to_avg):
            avg_text = f" 예측 기간 평균도 기준값보다 약 {drop_to_avg:.2f}%p 낮습니다."

        answer = (
            f"후속 질문으로 이해했습니다. {reservoir_name} 저수지의 6월 예측이 5월 말 관측값보다 낮게 보이는 이유는, "
            f"seq2seq 모델이 기준일 이후 30일 일별 흐름을 하락 방향으로 예측했기 때문입니다.{drop_text}{avg_text}\n\n"
            f"다만 이 모델은 원인을 직접 분해하는 인과 모델은 아닙니다. 그래서 '강수량 때문에', 'SPI 때문에'라고 단정하기보다는 "
            f"최근 저수율 흐름, 기상/SPI 입력 패턴, 계절 패턴을 함께 반영한 결과로 해석하는 것이 안전합니다."
        )
        return answer, table

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
        if self.base_daily_forecast is None or self.base_prediction_summary is None:
            daily_forecast = predict_seq2seq_forecasts(self.master_df)
            summary = summarize_min_rates(daily_forecast)
            if daily_forecast.empty or summary.empty:
                raise RuntimeError("seq2seq 예측 결과가 비어 있습니다.")
            self.base_daily_forecast = daily_forecast
            self.base_prediction_summary = summary
            self._save_base_prediction_cache()
        self.last_daily_forecast = self.base_daily_forecast
        self.last_prediction_summary = self.base_prediction_summary
        return self.base_daily_forecast.copy(), self.base_prediction_summary.copy()

    @staticmethod
    def _has_active_scenario(scenario: Optional[dict[str, Any]]) -> bool:
        if not scenario:
            return False
        return (
            scenario.get("rainfall_sum_30d") is not None
            or float(scenario.get("rainfall_multiplier", 1.0) or 1.0) != 1.0
            or float(scenario.get("spi3_delta", 0.0) or 0.0) != 0.0
            or float(scenario.get("spi6_delta", 0.0) or 0.0) != 0.0
            or float(scenario.get("temperature_delta", 0.0) or 0.0) != 0.0
        )

    def describe_scenario(self, scenario: Optional[dict[str, Any]]) -> str:
        if not self._has_active_scenario(scenario):
            return "기본 시나리오"
        parts = []
        if scenario.get("rainfall_sum_30d") is not None:
            parts.append(f"향후 30일 총강수량 {float(scenario['rainfall_sum_30d']):.0f}mm")
        elif float(scenario.get("rainfall_multiplier", 1.0) or 1.0) != 1.0:
            parts.append(f"강수량 배율 {float(scenario.get('rainfall_multiplier', 1.0)):.2f}")
        if float(scenario.get("spi3_delta", 0.0) or 0.0) != 0.0:
            parts.append(f"SPI3 {float(scenario.get('spi3_delta')):+.2f}")
        if float(scenario.get("spi6_delta", 0.0) or 0.0) != 0.0:
            parts.append(f"SPI6 {float(scenario.get('spi6_delta')):+.2f}")
        if float(scenario.get("temperature_delta", 0.0) or 0.0) != 0.0:
            parts.append(f"기온 {float(scenario.get('temperature_delta')):+.1f}도")
        return ", ".join(parts)

    def scenario_cache_key(self, scenario: Optional[dict[str, Any]]) -> tuple[Any, ...]:
        scenario = scenario or {}
        return (
            None if scenario.get("rainfall_sum_30d") is None else round(float(scenario.get("rainfall_sum_30d")), 3),
            round(float(scenario.get("rainfall_multiplier", 1.0) or 1.0), 3),
            round(float(scenario.get("spi3_delta", 0.0) or 0.0), 3),
            round(float(scenario.get("spi6_delta", 0.0) or 0.0), 3),
            round(float(scenario.get("temperature_delta", 0.0) or 0.0), 3),
        )

    def get_prediction_for_scenario(self, scenario: Optional[dict[str, Any]] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        base_daily, base_summary = self._ensure_prediction()
        if not self._has_active_scenario(scenario):
            self.last_daily_forecast = base_daily
            self.last_prediction_summary = base_summary
            return base_daily, base_summary

        key = self.scenario_cache_key(scenario)
        if key not in self.scenario_prediction_cache:
            if self.master_df is None:
                raise RuntimeError("master_dataset이 로드되지 않았습니다.")
            scenario_daily = predict_seq2seq_forecasts(self.master_df, scenario=scenario)
            scenario_summary = summarize_min_rates(scenario_daily)
            self.scenario_prediction_cache[key] = (scenario_daily, scenario_summary)

        daily_forecast, summary = self.scenario_prediction_cache[key]
        self.last_daily_forecast = daily_forecast
        self.last_prediction_summary = summary
        return daily_forecast.copy(), summary.copy()

    def apply_scenario_adjustment(self, daily_forecast: pd.DataFrame, scenario: Optional[dict[str, Any]]) -> pd.DataFrame:
        """Apply an explicit, transparent scenario adjustment to seq2seq output.

        The current seq2seq artifact does not accept future weather variables.
        This post-processing is therefore a scenario sensitivity estimate, not a
        retrained model forecast.
        """
        if not self._has_active_scenario(scenario) or self.master_df is None or daily_forecast.empty:
            return daily_forecast

        adjusted = daily_forecast.copy()
        adjusted["base_predicted_rate"] = adjusted["predicted_rate"]
        adjusted["scenario_applied"] = True
        adjusted["scenario_note"] = self.describe_scenario(scenario)
        adjusted["forecast_date"] = pd.to_datetime(adjusted["forecast_date"], errors="coerce")
        max_day = max(float(adjusted["day_ahead"].max()), 1.0)

        for reservoir_name, index in adjusted.groupby(RESERVOIR_COL).groups.items():
            reservoir_history = self.master_df[self.master_df[RESERVOIR_COL] == reservoir_name].sort_values(DATE_COL)
            recent_rain = pd.to_numeric(reservoir_history["강수량"], errors="coerce").tail(30).fillna(0)
            recent_sum = float(recent_rain.sum()) if not recent_rain.empty else 0.0
            if scenario.get("rainfall_sum_30d") is not None:
                scenario_sum = float(scenario["rainfall_sum_30d"])
            else:
                scenario_sum = recent_sum * float(scenario.get("rainfall_multiplier", 1.0) or 1.0)

            rainfall_delta = scenario_sum - recent_sum
            rainfall_adjustment = max(-18.0, min(18.0, rainfall_delta * 0.035))
            spi_adjustment = (
                float(scenario.get("spi3_delta", 0.0) or 0.0) * 2.5
                + float(scenario.get("spi6_delta", 0.0) or 0.0) * 1.5
            )
            temperature_adjustment = -0.8 * float(scenario.get("temperature_delta", 0.0) or 0.0)
            total_adjustment = max(-20.0, min(20.0, rainfall_adjustment + spi_adjustment + temperature_adjustment))

            day_factor = adjusted.loc[index, "day_ahead"].astype(float) / max_day
            adjusted.loc[index, "predicted_rate"] = (
                adjusted.loc[index, "base_predicted_rate"].astype(float) + total_adjustment * day_factor
            ).clip(0, 100)
            adjusted.loc[index, "scenario_recent_rainfall_30d"] = recent_sum
            adjusted.loc[index, "scenario_rainfall_sum_30d"] = scenario_sum
            adjusted.loc[index, "scenario_rainfall_delta_30d"] = rainfall_delta
            adjusted.loc[index, "scenario_rainfall_adjustment"] = rainfall_adjustment
            adjusted.loc[index, "scenario_spi_adjustment"] = spi_adjustment
            adjusted.loc[index, "scenario_temperature_adjustment"] = temperature_adjustment
            adjusted.loc[index, "scenario_total_adjustment_at_30d"] = total_adjustment
            adjusted.loc[index, "scenario_applied_adjustment"] = total_adjustment * day_factor

            city = reservoir_history[CITY_COL].dropna().iloc[-1] if CITY_COL in reservoir_history.columns and not reservoir_history[CITY_COL].dropna().empty else None
            for row_index in index:
                normal_rate = get_normal_rate_for_date(
                    self.master_df,
                    reservoir_name,
                    adjusted.at[row_index, "forecast_date"],
                    target_col=EMA_RATE_COL,
                )
                stage = classify_reservoir_stage(adjusted.at[row_index, "predicted_rate"], city=city, normal_rate=normal_rate)
                adjusted.at[row_index, "stage_code"] = stage["stage_code"]
                adjusted.at[row_index, "stage_label"] = stage["stage_label"]
                adjusted.at[row_index, "stage_basis"] = stage["basis"]
                adjusted.at[row_index, "stage_basis_value"] = stage["basis_value"]
                adjusted.at[row_index, "normal_rate"] = stage["normal_rate"]
                adjusted.at[row_index, "normal_ratio"] = stage["normal_ratio"]
                adjusted.at[row_index, "threshold_description"] = stage["threshold_description"]
                adjusted.at[row_index, "risk_code"] = stage["stage_code"]
                adjusted.at[row_index, "risk_label"] = stage["stage_label"]

        adjusted["forecast_date"] = adjusted["forecast_date"].dt.date.astype(str)
        return adjusted

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
            return filtered, f"{as_of_date} 기준 {month}월 예측입니다. 적용 기간은 {_fmt_date(filtered['forecast_date'].min())}부터 {_fmt_date(filtered['forecast_date'].max())}까지입니다."
        return forecast_df, f"{as_of_date} 기준 향후 30일 예측입니다. 적용 기간은 {_fmt_date(start)}부터 {_fmt_date(end)}까지입니다."

    def build_prediction_table(
        self,
        daily_forecast: pd.DataFrame,
        reservoir_names: list[str],
        period_info: dict[str, Any],
        scenario: Optional[dict[str, Any]] = None,
    ) -> tuple[pd.DataFrame, str]:
        period_df, period_message = self._filter_forecast_period(daily_forecast, period_info)
        if period_df.empty:
            return pd.DataFrame(columns=PREDICTION_TABLE_COLUMNS), period_message
        selected = period_df[period_df[RESERVOIR_COL].isin(reservoir_names)].copy()
        base_period_df = pd.DataFrame()
        if self._has_active_scenario(scenario) and self.base_daily_forecast is not None:
            base_period_df, _ = self._filter_forecast_period(self.base_daily_forecast, period_info)
        rows = []
        for reservoir_name, group in selected.groupby(RESERVOIR_COL):
            min_idx = group["predicted_rate"].idxmin()
            min_row = group.loc[min_idx]
            base_min_rate = None
            if not base_period_df.empty:
                base_group = base_period_df[base_period_df[RESERVOIR_COL] == reservoir_name]
                if not base_group.empty:
                    base_min_rate = float(base_group.loc[base_group["predicted_rate"].idxmin(), "predicted_rate"])
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
                | (
                    {
                        "기본 예상 최저 저수율(%)": round(base_min_rate, 2) if base_min_rate is not None else None,
                        "시나리오 반영 변화(%p)": round(float(min_row.get("predicted_rate")) - base_min_rate, 2)
                        if base_min_rate is not None
                        else None,
                        "최근 30일 강수량(mm)": round(float(min_row.get("scenario_recent_rainfall_30d")), 1)
                        if pd.notna(min_row.get("scenario_recent_rainfall_30d"))
                        else None,
                        "가정 강수량(mm)": round(float(min_row.get("scenario_rainfall_sum_30d")), 1)
                        if pd.notna(min_row.get("scenario_rainfall_sum_30d"))
                        else None,
                        "시나리오": self.describe_scenario(scenario),
                    }
                    if self._has_active_scenario(scenario)
                    else {}
                )
            )
        columns = PREDICTION_TABLE_COLUMNS + (
            ["기본 예상 최저 저수율(%)", "시나리오 반영 변화(%p)", "최근 30일 강수량(mm)", "가정 강수량(mm)", "시나리오"]
            if self._has_active_scenario(scenario)
            else []
        )
        return pd.DataFrame(rows, columns=columns), period_message

    def format_prediction_answer(self, table: pd.DataFrame, period_message: str, scenario: Optional[dict[str, Any]] = None) -> str:
        if table.empty:
            return period_message
        if len(table) == 1:
            row = table.iloc[0]
            ratio_text = ""
            if pd.notna(row.get("평년 대비 비율(%)")):
                ratio_text = f" 평년 대비 비율은 {float(row.get('평년 대비 비율(%)')):.2f}%입니다."
            scenario_text = ""
            if self._has_active_scenario(scenario):
                base_min = row.get("기본 예상 최저 저수율(%)")
                scenario_change = row.get("시나리오 반영 변화(%p)")
                recent_rain = row.get("최근 30일 강수량(mm)")
                scenario_rain = row.get("가정 강수량(mm)")
                scenario_text = f"\n\n반영한 가정은 {self.describe_scenario(scenario)}입니다."
                if pd.notna(base_min):
                    scenario_text += f" 기본 조건에서는 최저 저수율이 {float(base_min):.2f}%로 예상됐습니다."
                if pd.notna(scenario_change):
                    direction = "상승" if float(scenario_change) >= 0 else "하락"
                    scenario_text += f" 사용자님의 가정을 반영하면 이보다 약 {abs(float(scenario_change)):.2f}%p {direction}했습니다."
                if pd.notna(recent_rain) and pd.notna(scenario_rain):
                    scenario_text += (
                        f" 이유는 가정 강수량 {float(scenario_rain):.0f}mm가 최근 30일 강수량 {float(recent_rain):.0f}mm보다 많아, "
                        "저수율 하락 폭이 완화되는 방향으로 계산됐기 때문입니다."
                    )
            return (
                f"{period_message} {row[RESERVOIR_COL]} 저수지의 예상 최저 저수율은 "
                f"{row['예상 최저 저수율(%)']:.2f}%이며, 최저 예상일은 {row['최저 예상일']}입니다. "
                f"{row['판단 기준']}으로는 {row['단계']} 단계입니다.{ratio_text}{scenario_text}"
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
        excluded_region: Optional[str] = None,
        region_scope: Optional[str] = None,
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
        if excluded_region:
            reservoir_info_df = reservoir_info_df[(reservoir_info_df[RESERVOIR_COL] == OBONG_NAME) | (reservoir_info_df[CITY_COL] != excluded_region)].copy()
        result = optimize_from_seq2seq_summary(
            min_rate_summary_df=prediction_summary_df,
            reservoir_info_df=reservoir_info_df,
            distance_df=self.distance_df,
            target_rate=target_rate,
            safety_rate=safety_rate,
            max_distance_km=max_distance_km,
        )
        result.update({
            "target_rate": target_rate,
            "safety_rate": safety_rate,
            "max_distance_km": max_distance_km,
            "target_reservoir": OBONG_NAME,
            "allowed_region": allowed_region,
            "excluded_region": excluded_region,
            "region_scope": region_scope,
        })
        obong_row = prediction_summary_df[prediction_summary_df[RESERVOIR_COL] == OBONG_NAME].iloc[0]
        result["obong_predicted_min_rate"] = float(obong_row["predicted_min_rate"])
        self.last_optimization_result = result
        return result

    def format_optimization_answer(self, result: dict, candidate_limit: int = 5) -> tuple[str, pd.DataFrame]:
        transfers = pd.DataFrame(result.get("transfers", []))
        candidates = pd.DataFrame(result.get("candidates", []))
        if not transfers.empty:
            transfers = transfers.sort_values(["transfer_amount_1000m3", "distance_km"], ascending=[False, True]).reset_index(drop=True)
        else:
            transfers = pd.DataFrame()

        if not candidates.empty:
            display_table = candidates.copy()
            display_table["transfer_amount_1000m3"] = 0.0
            display_table["remaining_rate_after_transfer"] = pd.NA
            display_table["remaining_stage_label"] = pd.NA
            display_table["selection_status"] = "검토 후보"
            if not transfers.empty:
                transfer_lookup = transfers.set_index("source_reservoir")
                for index, row in display_table.iterrows():
                    source = row["source_reservoir"]
                    if source in transfer_lookup.index:
                        transfer_row = transfer_lookup.loc[source]
                        display_table.at[index, "transfer_amount_1000m3"] = transfer_row.get("transfer_amount_1000m3", 0.0)
                        display_table.at[index, "remaining_rate_after_transfer"] = transfer_row.get("remaining_rate_after_transfer")
                        display_table.at[index, "remaining_stage_label"] = transfer_row.get("remaining_stage_label")
                        display_table.at[index, "selection_status"] = "운송 배정"
            display_table["_selected_sort"] = display_table["selection_status"].map({"운송 배정": 0, "검토 후보": 1}).fillna(1)
            display_table = display_table.sort_values(
                ["_selected_sort", "transfer_amount_1000m3", "distance_km"],
                ascending=[True, False, True],
            ).drop(columns=["_selected_sort"]).head(candidate_limit).reset_index(drop=True)
            if result.get("region_scope") == "all" and "region" in display_table.columns and "타지역" not in set(display_table["region"].astype(str)):
                non_gangneung = candidates[candidates["region"].astype(str) == "타지역"].sort_values(
                    ["distance_km", "available_supply_1000m3"],
                    ascending=[True, False],
                )
                if not non_gangneung.empty and len(display_table) >= candidate_limit:
                    extra = non_gangneung.head(1).copy()
                    extra["transfer_amount_1000m3"] = 0.0
                    extra["remaining_rate_after_transfer"] = pd.NA
                    extra["remaining_stage_label"] = pd.NA
                    extra["selection_status"] = "검토 후보"
                    display_table = pd.concat([display_table.iloc[:-1], extra[display_table.columns]], ignore_index=True)
        else:
            display_table = transfers.head(candidate_limit).copy() if not transfers.empty else pd.DataFrame()

        shortage = float(result.get("unmet_water_1000m3", result.get("shortage_1000m3", 0.0)) or 0.0)
        feasible = bool(result.get("feasible"))
        required_water = float(result.get("required_water_1000m3", 0.0) or 0.0)
        total_available = float(result.get("total_available_supply_1000m3", result.get("total_supplied_1000m3", 0.0)) or 0.0)
        total_supplied = float(result.get("total_supplied_1000m3", 0.0) or 0.0)
        predicted_rate = result.get("obong_predicted_min_rate")
        target_rate = float(result.get("target_rate", 50) or 50)
        if required_water <= 0:
            rate_text = f"예상 최저 저수율이 {float(predicted_rate):.2f}%로 " if predicted_rate is not None else ""
            answer = (
                f"물 운송은 필요하지 않습니다. {rate_text}목표 저수율 {target_rate:.0f}% 이상이므로 "
                "추가 공급 물량을 계산할 필요가 없습니다."
            )
            return answer, pd.DataFrame()
        region_text = f"{result.get('allowed_region')} 저수지로만 제한하면 " if result.get("allowed_region") else ""
        if result.get("excluded_region"):
            region_text = f"{result.get('excluded_region')}를 제외한 타 지역 후보만 보면 "
            candidate_scope = f"100km 이내 {result.get('excluded_region')} 외 후보 저수지"
        else:
            candidate_scope = f"100km 이내 {result.get('allowed_region')} 후보 저수지" if result.get("allowed_region") else "100km 이내 후보 저수지"
            if result.get("region_scope") == "all":
                region_text = "강릉시와 타지역을 함께 검토하면 "
            elif not result.get("allowed_region"):
                region_text = "지역 제한 없이 보면 "
        answer = (
            f"{region_text}오봉 저수지를 목표 저수율 {target_rate:.0f}%까지 회복하려면 "
            f"약 {required_water:.2f}천m3가 필요합니다. "
            f"현재 {candidate_scope}에서 정상 또는 관심 단계로 공급 가능한 총량은 약 {total_available:.2f}천m3이고, "
            f"최적화가 실제 배정한 물량은 약 {total_supplied:.2f}천m3입니다. "
        )
        answer += "따라서 현재 조건에서는 목표 달성이 가능합니다." if feasible else f"약 {shortage:.2f}천m3가 부족하여 현재 조건에서는 목표 달성이 어렵습니다."

        selected_count = int(len(transfers))
        candidate_count = int(result.get("candidate_count_100km", len(candidates) if not candidates.empty else selected_count) or 0)
        if candidate_count < candidate_limit:
            answer += f"\n\n요청하신 {candidate_limit}개보다 적은 {candidate_count}개만 표시되는 이유는, 현재 조건을 통과한 공급 후보가 {candidate_count}개뿐이기 때문입니다."
        elif selected_count < candidate_limit:
            answer += (
                f"\n\n검토 가능한 후보는 {candidate_count}개이며, 아래에는 요청하신 대로 상위 {candidate_limit}개를 표시합니다. "
                f"다만 실제 운송 배정은 필요 물량을 채우는 데 필요한 {selected_count}개 저수지에만 들어갔습니다."
            )
        if result.get("region_scope") == "all" and not transfers.empty and "region" in transfers.columns:
            selected_regions = set(transfers["region"].astype(str))
            candidate_regions = set(candidates["region"].astype(str)) if not candidates.empty and "region" in candidates.columns else set()
            if "타지역" in candidate_regions and "타지역" not in selected_regions:
                answer += " 타지역 후보도 함께 검토했지만, 현재 비용 최소화 결과에서는 필요 물량을 가까운 강릉시 후보만으로 채워 타지역에는 실제 배정되지 않았습니다."

        if not transfers.empty:
            lines = [
                f"{index + 1}. {row['source_reservoir']}에서 {row['transfer_amount_1000m3']:.2f}천m3 (거리 {row['distance_km']:.2f}km, 현재 단계 {row.get('stage_label')})"
                for index, row in transfers.head(candidate_limit).iterrows()
            ]
            answer += f"\n\n실제 운송 배정은 다음과 같습니다.\n" + "\n".join(lines)
        else:
            answer += "\n\n현재 조건에서 표시할 운송 후보가 없습니다."
        return answer, display_table

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
        parsed_plan = parse_intent_with_gemini(message, fallback_plan=fallback_plan, known_reservoirs=known_reservoirs, latest_date=latest_date)
        intent_plan = self.build_contextual_execution_plan(
            message=message,
            parsed_plan=parsed_plan,
            target_rate=target_rate,
            safety_rate=safety_rate,
            max_distance_km=max_distance_km,
        )
        intent = intent_plan["intent"]
        scenario = intent_plan.get("scenario") or {}
        period_info = self.apply_intent_plan_to_period(self.parse_date_or_period(message), intent_plan)
        reservoir_names = intent_plan.get("target_reservoirs") or [OBONG_NAME]
        candidate_limit = intent_plan.get("top_k") or self.parse_candidate_limit(message)
        target_rate = float(intent_plan.get("target_rate", target_rate))
        safety_rate = float(intent_plan.get("safety_rate", safety_rate))
        max_distance_km = float(intent_plan.get("max_distance_km", max_distance_km))
        allowed_region = intent_plan.get("allowed_region")
        excluded_region = intent_plan.get("excluded_region")
        intent_plan["allowed_region"] = allowed_region
        intent_plan["excluded_region"] = excluded_region

        try:
            if intent == "prediction_explanation":
                answer, table = self.explain_prediction_change(message)
                answer = self._apply_gemini(
                    answer,
                    intent,
                    message,
                    use_gemini,
                    prediction_result={"rows": table.to_dict("records")} if not table.empty else None,
                    table=table,
                )
                self.last_intent_plan = intent_plan
                return self._response(answer, intent, table if not table.empty else None, self.last_prediction_summary, None, intent_plan)
            if intent == "observation_lookup":
                table = self.run_observation_lookup(reservoir_names, period_info["target_date"])
                answer = self.format_observation_answer(table)
                self.last_intent_plan = intent_plan
                return self._response(answer, intent, table if len(table) > 1 or intent_plan.get("need_table") else None, intent_plan=intent_plan)
            if intent == "prediction":
                daily_forecast, summary = self.get_prediction_for_scenario(scenario)
                table, period_message = self.build_prediction_table(daily_forecast, reservoir_names, period_info, scenario=scenario)
                answer = self.format_prediction_answer(table, period_message, scenario=scenario)
                answer = self._apply_gemini(answer, intent, message, use_gemini, prediction_result={"rows": table.head(5).to_dict("records")}, table=table)
                self.last_intent_plan = intent_plan
                return self._response(answer, intent, table if intent_plan.get("need_table") or len(table) > 1 else None, summary, None, intent_plan)
            if intent == "comparison_or_table":
                if not self._requests_prediction(message):
                    table = self.run_observation_lookup(reservoir_names, period_info["target_date"])
                    answer = self.format_observation_answer(table)
                    self.last_intent_plan = intent_plan
                    return self._response(answer, "observation_lookup", table, None, None, intent_plan)
                daily_forecast, summary = self.get_prediction_for_scenario(scenario)
                if any(keyword in message for keyword in ["목록", "표", "22개", "전체", "단계"]):
                    table = self.build_stage_list_table(summary)
                    answer = "아래 표는 향후 30일 예측 기준 저수지 단계 목록입니다."
                    self.last_intent_plan = intent_plan
                    return self._response(answer, intent, table, summary, None, intent_plan)
                table, period_message = self.build_prediction_table(daily_forecast, reservoir_names, period_info, scenario=scenario)
                answer = self.format_prediction_answer(table, period_message, scenario=scenario)
                self.last_intent_plan = intent_plan
                return self._response(answer, intent, table, summary, None, intent_plan)
            if intent == "optimization":
                result = self.run_optimization(
                    target_rate=target_rate,
                    safety_rate=safety_rate,
                    max_distance_km=max_distance_km,
                    allowed_region=allowed_region,
                    excluded_region=excluded_region,
                    region_scope=intent_plan.get("region_scope"),
                )
                answer, table = self.format_optimization_answer(result, candidate_limit=candidate_limit)
                result["allocations_display"] = table.to_dict("records") if not table.empty else []
                self.last_intent_plan = intent_plan
                self.last_optimization_plan = intent_plan
                return self._response(answer, intent, table if not table.empty else None, None, result, intent_plan)
            if intent == "pipeline":
                daily_forecast, summary = self.get_prediction_for_scenario(scenario)
                prediction_table, period_message = self.build_prediction_table(daily_forecast, [OBONG_NAME], period_info, scenario=scenario)
                prediction_answer = self.format_prediction_answer(prediction_table, period_message, scenario=scenario)
                result = self.run_optimization(
                    summary,
                    target_rate=target_rate,
                    safety_rate=safety_rate,
                    max_distance_km=max_distance_km,
                    allowed_region=allowed_region,
                    excluded_region=excluded_region,
                    region_scope=intent_plan.get("region_scope"),
                )
                optimization_answer, transfer_table = self.format_optimization_answer(result, candidate_limit=candidate_limit)
                result["allocations_display"] = transfer_table.to_dict("records") if not transfer_table.empty else []
                answer = f"{prediction_answer}\n\n{optimization_answer}"
                self.last_intent_plan = intent_plan
                self.last_optimization_plan = intent_plan
                return self._response(answer, intent, transfer_table if not transfer_table.empty else None, summary, result, intent_plan)
            if intent == "help":
                return self._response(self.help_message(), intent, intent_plan=intent_plan)
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
