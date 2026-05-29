"""Gemini-based intent and slot parser for the Ohbong chatbot.

Gemini is used only to convert a user message into a structured execution
plan. Prediction, observation lookup, and optimization are still performed by
local Python code.
"""

from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - runtime fallback for minimal environments
    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
        return False


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "gemini-2.0-flash"

VALID_INTENTS = {
    "observation_lookup",
    "prediction",
    "optimization",
    "pipeline",
    "comparison_or_table",
    "help",
    "unknown",
}

DEFAULT_PLAN: dict[str, Any] = {
    "intent": "unknown",
    "target_reservoirs": ["오봉"],
    "target_date": None,
    "target_month": None,
    "forecast_horizon": 30,
    "target_rate": 50,
    "safety_rate": 40,
    "max_distance_km": 100,
    "allowed_region": None,
    "top_k": None,
    "need_table": False,
    "scenario": {
        "rainfall_multiplier": 1.0,
        "spi3_delta": 0.0,
        "spi6_delta": 0.0,
        "temperature_delta": 0.0,
    },
}


def _load_settings() -> tuple[str | None, str]:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL") or DEFAULT_MODEL
    return api_key, model_name


def _strip_code_fence(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    return text


def _coerce_float(value: Any, default: float) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _coerce_int_or_none(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        number = int(float(value))
        return number if number > 0 else None
    except (TypeError, ValueError):
        return None


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = _strip_code_fence(text)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Gemini intent parser response is not a JSON object.")
    return parsed


def normalize_intent_plan(
    raw_plan: dict[str, Any] | None,
    fallback_plan: dict[str, Any] | None = None,
    known_reservoirs: list[str] | None = None,
) -> dict[str, Any]:
    """Merge parser output with defaults and keep only safe, known fields."""
    base = deepcopy(DEFAULT_PLAN)
    if fallback_plan:
        for key, value in fallback_plan.items():
            if key == "scenario" and isinstance(value, dict):
                base["scenario"].update(value)
            elif key in base and value is not None:
                base[key] = value

    raw = raw_plan or {}
    plan = deepcopy(base)

    intent = raw.get("intent", base["intent"])
    plan["intent"] = intent if intent in VALID_INTENTS else base["intent"]

    reservoirs = raw.get("target_reservoirs", base["target_reservoirs"])
    if isinstance(reservoirs, str):
        reservoirs = [reservoirs]
    if not isinstance(reservoirs, list):
        reservoirs = base["target_reservoirs"]
    reservoirs = [str(name).strip() for name in reservoirs if str(name).strip()]
    if known_reservoirs:
        known = set(known_reservoirs)
        reservoirs = [name for name in reservoirs if name in known]
    plan["target_reservoirs"] = reservoirs or base["target_reservoirs"]

    for key in ("target_date", "target_month"):
        plan[key] = raw.get(key) if raw.get(key) is not None else base.get(key)

    plan["forecast_horizon"] = int(_coerce_float(raw.get("forecast_horizon"), base["forecast_horizon"]))
    plan["target_rate"] = _coerce_float(raw.get("target_rate"), base["target_rate"])
    plan["safety_rate"] = _coerce_float(raw.get("safety_rate"), base["safety_rate"])
    plan["max_distance_km"] = _coerce_float(raw.get("max_distance_km"), base["max_distance_km"])

    allowed_region = raw.get("allowed_region", base.get("allowed_region"))
    if isinstance(allowed_region, str):
        allowed_region = allowed_region.strip() or None
    else:
        allowed_region = None
    plan["allowed_region"] = allowed_region

    top_k = _coerce_int_or_none(raw.get("top_k", base.get("top_k")))
    plan["top_k"] = min(top_k, 50) if top_k is not None else None
    plan["need_table"] = bool(raw.get("need_table", base.get("need_table"))) or plan["top_k"] is not None

    scenario = raw.get("scenario") if isinstance(raw.get("scenario"), dict) else {}
    plan["scenario"] = {
        "rainfall_multiplier": _coerce_float(
            scenario.get("rainfall_multiplier"), base["scenario"]["rainfall_multiplier"]
        ),
        "spi3_delta": _coerce_float(scenario.get("spi3_delta"), base["scenario"]["spi3_delta"]),
        "spi6_delta": _coerce_float(scenario.get("spi6_delta"), base["scenario"]["spi6_delta"]),
        "temperature_delta": _coerce_float(
            scenario.get("temperature_delta"), base["scenario"]["temperature_delta"]
        ),
    }
    return plan


def parse_intent_with_gemini(
    message: str,
    fallback_plan: dict[str, Any] | None = None,
    known_reservoirs: list[str] | None = None,
    latest_date: str | None = None,
) -> dict[str, Any]:
    """Return a normalized intent plan. Falls back quietly when Gemini is unavailable."""
    api_key, model_name = _load_settings()
    fallback = normalize_intent_plan(fallback_plan, known_reservoirs=known_reservoirs)
    if not api_key:
        return fallback

    known_text = ", ".join(known_reservoirs or [])
    prompt = f"""
너는 저수지 의사결정 챗봇의 intent/slot parser다.
반드시 JSON 객체 하나만 반환하라. 설명 문장, markdown, 코드블록은 금지한다.
계산은 하지 말고 사용자 질문의 의도와 조건만 구조화하라.

반환 JSON schema:
{{
  "intent": "observation_lookup | prediction | optimization | pipeline | comparison_or_table | help | unknown",
  "target_reservoirs": ["오봉"],
  "target_date": null,
  "target_month": null,
  "forecast_horizon": 30,
  "target_rate": 50,
  "safety_rate": 40,
  "max_distance_km": 100,
  "allowed_region": null,
  "top_k": null,
  "need_table": false,
  "scenario": {{
    "rainfall_multiplier": 1.0,
    "spi3_delta": 0.0,
    "spi6_delta": 0.0,
    "temperature_delta": 0.0
  }}
}}

분류 규칙:
- 관측 저수율, 현재, 오늘, 특정 날짜의 저수율을 묻는 질문은 observation_lookup.
- 예측, 다음 달, 향후, 앞으로, 6월 예측은 prediction.
- 어디서, 얼마 가져와, 운송, 최적화, 공급은 optimization.
- 예측과 최적화를 동시에 요청하면 pipeline.
- 여러 저수지 비교, 목록, 표, 22개 저수지 결과는 comparison_or_table.
- "최종 후보 5개", "5개만", "상위 3개"처럼 개수를 말하면 top_k에 숫자를 넣고 need_table=true.
- "강릉시 저수지", "강릉 소재", "강릉에 있는", "강릉시로만"은 allowed_region="강릉시".
- "타지역 포함", "섞어서", "전체 후보", "강릉시가 아닌 곳도"는 allowed_region=null.
- "비 안", "무강수", "강수 없음"은 rainfall_multiplier=0.0.
- "강수 절반", "비 절반"은 rainfall_multiplier=0.5.
- SPI3/SPI6/기온/목표 저수율/최대 거리 숫자는 해당 필드에 넣어라.
- target_reservoirs는 가능한 한 알려진 저수지명만 사용하라. 없으면 ["오봉"].
- "오늘" 또는 "현재" 날짜는 최신 데이터 날짜를 target_date로 둔다.

최신 데이터 날짜: {latest_date or "unknown"}
알려진 저수지: {known_text}
사용자 질문: {message}
""".strip()

    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model=model_name, contents=prompt)
        text = getattr(response, "text", "") or ""
        raw_plan = _extract_json_object(text)
        return normalize_intent_plan(raw_plan, fallback_plan=fallback, known_reservoirs=known_reservoirs)
    except Exception:
        return fallback


if __name__ == "__main__":
    sample = "위험하면 어디서 얼마나 가져와야 해? 강릉시 저수지로만 5개 알려줘"
    print(json.dumps(parse_intent_with_gemini(sample), ensure_ascii=False, indent=2))
