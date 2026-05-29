import json
import os
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "gemini-2.0-flash"


def _load_gemini_settings() -> tuple[Optional[str], str]:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL") or DEFAULT_MODEL
    return api_key, model_name


def is_gemini_available() -> bool:
    api_key, _ = _load_gemini_settings()
    return bool(api_key)


def _compact_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


def format_answer_with_gemini(
    user_message: str,
    intent: str,
    prediction_result: Optional[dict] = None,
    optimization_result: Optional[dict] = None,
    observation_result: Optional[dict] = None,
    table_preview: Optional[list[dict]] = None,
    fallback_answer: Optional[str] = None,
) -> str:
    api_key, model_name = _load_gemini_settings()
    if not api_key:
        return fallback_answer or ""

    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        prompt = f"""
너는 오봉 저수지 관리자를 돕는 의사결정 지원 챗봇이다.

반드시 지킬 규칙:
- 사용자가 물어본 내용에만 답하라.
- 관측 저수율만 물었으면 예측이나 최적화 이야기를 하지 마라.
- 예측만 물었으면 운송 최적화 이야기를 하지 마라.
- 최적화는 사용자가 명시적으로 요청했을 때만 설명하라.
- 제공된 수치를 바꾸거나 새로 계산하지 마라.
- 없는 내용을 추측하지 마라.
- 답변은 짧고 명확하게 작성하라.
- 기존의 "40% 미만 위험, 40~60 주의, 60 이상 안정" 기준은 절대 사용하지 마라.
- 강릉시 저수지는 저수율 자체 기준으로 정상/관심/주의/경계/심각을 판단한다.
- 강릉시 외 저수지는 평년 저수율 대비 비율 기준으로 정상/관심/주의/경계/심각을 판단한다.
- 제공된 stage_label 또는 단계 값을 그대로 사용하라.
- 임의로 위험/주의/안정으로 다시 분류하지 마라.
- predicted_min_rate는 "예상 최저 저수율"이라고 표현하라.
- feasible=False는 "현재 조건에서는 목표 달성이 어렵습니다"라고 표현하라.
- feasible=True는 "현재 조건에서는 목표 달성이 가능합니다"라고 표현하라.

사용자 질문:
{user_message}

분류된 의도:
{intent}

관측값 조회 결과:
{_compact_json(observation_result or {})}

예측 결과:
{_compact_json(prediction_result or {})}

최적화 결과:
{_compact_json(optimization_result or {})}

표 미리보기:
{_compact_json(table_preview or [])}

기본 답변:
{fallback_answer or ""}
"""
        response = client.models.generate_content(model=model_name, contents=prompt)
        text = getattr(response, "text", None)
        return text.strip() if text else (fallback_answer or "")
    except Exception:
        return fallback_answer or ""
