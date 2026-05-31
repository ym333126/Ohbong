import json
import os
from pathlib import Path
from typing import Any, Optional

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - keeps non-Gemini fallback usable
    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
        return False


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
- 답변은 짧고 친절하게 작성하라. 기본은 3~5문장 안으로 끝내라.
- 내부 구현, 모델 구조, 후처리, 민감도 추정 같은 기술 설명은 사용자가 명시적으로 묻지 않으면 말하지 마라.
- 사용자가 예측만 물으면 운송 방안 안내 문구를 덧붙이지 마라.
- "왜", "이유", "낮아졌지" 같은 후속 질문은 직전 예측 결과에 대한 설명으로 답하라.
- 후속 설명에서는 제공된 기준 관측값과 예측값의 차이를 중심으로 말하고, 제공되지 않은 원인은 단정하지 마라.
- 시나리오가 제공되면 "반영한 가정"과 "기본 예측 대비 변화"를 구분해서 설명하라.
- 시나리오 반영 변화, 최근 30일 강수량, 가정 강수량이 제공되면 그 수치를 이용해 왜 결과가 바뀌었는지 설명하라.
- 시나리오 가정이 있으면 결과가 왜 올라가거나 내려갔는지 관리자가 이해할 수 있는 말로 간단히 설명하라.
- 사용자가 입력한 가정을 답변에서 "장마", "가뭄" 같은 별도 해석어로 바꿔 부르지 마라. 반드시 "사용자님의 가정" 또는 "반영한 가정"이라고 표현하라.
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
