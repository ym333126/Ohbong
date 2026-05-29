import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chatbot.chatbot_engine import OhbongChatbot  # noqa: E402
from src.chatbot.gemini_answer_formatter import is_gemini_available  # noqa: E402


PAGE_TITLE = "오봉 저수지 위험 예측 및 물 운송 의사결정 챗봇"
PAGE_SUBTITLE = "향후 30일 저수율 예측 · 가뭄 단계 판단 · 인근 저수지 물 운송 최적화"

EXAMPLE_QUESTIONS = [
    "오봉 저수지 5월 26일 저수율을 알려줘",
    "6월 오봉 저수지 저수율 예측해줘",
    "오봉, 경포, 장현 6월 예측 비교해줘",
    "22개 저수지 예측 결과 표로 보여줘",
    "어디서 얼만큼 가져와야하는지 최종 후보 5개까지 알려줘",
    "다음 달 위험하면 최적화까지 해줘",
]


def initialize_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "안녕하세요. 오봉 저수지 관측값 조회, 향후 30일 예측, 물 운송 최적화를 도와드릴게요.",
                "table": None,
            }
        ]
    if "chatbot_error" not in st.session_state:
        st.session_state.chatbot_error = None
    if "chatbot" not in st.session_state and st.session_state.chatbot_error is None:
        with st.spinner("데이터와 모델을 불러오는 중입니다..."):
            try:
                st.session_state.chatbot = OhbongChatbot()
            except Exception as error:
                st.session_state.chatbot_error = str(error)


def render_sidebar() -> dict[str, Any]:
    st.sidebar.header("설정")
    st.sidebar.text_input("대상 저수지", value="오봉", disabled=True)
    target_rate = st.sidebar.number_input("오봉 목표 저수율 (%)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
    safety_rate = st.sidebar.number_input("기존 호환용 안전 저수율 (%)", min_value=0.0, max_value=100.0, value=40.0, step=1.0)
    max_distance_km = st.sidebar.number_input("최대 운송 거리 (km)", min_value=1.0, max_value=300.0, value=100.0, step=5.0)
    st.sidebar.number_input("예측 기간", value=30, disabled=True)
    st.sidebar.caption("공급 후보는 중앙 가뭄 단계 정책상 정상 또는 관심 단계만 허용합니다.")
    st.sidebar.text_input("데이터 소스", value="Supabase", disabled=True)

    gemini_available = is_gemini_available()
    use_gemini = st.sidebar.checkbox("Gemini 답변 정리 사용", value=gemini_available, disabled=not gemini_available)
    if not gemini_available:
        st.sidebar.info("GOOGLE_API_KEY 또는 GEMINI_API_KEY가 없어 기본 답변 템플릿을 사용합니다.")
    return {
        "target_rate": target_rate,
        "safety_rate": safety_rate,
        "max_distance_km": max_distance_km,
        "use_gemini": use_gemini,
    }


def render_project_card() -> None:
    with st.container(border=True):
        st.markdown(
            """
            **가뭄 단계 판단 기준**

            - 강릉시 저수지: 저수율 자체 기준으로 정상/관심/주의/경계/심각을 판단합니다.
            - 강릉시 외 저수지: 평년 저수율 대비 비율 기준으로 정상/관심/주의/경계/심각을 판단합니다.
            - 강릉시 기준: 정상 35% 이상, 관심 30~35%, 주의 25~30%, 경계 20~25%, 심각 20% 미만
            - 강릉시 외 기준: 정상 70% 이상, 관심 60~70%, 주의 50~60%, 경계 40~50%, 심각 40% 미만
            - 물 공급 후보는 기본적으로 정상 또는 관심 단계 저수지만 허용합니다.
            """
        )


def run_chatbot_message(message: str, settings: dict[str, Any]) -> dict[str, Any]:
    chatbot = st.session_state.get("chatbot")
    if chatbot is None:
        return {
            "answer": "챗봇이 초기화되지 않았습니다. Supabase 연결과 모델 artifact를 확인해 주세요.",
            "intent": "error",
            "intent_plan": None,
            "table": None,
            "prediction_summary": None,
            "optimization_result": None,
        }
    try:
        result = chatbot.handle_message(
            message,
            use_gemini=settings.get("use_gemini", True),
            target_rate=settings.get("target_rate", 50.0),
            safety_rate=settings.get("safety_rate", 40.0),
            max_distance_km=settings.get("max_distance_km", 100.0),
        )
        if isinstance(result, str):
            return {
                "answer": result,
                "intent": "legacy",
                "intent_plan": None,
                "table": None,
                "prediction_summary": None,
                "optimization_result": None,
            }
        return result
    except Exception as error:
        return {
            "answer": f"챗봇 응답 생성에 실패했습니다: {error}",
            "intent": "error",
            "intent_plan": None,
            "table": None,
            "prediction_summary": None,
            "optimization_result": None,
        }


def append_assistant_result(result: dict[str, Any]) -> None:
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.get("answer", ""),
            "table": result.get("table"),
            "intent": result.get("intent"),
            "intent_plan": result.get("intent_plan"),
        }
    )


def render_example_buttons(settings: dict[str, Any]) -> None:
    st.subheader("예시 질문")
    cols = st.columns(3)
    for index, question in enumerate(EXAMPLE_QUESTIONS):
        with cols[index % 3]:
            if st.button(question, use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": question, "table": None})
                with st.spinner("답변을 생성하는 중입니다..."):
                    result = run_chatbot_message(question, settings)
                append_assistant_result(result)
                st.rerun()


def render_chat() -> None:
    st.subheader("채팅")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            table = message.get("table")
            if isinstance(table, pd.DataFrame) and not table.empty:
                intent_plan = message.get("intent_plan") or {}
                top_k = intent_plan.get("top_k")
                if top_k and message.get("intent") in {"optimization", "pipeline"}:
                    table = table.head(int(top_k))
                st.dataframe(table, use_container_width=True, hide_index=True)


def render_prediction_metrics() -> None:
    chatbot = st.session_state.get("chatbot")
    summary = getattr(chatbot, "last_prediction_summary", None) if chatbot else None
    if summary is None or summary.empty:
        return

    label_col = "stage_label" if "stage_label" in summary.columns else "risk_label"
    if label_col not in summary.columns:
        return
    counts = summary[label_col].value_counts().to_dict()
    cols = st.columns(5)
    for col, label in zip(cols, ["정상", "관심", "주의", "경계", "심각"]):
        col.metric(f"{label} 저수지 수", int(counts.get(label, 0)))


def render_prediction_table() -> None:
    chatbot = st.session_state.get("chatbot")
    summary = getattr(chatbot, "last_prediction_summary", None) if chatbot else None
    if summary is None or summary.empty:
        return
    display_columns = [
        "저수지명",
        "시군",
        "predicted_min_rate",
        "min_forecast_date",
        "stage_label",
        "stage_basis",
        "normal_rate_at_min",
        "normal_ratio_at_min",
    ]
    existing = [column for column in display_columns if column in summary.columns]
    with st.expander("최근 22개 저수지 예측 결과 보기", expanded=False):
        st.dataframe(summary[existing] if existing else summary, use_container_width=True, hide_index=True)


def render_optimization_result() -> None:
    chatbot = st.session_state.get("chatbot")
    result = getattr(chatbot, "last_optimization_result", None) if chatbot else None
    if not result:
        return
    with st.expander("최근 물 운송 최적화 결과 보기", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("목표 달성", "가능" if result.get("feasible") else "어려움")
        col2.metric("필요 물량(천m3)", f"{result.get('required_water_1000m3', 0):.2f}")
        col3.metric("총 공급량(천m3)", f"{result.get('total_supplied_1000m3', 0):.2f}")
        shortage = result.get("shortage_1000m3", result.get("unmet_water_1000m3", 0.0))
        col4.metric("부족량(천m3)", f"{shortage:.2f}")
        transfers = result.get("allocations_display") or result.get("allocations") or result.get("transfers") or []
        if transfers:
            transfer_df = pd.DataFrame(transfers)
            display_columns = [
                "source_reservoir",
                "transfer_amount_1000m3",
                "distance_km",
                "available_supply_1000m3",
                "predicted_min_rate",
                "stage_label",
                "normal_ratio_at_min",
                "remaining_rate_after_transfer",
                "remaining_stage_label",
                "region",
            ]
            existing = [column for column in display_columns if column in transfer_df.columns]
            st.dataframe(transfer_df[existing] if existing else transfer_df, use_container_width=True, hide_index=True)
        else:
            st.info("표시할 운송 배분 결과가 없습니다.")


def main() -> None:
    st.set_page_config(page_title="Ohbong Reservoir Chatbot", page_icon="O", layout="wide")
    initialize_state()
    settings = render_sidebar()
    st.title(PAGE_TITLE)
    st.caption(PAGE_SUBTITLE)
    render_project_card()
    if st.session_state.chatbot_error:
        st.error(f"챗봇 초기화에 실패했습니다.\n\n{st.session_state.chatbot_error}")
        return
    render_prediction_metrics()
    render_example_buttons(settings)
    render_chat()
    user_message = st.chat_input("오봉 저수지 관측값, 예측, 운송 최적화를 질문해 보세요.")
    if user_message:
        st.session_state.messages.append({"role": "user", "content": user_message, "table": None})
        with st.spinner("답변을 생성하는 중입니다..."):
            result = run_chatbot_message(user_message, settings)
        append_assistant_result(result)
        st.rerun()
    render_prediction_table()
    render_optimization_result()


if __name__ == "__main__":
    main()

# Run with:
# python -m streamlit run src/app.py
