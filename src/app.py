import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chatbot.chatbot_engine import OhbongChatbot  # noqa: E402
from src.chatbot.gemini_answer_formatter import is_gemini_available  # noqa: E402


DISPLAY_COLUMN_LABELS = {
    "저수지명": "저수지명",
    "시군": "지역",
    "as_of_date": "기준일",
    "forecast_start_date": "예측 시작일",
    "forecast_end_date": "예측 종료일",
    "predicted_min_rate": "예상 최저 저수율(%)",
    "lower_bound_at_min": "하한 예측 저수율(%)",
    "min_forecast_date": "최저 예상일",
    "stage_label": "단계",
    "stage_basis": "판단 기준",
    "normal_rate_at_min": "평년 저수율(%)",
    "normal_ratio_at_min": "평년 대비 비율(%)",
    "기본 예상 최저 저수율(%)": "기본 예상 최저 저수율(%)",
    "시나리오 반영 변화(%p)": "시나리오 반영 변화(%p)",
    "최근 30일 강수량(mm)": "최근 30일 강수량(mm)",
    "가정 강수량(mm)": "가정 강수량(mm)",
    "시나리오": "시나리오",
    "source_reservoir": "공급 저수지",
    "target_reservoir": "대상 저수지",
    "transfer_amount_1000m3": "운송 물량(천m³)",
    "distance_km": "거리(km)",
    "available_supply_1000m3": "공급 가능량(천m³)",
    "total_available_supply_1000m3": "총 공급 가능량(천m³)",
    "remaining_rate_after_transfer": "운송 후 저수율(%)",
    "remaining_stage_label": "운송 후 단계",
    "region": "구분",
    "selection_status": "배정 여부",
}

BASIS_LABELS = {
    "absolute_rate": "강릉시 저수율 기준",
    "normal_ratio": "평년 대비 비율 기준",
    "unknown": "판단 기준 없음",
}

STAGE_COLORS = {
    "정상": "#16a34a",
    "관심": "#2563eb",
    "주의": "#d97706",
    "경계": "#ea580c",
    "심각": "#dc2626",
}

STAGE_ICONS = {
    "정상": "🟢",
    "관심": "🔵",
    "주의": "🟡",
    "경계": "🟠",
    "심각": "🔴",
}

HIDDEN_DISPLAY_COLUMNS = {"truck", "trucks", "remaining_stage_code"}
CHAT_TABLE_CORE_COLUMNS = [
    "저수지명",
    "예측 기간",
    "예상 최저 저수율(%)",
    "최저 예상일",
    "평균 예측 저수율(%)",
    "단계",
    "판단 기준",
]


def drop_empty_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    cleaned = df.copy()
    empty_like = {"", "-", "None", "none", "null", "NaN", "nan", "NaT"}
    columns_to_drop = []
    for column in cleaned.columns:
        series = cleaned[column]
        missing_mask = series.isna() | series.astype(str).str.strip().isin(empty_like)
        if missing_mask.all():
            columns_to_drop.append(column)
    return cleaned.drop(columns=columns_to_drop)


PAGE_TITLE = "💧오봉 저수지 위험 예측 및 물 운송 의사결정 챗봇 🤖"
PAGE_SUBTITLE = "향후 30일 저수율 예측 · 저수율 단계 판단 · 인근 저수지 물 운송 최적화"

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
                "timestamp": current_time_label(),
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
    target_rate = st.sidebar.number_input(
        "운송 목표 저수율 (%)",
        min_value=0.0,
        max_value=100.0,
        value=35.0,
        step=1.0,
        help="운송 최적화를 요청했을 때만 사용하는 목표입니다. 강릉시 저수지의 정상 단계 하한은 35%입니다.",
    )
    max_distance_km = st.sidebar.number_input("최대 운송 거리 (km)", min_value=1.0, max_value=300.0, value=100.0, step=5.0)
    st.sidebar.number_input("예측 기간", value=30, disabled=True)
    st.sidebar.caption("강릉시 저수지는 35% 이상이면 정상 단계입니다. 운송은 예상 최저 저수율이 운송 목표보다 낮을 때만 검토합니다.")
    st.sidebar.caption("공급 후보는 중앙 단계 정책상 정상 또는 관심 단계만 허용합니다.")
    st.sidebar.text_input("데이터 소스", value="Supabase", disabled=True)

    gemini_available = is_gemini_available()
    use_gemini = st.sidebar.checkbox("Gemini 답변 정리 사용", value=gemini_available, disabled=not gemini_available)
    if not gemini_available:
        st.sidebar.info("GOOGLE_API_KEY 또는 GEMINI_API_KEY가 없어 기본 답변 템플릿을 사용합니다.")
    return {
        "target_rate": target_rate,
        "safety_rate": 40.0,
        "max_distance_km": max_distance_km,
        "use_gemini": use_gemini,
    }


def render_drought_criteria_section() -> None:
    normal = STAGE_COLORS["정상"]
    attention = STAGE_COLORS["관심"]
    caution = STAGE_COLORS["주의"]
    alert = STAGE_COLORS["경계"]
    severe = STAGE_COLORS["심각"]
    st.markdown(
        f"""
        <style>
            .drought-criteria-card {{
                background: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 14px;
                padding: 22px 24px;
                margin: 8px 0 18px 0;
            }}
            .drought-criteria-card h3 {{
                margin: 0 0 14px 0;
                font-size: 1.2rem;
                font-weight: 700;
                color: #1f2937;
            }}
            .drought-criteria-description {{
                margin: 0 0 6px 0;
                color: #4b5563;
                font-size: 0.95rem;
                line-height: 1.55;
            }}
            .drought-criteria-table-wrap {{
                width: 100%;
                overflow-x: auto;
                margin-top: 16px;
            }}
            .drought-criteria-table {{
                width: 100%;
                border-collapse: collapse;
                background: #ffffff;
                min-width: 760px;
            }}
            .drought-criteria-table th,
            .drought-criteria-table td {{
                border: 1px solid #e5e7eb;
                padding: 11px 12px;
                text-align: center;
                white-space: nowrap;
                font-size: 0.94rem;
            }}
            .drought-criteria-table th:first-child,
            .drought-criteria-table td:first-child {{
                color: #111827;
                font-weight: 700;
            }}
            .stage-normal {{ color: {normal}; font-weight: 700; }}
            .stage-attention {{ color: {attention}; font-weight: 700; }}
            .stage-caution {{ color: {caution}; font-weight: 700; }}
            .stage-alert {{ color: {alert}; font-weight: 700; }}
            .stage-severe {{ color: {severe}; font-weight: 700; }}
            .criteria-value {{ color: #111827; font-weight: 400; }}
        </style>
        <div class="drought-criteria-card">
            <h3>저수율 단계 판단 기준</h3>
            <p class="drought-criteria-description">강릉시 저수지: 저수율 자체 기준으로 정상/관심/주의/경계/심각을 판단합니다.</p>
            <p class="drought-criteria-description">강릉시 외 저수지: 평년 저수율 대비 비율 기준으로 정상/관심/주의/경계/심각을 판단합니다.</p>
            <div class="drought-criteria-table-wrap">
                <table class="drought-criteria-table">
                    <thead>
                        <tr>
                            <th>구분</th>
                            <th class="stage-normal">🟢 정상</th>
                            <th class="stage-attention">🔵 관심</th>
                            <th class="stage-caution">🟡 주의</th>
                            <th class="stage-alert">🟠 경계</th>
                            <th class="stage-severe">🔴 심각</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>강릉시 저수지</td>
                            <td class="criteria-value">35% 이상</td>
                            <td class="criteria-value">30% 이상 35% 미만</td>
                            <td class="criteria-value">25% 이상 30% 미만</td>
                            <td class="criteria-value">20% 이상 25% 미만</td>
                            <td class="criteria-value">20% 미만</td>
                        </tr>
                        <tr>
                            <td>강릉시 외 저수지</td>
                            <td class="criteria-value">70% 이상</td>
                            <td class="criteria-value">60% 이상 70% 미만</td>
                            <td class="criteria-value">50% 이상 60% 미만</td>
                            <td class="criteria-value">40% 이상 50% 미만</td>
                            <td class="criteria-value">40% 미만</td>
                        </tr>
                        <tr>
                            <td>물 공급 후보</td>
                            <td class="criteria-value">가능</td>
                            <td class="criteria-value">가능</td>
                            <td class="criteria-value">제외</td>
                            <td class="criteria-value">제외</td>
                            <td class="criteria-value">제외</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_project_card() -> None:
    render_drought_criteria_section()


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
            "timestamp": current_time_label(),
        }
    )


def current_time_label() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def prepare_display_dataframe(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    display_df = df.copy()
    display_df = display_df.drop(columns=[column for column in HIDDEN_DISPLAY_COLUMNS if column in display_df.columns])

    if columns:
        existing = [column for column in columns if column in display_df.columns and column not in HIDDEN_DISPLAY_COLUMNS]
        display_df = display_df[existing] if existing else display_df

    if "stage_basis" in display_df.columns:
        display_df["stage_basis"] = display_df["stage_basis"].map(BASIS_LABELS).fillna(display_df["stage_basis"])

    numeric_columns = [
        "predicted_min_rate",
        "lower_bound_at_min",
        "normal_rate_at_min",
        "normal_ratio_at_min",
        "transfer_amount_1000m3",
        "distance_km",
        "available_supply_1000m3",
        "remaining_rate_after_transfer",
    ]
    for column in numeric_columns:
        if column in display_df.columns:
            display_df[column] = pd.to_numeric(display_df[column], errors="coerce").round(2)

    display_df = drop_empty_display_columns(display_df)
    return display_df.rename(columns=DISPLAY_COLUMN_LABELS)


def prepare_chat_table(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if set(CHAT_TABLE_CORE_COLUMNS).issubset(set(df.columns)):
        return prepare_display_dataframe(df, CHAT_TABLE_CORE_COLUMNS)
    return prepare_display_dataframe(df)


def render_example_buttons(settings: dict[str, Any]) -> None:
    st.subheader("예시 질문")
    cols = st.columns(3)
    for index, question in enumerate(EXAMPLE_QUESTIONS):
        with cols[index % 3]:
            if st.button(question, use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": question, "table": None, "timestamp": current_time_label()})
                with st.spinner("답변을 생성하는 중입니다..."):
                    result = run_chatbot_message(question, settings)
                append_assistant_result(result)
                st.rerun()


def render_chat() -> None:
    st.subheader("채팅")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("timestamp"):
                st.caption(message["timestamp"])
            st.markdown(message["content"])
            table = message.get("table")
            if isinstance(table, pd.DataFrame) and not table.empty:
                intent_plan = message.get("intent_plan") or {}
                top_k = intent_plan.get("top_k")
                if top_k and message.get("intent") in {"optimization", "pipeline"}:
                    table = table.head(int(top_k))
                st.dataframe(prepare_chat_table(table), use_container_width=True, hide_index=True)


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
        col.metric(f"{STAGE_ICONS.get(label, '')} {label} 저수지 수", int(counts.get(label, 0)))


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
        "기본 예상 최저 저수율(%)",
        "시나리오 반영 변화(%p)",
        "최근 30일 강수량(mm)",
        "가정 강수량(mm)",
        "시나리오",
    ]
    with st.expander("최근 22개 저수지 예측 결과 보기", expanded=False):
        st.dataframe(prepare_display_dataframe(summary, display_columns), use_container_width=True, hide_index=True)


def render_optimization_result() -> None:
    chatbot = st.session_state.get("chatbot")
    result = getattr(chatbot, "last_optimization_result", None) if chatbot else None
    if not result:
        return
    with st.expander("최근 물 운송 최적화 결과 보기", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("목표 달성", "가능" if result.get("feasible") else "어려움")
        col2.metric("필요 물량(천m3)", f"{result.get('required_water_1000m3', 0):.2f}")
        col3.metric("실제 배정량(천m3)", f"{result.get('total_supplied_1000m3', 0):.2f}")
        shortage = result.get("shortage_1000m3", result.get("unmet_water_1000m3", 0.0))
        col4.metric("부족량(천m3)", f"{shortage:.2f}")
        transfers = result.get("allocations_display") or result.get("allocations") or result.get("transfers") or []
        if transfers:
            transfer_df = pd.DataFrame(transfers)
            display_columns = [
                "source_reservoir",
                "selection_status",
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
            st.dataframe(prepare_display_dataframe(transfer_df, display_columns), use_container_width=True, hide_index=True)
        elif float(result.get("required_water_1000m3", 0.0) or 0.0) <= 0:
            st.info("예상 최저 저수율이 목표 저수율 이상이라 운송 배분이 필요하지 않습니다.")
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
        st.session_state.messages.append({"role": "user", "content": user_message, "table": None, "timestamp": current_time_label()})
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
