# Ohbong Reservoir Risk Prediction

오봉 저수지의 향후 30일 저수율과 가뭄 단계를 예측하고, 필요 시 인근 저수지 물 운송 최적화를 지원하는 의사결정 프로젝트입니다.

## 주요 기능

- Supabase `public.master_dataset` 연동
- Supabase `public.reservoir_distances` 도로거리 데이터 연동
- 저수지별 시계열 전처리
- seq2seq Transformer 기반 향후 30일 저수율 예측
- 중앙 가뭄 단계 정책 기반 정상/관심/주의/경계/심각 판단
- 정상 또는 관심 단계 저수지만 공급 후보로 사용하는 물 운송 최적화
- Streamlit 기반 챗봇 UI

## 가뭄 단계 판단 기준

강릉시 저수지는 저수율 자체 기준을 사용합니다.

- 정상: 저수율 35% 이상
- 관심: 저수율 30% 이상 35% 미만
- 주의: 저수율 25% 이상 30% 미만
- 경계: 저수율 20% 이상 25% 미만
- 심각: 저수율 20% 미만

강릉시 외 저수지는 평년 저수율 대비 비율을 사용합니다.

- 정상: 평년 대비 70% 이상
- 관심: 평년 대비 60% 이상 70% 미만
- 주의: 평년 대비 50% 이상 60% 미만
- 경계: 평년 대비 40% 이상 50% 미만
- 심각: 평년 대비 40% 미만

평년 대비 비율:

```text
normal_ratio = 현재 또는 예측 저수율 / 평년 저수율 * 100
```

## 사용 기술

- Python
- pandas
- PyTorch
- scikit-learn
- Supabase
- PuLP
- Streamlit
- Google Gemini API

## 설치 및 실행

Windows PowerShell 기준:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 환경변수

`.env.example`을 참고하여 프로젝트 루트에 `.env` 파일을 생성합니다.

```text
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_or_publishable_key
SUPABASE_TABLE=master_dataset
GOOGLE_API_KEY=your_google_gemini_api_key
GEMINI_MODEL=gemini-2.0-flash
```

`.env`는 GitHub에 포함하지 않습니다.

## 주요 실행 명령어

```powershell
python src/db/supabase_client.py
python src/model/predict_seq2seq.py
python src/optimization/transport_optimizer.py
python src/pipeline/run_prediction_optimization.py
python -m streamlit run src/app.py
```

## 주의사항

- `.env`, 원본 데이터, 모델 가중치, 결과 파일은 GitHub에 포함하지 않습니다.
- 저수지 단계 판단은 `src/utils/risk_policy.py`의 중앙 정책 함수만 사용합니다.
- 평년 저수율 계산은 `src/utils/normal_rate.py`에서 수행합니다.
- 운영 배포 시 평년값 계산에는 기준일 이후 데이터가 포함되지 않도록 조정해야 합니다.
