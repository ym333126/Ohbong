# Ohbong Reservoir Risk Prediction

오봉 저수지의 향후 30일 최저 저수율과 위험 여부를 예측하여, 2025년 8월과 같은 물 부족 사태를 사전에 감지하기 위한 의사결정 지원 프로젝트입니다.

## 주요 기능

- Supabase `public.master_dataset` 연동
- 저수지별 시계열 전처리
- Transformer 기반 저수율 예측
- 향후 30일 최저 저수율 예측
- 위험 기준 이하 진입 여부 판단
- 2025년 8월 오봉 저수지 위기 사전탐지 테스트

## 사용 기술

- Python
- pandas
- PyTorch
- scikit-learn
- Supabase
- Streamlit

## 설치 및 실행 준비

Windows PowerShell 기준:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 환경변수 설정

`.env.example`을 참고하여 프로젝트 루트에 `.env` 파일을 생성합니다.

```text
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_or_publishable_key
SUPABASE_TABLE=master_dataset
```

주의: `.env`에는 실제 Supabase URL과 anon 또는 publishable key가 들어가므로 GitHub에 포함하지 않습니다.

## 주요 실행 명령어

```powershell
python src/db/supabase_client.py
python src/preprocessing/prepare_timeseries.py
python src/model/train_transformer_scenario_minrisk.py
python src/model/predict_minrisk.py
python src/model/test_ohbong_crisis_detection.py
```

## 주의사항

- `.env`는 GitHub에 포함하지 않습니다.
- 원본 데이터 파일은 GitHub에 포함하지 않습니다.
- 모델 가중치와 scaler 파일은 GitHub에 포함하지 않습니다.
- 모델 평가 결과 CSV, JSON, PNG 파일은 GitHub에 포함하지 않습니다.
- GitHub에는 코드, `requirements.txt`, `README.md`, `.env.example`, `.gitignore` 중심으로 업로드합니다.
