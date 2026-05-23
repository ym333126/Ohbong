from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from supabase import Client, create_client


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "transformer_master_dataset.csv"
DEFAULT_TABLE_NAME = "master_dataset"

REQUIRED_COLUMNS = [
    "저수지명",
    "시군",
    "강수량매핑용주소",
    "SPI매핑용주소",
    "상세위치",
    "위도",
    "경도",
    "유효저수량(천m3)",
    "날짜",
    "저수율",
    "평균기온(℃)",
    "기온_7일평균",
    "강수량",
    "강수량_3일누적",
    "강수량_7일누적",
    "SPI3",
    "SPI6",
    "저수율_5일EMA",
]


def _load_env() -> None:
    env_path = PROJECT_ROOT / ".env"
    load_dotenv(env_path)


def get_supabase_client() -> Client:
    """Create a Supabase client from .env settings."""
    _load_env()

    import os

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    missing = []
    if not supabase_url:
        missing.append("SUPABASE_URL")
    if not supabase_key:
        missing.append("SUPABASE_KEY")

    if missing:
        raise RuntimeError(
            ".env 설정이 부족합니다. "
            f"누락된 환경변수: {', '.join(missing)}. "
            f"프로젝트 루트의 .env 파일을 확인하세요: {PROJECT_ROOT / '.env'}"
        )

    try:
        return create_client(supabase_url, supabase_key)
    except Exception as exc:
        raise RuntimeError(
            "Supabase 클라이언트 생성에 실패했습니다. "
            "SUPABASE_URL 형식이 올바른지, SUPABASE_KEY가 anon public key인지 확인하세요. "
            "service_role key는 사용하지 않습니다."
        ) from exc


def validate_columns(df: pd.DataFrame) -> None:
    """Validate required Korean column names without renaming them."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(
            "필수 컬럼이 누락되었습니다. "
            f"누락 컬럼: {missing_columns}. "
            "DataFrame 컬럼명을 임의로 변경하지 말고 원본 컬럼명을 유지하세요."
        )


def _postprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    validate_columns(df)
    df = df.copy()
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")

    if df["날짜"].isna().any():
        invalid_count = int(df["날짜"].isna().sum())
        raise ValueError(
            f"날짜 컬럼을 datetime으로 변환하지 못한 행이 {invalid_count}개 있습니다. "
            "원본 날짜 형식을 확인하세요."
        )

    return df


def load_master_dataset_from_supabase(
    table_name: Optional[str] = None,
    page_size: int = 1000,
) -> pd.DataFrame:
    """Load all rows from public.master_dataset with range pagination."""
    _load_env()

    import os

    selected_table = table_name or os.getenv("SUPABASE_TABLE") or DEFAULT_TABLE_NAME
    client = get_supabase_client()

    all_rows = []
    start = 0

    try:
        while True:
            end = start + page_size - 1
            response = (
                client.table(selected_table)
                .select("*")
                .range(start, end)
                .execute()
            )

            rows = response.data or []
            if not rows:
                break

            all_rows.extend(rows)

            if len(rows) < page_size:
                break

            start += page_size
    except Exception as exc:
        raise RuntimeError(
            "Supabase 데이터 로딩에 실패했습니다. "
            f"대상 테이블은 public.{selected_table} 입니다. "
            "가능한 원인: .env의 SUPABASE_URL 또는 SUPABASE_KEY 오류, "
            "anon public key 권한 부족, RLS 정책 제한, 테이블명 오류, 네트워크 문제. "
            "Supabase에서 public.master_dataset에 anon role이 SELECT 가능하도록 설정되어 있는지 확인하세요."
        ) from exc

    if not all_rows:
        raise ValueError(
            f"Supabase public.{selected_table}에서 데이터를 가져왔지만 행이 없습니다. "
            "테이블에 데이터가 있는지, anon role SELECT 권한이 있는지 확인하세요."
        )

    df = pd.DataFrame(all_rows)
    return _postprocess_dataset(df)


def load_master_dataset_from_csv(csv_path: Optional[Path | str] = None) -> pd.DataFrame:
    """Load the fallback CSV dataset from the local data directory."""
    selected_path = Path(csv_path) if csv_path else DEFAULT_CSV_PATH

    if not selected_path.exists():
        raise FileNotFoundError(
            "백업용 CSV 파일을 찾을 수 없습니다. "
            f"기본 경로: {DEFAULT_CSV_PATH}. "
            "Supabase 연결 실패 시 CSV fallback을 사용하려면 "
            "data/transformer_master_dataset.csv 파일을 프로젝트에 추가하세요."
        )

    df = pd.read_csv(selected_path)
    return _postprocess_dataset(df)


def load_master_dataset(source: str = "supabase") -> pd.DataFrame:
    """Load the master dataset from Supabase, CSV, or Supabase-then-CSV auto mode."""
    normalized_source = source.lower().strip()

    if normalized_source == "supabase":
        return load_master_dataset_from_supabase()

    if normalized_source == "csv":
        return load_master_dataset_from_csv()

    if normalized_source == "auto":
        try:
            return load_master_dataset_from_supabase()
        except Exception as supabase_error:
            print("[안내] Supabase 로딩에 실패하여 CSV fallback을 시도합니다.")
            print(f"[Supabase 오류] {supabase_error}")
            try:
                return load_master_dataset_from_csv()
            except Exception as csv_error:
                raise RuntimeError(
                    "Supabase와 CSV fallback 로딩이 모두 실패했습니다. "
                    "먼저 .env의 Supabase 설정과 권한을 확인하고, "
                    "백업 CSV가 필요한 경우 data/transformer_master_dataset.csv 파일을 추가하세요."
                ) from csv_error

    raise ValueError(
        "source 값은 'supabase', 'csv', 'auto' 중 하나여야 합니다. "
        f"입력값: {source}"
    )


def _print_dataset_summary(df: pd.DataFrame) -> None:
    print("=== Ohbong master_dataset 로딩 테스트 ===")
    print(f"전체 행 수: {len(df):,}")
    print(f"컬럼 목록: {list(df.columns)}")
    print(f"저수지 개수: {df['저수지명'].nunique():,}")
    print(f"날짜 범위: {df['날짜'].min().date()} ~ {df['날짜'].max().date()}")

    ohbong_df = df[df["저수지명"] == "오봉"].sort_values("날짜")
    print("\n오봉 데이터 5행:")
    if ohbong_df.empty:
        print("저수지명 == '오봉'인 데이터가 없습니다.")
    else:
        print(ohbong_df.head(5).to_string(index=False))


if __name__ == "__main__":
    try:
        dataset = load_master_dataset(source="auto")
        _print_dataset_summary(dataset)
    except Exception as error:
        print("데이터 로딩 테스트에 실패했습니다.")
        print(error)
