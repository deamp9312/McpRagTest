"""
모든 단계(step)에서 공통으로 사용하는 상수와 유틸 함수 모음.

[Java 비유]
- 이 파일 하나가 Java의 `public final class Common { static final ...; static ... }` 와 같은 역할입니다.
- Python에서는 "파일 = 모듈"이며, 다른 파일에서 `import common` 하면
  `common.DATA_DIR`, `common.clean_text(...)` 처럼 static 멤버처럼 꺼내 쓸 수 있습니다.

[주석 읽는 법]
- 처음 보는 라이브러리 메서드 앞에는 아래 형식의 설명 블록을 붙였습니다.
    # ┌ [메서드] 이름(파라미터)
    # │ 역할 : 무엇을 하는지
    # │ 입력 : 파라미터 의미
    # │ 반환 : 돌려주는 값의 타입과 의미
    # └ 내부 : 안에서 실제로 일어나는 일 (+ Java 비유)
"""

# os: 운영체제 기능(환경 변수, 경로 등)을 다루는 표준 라이브러리 (Java의 System 클래스와 비슷)
import os
# re: 정규식 표준 라이브러리 (Java의 java.util.regex.Pattern 에 해당)
import re
# json: JSON 직렬화/역직렬화 표준 라이브러리 (Java의 Jackson ObjectMapper 와 비슷한 역할)
import json
# Path: 파일 경로 객체 (Java의 java.nio.file.Path 와 거의 동일한 개념)
from pathlib import Path

# ┌ [메서드] load_dotenv(dotenv_path)
# │ 역할 : .env 파일의 "KEY=VALUE" 줄들을 읽어 현재 프로세스의 환경 변수로 등록
# │ 입력 : .env 파일 경로 (생략하면 현재 디렉터리부터 위로 올라가며 .env 를 찾음)
# │ 반환 : bool - 환경 변수를 하나라도 읽었으면 True
# └ 내부 : 이미 같은 이름의 환경 변수가 있으면 덮어쓰지 않음 (override=False 기본값)
#          Java 로 치면 application.properties 를 읽어 System.setProperty 하는 것과 비슷
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# 상수 정의 (Python은 final 키워드가 없어서 "대문자 이름 = 상수" 라는 관례만 사용합니다)
# ---------------------------------------------------------------------------

# __file__ : 현재 파일의 경로가 담긴 내장 변수
# Path.resolve() : 상대경로/심볼릭 링크를 풀어 절대경로로 변환 (Java: path.toRealPath())
# Path.parent    : 상위 디렉터리 Path (Java: path.getParent())
STUDY_DIR = Path(__file__).resolve().parent
# 저장소 루트 디렉터리 (study 폴더의 한 단계 위) - .env 파일을 여기서 찾습니다
ROOT_DIR = STUDY_DIR.parent
# Path 끼리는 `/` 연산자로 이어 붙입니다 (Java: STUDY_DIR.resolve("data"))
DATA_DIR = STUDY_DIR / "data"
# step01 결과물: 웹 페이지에서 추출한 순수 텍스트
RAW_TEXT_FILE = DATA_DIR / "raw_text.txt"
# step02 결과물: 청크(조각) 목록을 JSON 으로 저장
CHUNKS_FILE = DATA_DIR / "chunks.json"
# step04 결과물: FAISS 벡터 인덱스 파일 이름 (디렉터리는 DATA_DIR)
INDEX_NAME = "CWC_index"

# 학습에 사용할 원본 문서: 2023 크리켓 월드컵 위키백과 문서 (책/노트북과 동일)
SOURCE_URL = "https://en.wikipedia.org/wiki/2023_Cricket_World_Cup"
# 텍스트 → 벡터 변환에 쓸 OpenAI 임베딩 모델 (1536차원 벡터를 만듭니다)
EMBEDDING_MODEL = "text-embedding-3-small"
# 답변 생성에 쓸 OpenAI 채팅 모델 (저렴하고 빠른 모델)
CHAT_MODEL = "gpt-4o-mini"


def load_api_key() -> None:
    """
    .env 파일에서 OPENAI_API_KEY 를 읽어 환경 변수로 등록합니다.
    `-> None` 은 반환 타입 힌트로, Java의 `void` 와 같습니다 (강제는 아니고 문서 역할).
    """
    # 저장소 루트의 .env 파일을 읽어서 os.environ(환경 변수 Map)에 넣습니다
    load_dotenv(ROOT_DIR / ".env")
    # ┌ [메서드] os.environ.get(key)
    # │ 역할 : 환경 변수 값 조회
    # │ 반환 : 문자열, 없으면 None(=null)
    # └ 내부 : os.environ 은 dict(Map) 처럼 동작하는 객체. Java 의 System.getenv(key) 와 동일
    # `not 값` 은 값이 None 이거나 빈 문자열이면 True (Java: value == null || value.isEmpty())
    if not os.environ.get("OPENAI_API_KEY"):
        # raise 는 Java의 throw 와 같습니다. RuntimeError 는 RuntimeException 에 해당
        raise RuntimeError(
            "OPENAI_API_KEY 가 없습니다. 저장소 루트에 .env 파일을 만들고 "
            "OPENAI_API_KEY=sk-... 를 적어 주세요. (example_dot_env 참고)"
        )


def clean_text(text: str) -> str:
    """
    검색된 청크에서 LLM 에 넘기기 전에 불필요한 잡음을 제거합니다.
    `text: str` 은 파라미터 타입 힌트(Java의 String text), `-> str` 은 반환 타입(String)입니다.
    """
    # ┌ [메서드] str.replace(old, new)
    # │ 역할 : 문자열 안의 old 를 모두 new 로 바꾼 "새 문자열" 반환 (원본은 불변, Java 와 동일)
    # └ 여기서는 '\xa0'(줄바꿈 없는 공백, HTML의 &nbsp;)을 일반 공백으로 교체
    text = text.replace("\xa0", " ")
    # ┌ [메서드] re.sub(pattern, repl, string)
    # │ 역할 : 정규식 pattern 에 맞는 모든 부분을 repl 로 치환한 새 문자열 반환
    # └ Java : string.replaceAll(pattern, repl) 과 동일
    # <태그> 형태의 HTML 태그 제거. r"..." 은 백슬래시를 이스케이프하지 않는 raw 문자열
    text = re.sub(r"<[^>]+>", "", text)
    # 위키백과 각주 표시 [7], [39] 같은 대괄호 내용 제거 (.*? 는 가장 짧게 매칭하는 non-greedy)
    text = re.sub(r"\[.*?\]", "", text)
    # ┌ [메서드] str.split() / "구분자".join(리스트)
    # │ split() : 인자 없이 호출하면 공백·탭·줄바꿈 전부를 기준으로 쪼개고 빈 문자열은 버림 → list[str]
    # │ join()  : 리스트 요소들을 앞의 문자열(" ")로 이어 붙임 (Java: String.join(" ", list))
    # └ 결과   : 연속된 공백/줄바꿈이 공백 1개로 정리됨
    return " ".join(text.split())


def save_json(path: Path, data) -> None:
    """파이썬 객체(list/dict)를 JSON 파일로 저장합니다."""
    # ┌ [메서드] Path.mkdir(parents=True, exist_ok=True)
    # │ 역할 : 디렉터리 생성
    # │ 입력 : parents=True → 중간 경로까지 생성, exist_ok=True → 이미 있어도 예외 없음
    # └ Java : Files.createDirectories(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # ┌ [메서드] json.dumps(obj, ensure_ascii=False, indent=2)
    # │ 역할 : 파이썬 객체(dict/list/str/int...)를 JSON 문자열로 변환 (dump + s = "to string")
    # │ 입력 : ensure_ascii=False → 한글을 \uXXXX 로 바꾸지 않음, indent=2 → 2칸 들여쓰기
    # └ Java : objectMapper.writerWithDefaultPrettyPrinter().writeValueAsString(obj)
    # ┌ [메서드] Path.write_text(text, encoding)
    # └ 역할 : 파일을 열고 → 문자열 쓰고 → 닫기를 한 번에 (Java: Files.writeString(path, text, UTF_8))
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path):
    """JSON 파일을 읽어 파이썬 객체(list/dict)로 돌려줍니다."""
    # Path.exists(): 파일 존재 여부 (Java: Files.exists(path))
    if not path.exists():
        # f"...{변수}..." 는 f-string 으로, 중괄호 안의 표현식 값이 문자열에 끼워집니다
        raise FileNotFoundError(f"{path} 가 없습니다. 이전 단계 스크립트를 먼저 실행하세요.")
    # ┌ [메서드] Path.read_text(encoding) → 파일 전체를 문자열로 (Java: Files.readString)
    # └ [메서드] json.loads(문자열)       → JSON 문자열을 dict/list 로 파싱 (Java: objectMapper.readValue)
    return json.loads(path.read_text(encoding="utf-8"))


def print_title(title: str) -> None:
    """콘솔 출력을 보기 좋게 구분하기 위한 제목 출력 함수."""
    # 문자열 * 정수 는 문자열 반복입니다. "=" * 80 은 "=" 를 80번 이어 붙인 문자열 (Java: "=".repeat(80))
    # print(...) 는 Java의 System.out.println 과 같습니다 (여러 인자를 주면 공백으로 이어서 출력)
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
