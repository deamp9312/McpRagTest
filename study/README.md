# RAG 단계별 학습 가이드 (Java 개발자용)

이 폴더는 저장소의 노트북(`Chapters/`) 예제를 **실행 가능한 `.py` 스크립트 9개**로 다시 구성한 학습 코스입니다.
모든 코드에 줄 단위로 한국어 주석과 Java 비유를 달아 두었습니다.

## 전체 흐름

```
[인덱싱 파이프라인 - 미리 한 번 실행]                     [생성 파이프라인 - 질문마다 실행]
 웹 문서 ─▶ 텍스트 ─▶ 청크 ─▶ 임베딩 ─▶ FAISS 저장        질문 ─▶ 검색(Retrieve) ─▶ 프롬프트 보강(Augment) ─▶ LLM 답변(Generate)
  step01    step01   step02   step03    step04              step05(키워드) / step06(전체)

 평가: step07 (RAGAS)        고급 기법: step08 (질의 최적화), step09 (하이브리드 검색 + 압축)
```

| 단계 | 파일 | 배우는 것 | API 키 | 원본 노트북 |
|---|---|---|---|---|
| 01 | `step01_load_document.py` | 웹 문서 로딩, HTML → 텍스트 | 불필요 | Chapter-03 |
| 02 | `step02_chunking.py` | 청킹, chunk_size / overlap | 불필요 | Chapter-03 |
| 03 | `step03_embeddings.py` | 임베딩, 코사인 유사도 직접 계산 | 필요 | Chapter-03 |
| 04 | `step04_vector_store.py` | FAISS 인덱스 구축·저장·검색 | 필요 | Chapter-03 |
| 05 | `step05_keyword_retrievers.py` | TF-IDF, BM25 키워드 검색 | 불필요 | Chapter-04 (xtra) |
| 06 | `step06_generation.py` | Retrieve → Augment → Generate | 필요 | Chapter-04 |
| 07 | `step07_evaluation.py` | RAGAS 로 품질 평가 | 필요 | Chapter-05 |
| 08 | `step08_query_optimization.py` | Multi-Query, Step-back, HyDE | 필요 | Chapter-06 |
| 09 | `step09_hybrid_and_compression.py` | 하이브리드 검색(RRF), 검색 후 압축 | 필요 | Chapter-06 |

단계마다 이전 단계의 결과 파일(`study/data/`)을 사용하므로 **01번부터 순서대로** 실행하세요.

## 환경 설정

> 노트북의 고정 버전(`faiss-cpu==1.10.0` 등)은 최신 Python(3.13+)용 패키지가 없을 수 있으니 **Python 3.11 또는 3.12** 를 권장합니다.

```bash
# 저장소 루트에서 실행
brew install python@3.12                 # 없으면 설치 (macOS)
python3.12 -m venv .venv                 # 가상환경 생성 (Java 의 프로젝트별 의존성 격리와 같은 개념)
source .venv/bin/activate                # 가상환경 활성화
pip install -r requirements.txt          # 의존성 설치 (Maven/Gradle 의 dependency 설치에 해당)

cp example_dot_env .env                  # .env 파일 생성 후 OPENAI_API_KEY 값을 실제 키로 수정
```

실행 예:

```bash
python study/step01_load_document.py
python study/step02_chunking.py
...
```

OpenAI 비용: 전체 단계를 한 번 실행해도 보통 수십 원~수백 원 수준입니다 (`text-embedding-3-small`, `gpt-4o-mini` 사용).

## Java 개발자를 위한 Python 핵심 대응표

| Python | Java | 비고 |
|---|---|---|
| `import common` / `common.foo()` | `import static ...Common.*` | 파일 하나 = 모듈 하나 |
| `def f(x: str) -> int:` | `static int f(String x)` | 타입 힌트는 선택 사항 (강제 X) |
| `if __name__ == "__main__": main()` | `public static void main(String[] args)` | 직접 실행할 때만 동작 |
| `obj = MyClass(a=1, b=2)` | `new MyClass(1, 2)` / Builder | `new` 없음, 이름으로 인자 전달 가능 |
| `None`, `True`, `False` | `null`, `true`, `false` | |
| `list` `[1, 2]` / `dict` `{"k": v}` | `List.of(1, 2)` / `Map.of("k", v)` | |
| `[x.name for x in items]` | `items.stream().map(x -> x.name).toList()` | 리스트 컴프리헨션 |
| `lambda x: x[1]` | `x -> x[1]` | |
| `f"값={v}"` | `String.format("값=%s", v)` | f-string |
| `s[:100]`, `s[-100:]` | `substring(0,100)`, 끝 100자 | 슬라이싱 |
| `for i, v in enumerate(lst):` | `for (int i=0; ...)` | 인덱스+값 동시 |
| `a, b = func()` | `record` 반환 후 꺼내기 | 튜플 언패킹 |
| `raise RuntimeError("...")` | `throw new RuntimeException("...")` | |
| `"""여러 줄"""` | `"""텍스트 블록"""` (Java 15+) | |
| 들여쓰기(4칸) | `{ }` | 들여쓰기가 곧 블록 문법 |

## 단계별 핵심 개념 요약

- **청크(Chunk)**: LLM 에 문서 전체를 넣을 수 없으니 검색 단위로 자른 조각. 크기·겹침이 품질을 좌우합니다.
- **임베딩(Embedding)**: 텍스트 → 의미 벡터. 비슷한 의미 = 가까운 벡터. 저장할 때와 검색할 때 **같은 모델**을 써야 합니다.
- **벡터 저장소(FAISS)**: 벡터 유사도 검색 전용 저장소. 실무에서는 pgvector, Elasticsearch, Pinecone 등도 많이 씁니다.
- **Dense vs Sparse 검색**: 의미 기반(임베딩) vs 단어 일치 기반(BM25). 서로 보완적이라 하이브리드로 합치는 경우가 많습니다.
- **Augment**: RAG 의 핵심은 결국 "검색 결과를 프롬프트에 붙여 넣는 것" 입니다 (step06 의 프롬프트 출력으로 확인).
- **평가**: 검색 품질(context recall)과 생성 품질(faithfulness)을 나눠 측정해야 어디를 고칠지 알 수 있습니다.

## 추천 실습 과제

1. `step02` 의 `chunk_size` 를 500 / 2000 으로 바꾸고 → `step04` → `step07` 을 다시 실행해 점수를 비교하기
2. `step06` 의 `k` 값(검색 개수)을 바꿔 답변 변화 관찰하기
3. `step07` 의 `GOLDEN_SET` 에 직접 질문/정답을 추가해 보기
4. `step09` 의 하이브리드 검색 결과를 `step06` 의 `rag()` 에 적용해 보기
5. (도전) `study/data/raw_text.txt` 대신 사내 문서나 관심 있는 문서로 바꿔서 나만의 RAG 만들기
