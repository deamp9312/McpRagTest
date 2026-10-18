"""
[STEP 02] 청킹 (인덱싱 파이프라인 2단계: Split)  -  원본: Chapters/Chapter-03/indexing_pipeline.ipynb

LLM 에 문서 전체를 넣을 수는 없으므로, 문서를 "검색 가능한 작은 조각(chunk)" 으로 자릅니다.
- chunk_size    : 조각 하나의 최대 길이(글자 수)
- chunk_overlap : 앞뒤 조각이 겹치는 길이. 문장이 경계에서 잘려 의미가 끊기는 것을 줄여 줍니다.

여기서는 두 가지 분할기를 비교합니다.
- CharacterTextSplitter          : 구분자 1개로만 자름 → 조각 길이가 들쭉날쭉
- RecursiveCharacterTextSplitter : 구분자 여러 개를 큰 단위→작은 단위 순으로 재귀 적용 → 더 균일

실행: python study/step02_chunking.py   (API 키 불필요)
결과: study/data/chunks.json
"""

# statistics: 평균/중앙값 등 간단한 통계 함수 표준 라이브러리
import statistics

# 두 가지 텍스트 분할기 import
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

import common


def print_stats(name: str, chunks: list) -> None:
    """청크 길이 통계를 출력합니다. `chunks: list` 는 Java의 List<?> 파라미터에 해당"""
    # 리스트 컴프리헨션: [표현식 for 변수 in 컬렉션] → 각 요소에 표현식을 적용한 새 리스트
    # Java Stream 으로 쓰면 chunks.stream().map(c -> c.getPageContent().length()).toList()
    lengths = [len(c.page_content) for c in chunks]
    # ┌ [내장 함수] min(리스트), max(리스트) → 최솟값/최댓값 (Java: Collections.min / max)
    # │ [메서드] statistics.mean(리스트)   → 산술 평균
    # │ [메서드] statistics.median(리스트) → 중앙값 (정렬했을 때 가운데 값)
    # └ [내장 함수] round(x, 1)            → 소수 첫째 자리까지 반올림
    print(f"[{name}] 청크 수={len(chunks)}, "
          f"최소={min(lengths)}, 최대={max(lengths)}, "
          f"평균={round(statistics.mean(lengths), 1)}, 중앙값={statistics.median(lengths)}")


def main() -> None:
    # Path.read_text: step01 에서 저장한 텍스트 파일 전체를 문자열로 읽기 (Java: Files.readString)
    text = common.RAW_TEXT_FILE.read_text(encoding="utf-8")
    print(f"원문 길이: {len(text):,} 글자")

    common.print_title("1) CharacterTextSplitter: 단일 구분자('\\n')로 자르기")

    # ┌ [생성자] CharacterTextSplitter(separator, chunk_size, chunk_overlap)
    # │ 역할 : "구분자 하나" 로 텍스트를 자르는 분할기 생성
    # │ 입력 : separator     - 자르는 기준 문자열 (기본값 "\n\n")
    # │        chunk_size    - 청크 최대 길이 (기본 4000). 길이는 기본적으로 글자 수(len)로 셈
    # │        chunk_overlap - 이웃 청크와 겹칠 길이 (기본 200)
    # └ 키워드 인자(name=value) 방식 호출. Java 의 Builder 처럼 이름으로 값을 넘기므로 순서 무관
    char_splitter = CharacterTextSplitter(
        separator="\n",      # 줄바꿈 기준으로 자름
        chunk_size=1000,     # 조각 하나의 목표 최대 길이 1000자
        chunk_overlap=100,   # 이웃 조각과 100자 겹치게
    )

    # ┌ [메서드] splitter.create_documents(texts, metadatas=None)
    # │ 역할 : 문자열 리스트를 청크 단위 Document 리스트로 변환
    # │ 입력 : texts     - list[str] (문서 여러 개를 한 번에 넣을 수 있어서 리스트로 받음)
    # │        metadatas - 각 문서에 붙일 메타데이터 리스트 (선택)
    # │ 반환 : list[Document] - 청크 1개당 Document 1개
    # └ 내부 : ① separator 로 텍스트를 잘게 쪼갬 → ② 조각들을 chunk_size 를 넘지 않을 때까지 이어 붙여
    #          청크 1개 완성 → ③ 다음 청크는 직전 청크의 끝부분(chunk_overlap 이내)을 다시 포함하여 시작
    #          주의: 구분자로 쪼갠 조각 하나가 chunk_size 보다 길면 더 자르지 못하고 큰 청크가 됨
    #          (참고: 이미 Document 리스트가 있다면 split_documents(docs) 를 쓰면 됨)
    char_chunks = char_splitter.create_documents([text])
    print_stats("Character", char_chunks)

    common.print_title("2) RecursiveCharacterTextSplitter: 여러 구분자를 재귀적으로 적용")

    # ┌ [생성자] RecursiveCharacterTextSplitter(separators, chunk_size, chunk_overlap)
    # │ 역할 : 구분자 "목록" 을 순서대로 시도하는 분할기 생성 (실무에서 가장 많이 쓰는 기본 분할기)
    # │ 입력 : separators - 우선순위 순 구분자 리스트 (기본값 ["\n\n", "\n", " ", ""])
    # └ 내부 : 먼저 "\n\n"(문단)으로 자르고, 그래도 chunk_size 보다 긴 조각만 다음 구분자 "\n" 으로,
    #          그래도 길면 "." → " " 순으로 재귀적으로 더 잘게 자름 → 의미 단위를 최대한 지키면서 크기 보장
    recursive_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],  # 문단 → 줄 → 문장 → 단어 순으로 시도
        chunk_size=1000,
        chunk_overlap=100,
    )
    # 사용법은 위와 동일 (같은 부모 클래스 TextSplitter 의 메서드 → Java 의 상속/다형성)
    recursive_chunks = recursive_splitter.create_documents([text])
    print_stats("Recursive", recursive_chunks)

    common.print_title("3) overlap 확인: 4번 청크의 끝과 5번 청크의 시작이 겹치는지 보기")

    # [-200:] 은 "뒤에서 200글자" 를 의미합니다 (Java: s.substring(s.length() - 200))
    # ┌ [내장 함수] repr(obj)
    # └ 역할 : 디버그용 문자열 표현. 문자열이면 따옴표와 함께 \n 같은 특수문자를 눈에 보이게 출력
    print("4번 청크 끝 200자 :", repr(recursive_chunks[4].page_content[-200:]))
    print("5번 청크 앞 200자 :", repr(recursive_chunks[5].page_content[:200]))

    common.print_title("4) 다음 단계를 위해 Recursive 청크를 JSON 으로 저장")

    # ┌ [내장 함수] enumerate(컬렉션)
    # │ 역할 : (인덱스, 값) 쌍을 차례로 돌려주는 반복자
    # └ Java : for (int i = 0; i < list.size(); i++) { var chunk = list.get(i); ... } 와 같은 효과
    # 각 청크를 dict(Java의 Map) 로 바꿔 리스트로 모읍니다. chunk_id 는 나중에 중복 제거/추적용
    data = [
        {"id": i, "content": chunk.page_content, "metadata": {"source": common.SOURCE_URL, "chunk_id": i}}
        for i, chunk in enumerate(recursive_chunks)
    ]
    # common.py 의 유틸로 JSON 저장
    common.save_json(common.CHUNKS_FILE, data)
    print(f"저장 완료 → {common.CHUNKS_FILE} ({len(data)}개 청크)")

    # TIP: chunk_size / chunk_overlap 값을 바꿔 가며 다시 실행해 보세요.
    #      너무 작으면 문맥이 부족하고, 너무 크면 검색 정확도가 떨어지고 LLM 비용이 커집니다.


if __name__ == "__main__":
    main()
