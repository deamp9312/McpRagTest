"""
[STEP 03] 임베딩 이해하기 (인덱싱 파이프라인 3단계: Embed)  -  원본: Chapters/Chapter-03/indexing_pipeline.ipynb

임베딩(embedding) = 텍스트를 "의미를 담은 숫자 배열(벡터)" 로 바꾸는 것.
의미가 비슷한 문장은 벡터 공간에서 가까운 위치에 놓입니다.
"가까움" 은 보통 코사인 유사도(cosine similarity)로 측정합니다 (1에 가까울수록 비슷함).

이 단계에서는 문장 몇 개로 직접 유사도를 계산해 "검색이 왜 동작하는지" 를 체감합니다.

실행: python study/step03_embeddings.py   (OpenAI API 키 필요, 비용 극소)
"""

# numpy: 수치 계산 라이브러리. 벡터/행렬 연산을 C 로 구현해 빠르게 처리 (`as np` 는 별칭 지정)
import numpy as np
# OpenAIEmbeddings: OpenAI 임베딩 API 를 감싼 LangChain 클래스
from langchain_openai import OpenAIEmbeddings

import common


def cosine_similarity(a: list, b: list) -> float:
    """두 벡터 사이의 코사인 유사도 = (a·b) / (|a| * |b|)  → -1 ~ 1, 클수록 방향(의미)이 비슷"""
    # ┌ [메서드] np.array(리스트)
    # └ 역할 : 파이썬 list 를 numpy 배열로 변환. 배열끼리 곱셈/합 등을 for 문 없이 한 번에 계산 가능
    # 한 줄에 두 변수 동시 대입 (튜플 언패킹)
    va, vb = np.array(a), np.array(b)
    # ┌ [메서드] np.dot(a, b)         → 내적: 같은 위치 원소끼리 곱해서 모두 더한 값 (Σ a[i]*b[i])
    # │ [메서드] np.linalg.norm(v)    → 벡터의 길이(크기): √(Σ v[i]²)
    # └ [내장 함수] float(x)          → numpy 실수 타입을 일반 파이썬 float 로 변환
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))


def main() -> None:
    # .env 에서 API 키 로드 (없으면 예외 발생)
    common.load_api_key()

    # ┌ [생성자] OpenAIEmbeddings(model, dimensions=None, chunk_size=1000, ...)
    # │ 역할 : OpenAI 임베딩 API 클라이언트 생성 (이 시점에는 API 호출 없음)
    # │ 입력 : model      - 사용할 임베딩 모델 이름
    # │        dimensions - 벡터 차원을 줄이고 싶을 때 지정 (3-small 기본 1536)
    # │        chunk_size - 한 번의 API 요청에 묶어 보낼 텍스트 개수 (배치 크기)
    # └ API 키는 환경 변수 OPENAI_API_KEY 에서 자동으로 읽음
    embeddings = OpenAIEmbeddings(model=common.EMBEDDING_MODEL)

    common.print_title("1) 문장 하나를 벡터로 바꿔 보기")

    # ┌ [메서드] embeddings.embed_query(text)
    # │ 역할 : 문자열 1개를 벡터 1개로 변환 (보통 "검색 질문" 을 임베딩할 때 사용)
    # │ 입력 : str
    # │ 반환 : list[float] - 길이 1536 의 실수 리스트
    # └ 내부 : OpenAI /v1/embeddings REST API 를 HTTP 로 호출 → 응답 JSON 의 벡터를 꺼내 반환
    vector = embeddings.embed_query("Australia won the Cricket World Cup 2023")
    print(f"벡터 차원: {len(vector)}")
    # 앞 5개 숫자만 출력. 숫자 하나하나에 사람이 읽을 의미는 없고, 고차원 공간의 '좌표' 일 뿐입니다
    print(f"앞 5개 값: {vector[:5]}")

    common.print_title("2) 질문과 문장들 사이의 유사도 비교")

    # 검색 질문
    query = "Who won the 2023 World Cup?"
    # 비교할 후보 문장들 (Java의 List.of(...) 와 같은 리스트 리터럴)
    sentences = [
        "Australia defeated India in the final to win the 2023 Cricket World Cup.",  # 정답에 가까움
        "Virat Kohli scored the most runs in the tournament.",                       # 관련은 있으나 정답 아님
        "The tournament was hosted by India.",                                      # 약간 관련
        "Polar bears depend on sea ice to hunt seals.",                              # 전혀 무관
    ]

    # 질문 1개 임베딩
    query_vec = embeddings.embed_query(query)
    # ┌ [메서드] embeddings.embed_documents(texts)
    # │ 역할 : 문자열 여러 개를 한꺼번에 벡터로 변환 (보통 "저장할 문서/청크" 를 임베딩할 때 사용)
    # │ 입력 : list[str]
    # │ 반환 : list[list[float]] - 입력 순서와 같은 순서의 벡터 리스트 (2차원 배열)
    # └ 내부 : chunk_size(1000)개씩 묶어 API 를 호출 → 호출 횟수를 줄이는 배치 처리
    sentence_vecs = embeddings.embed_documents(sentences)

    # ┌ [내장 함수] zip(리스트A, 리스트B)
    # └ 역할 : 같은 인덱스끼리 짝지은 (A[i], B[i]) 튜플을 차례로 반환. 길이가 다르면 짧은 쪽에 맞춤
    scored = [(s, cosine_similarity(query_vec, v)) for s, v in zip(sentences, sentence_vecs)]
    # ┌ [메서드] list.sort(key=함수, reverse=True)
    # │ 역할 : 리스트를 "제자리에서" 정렬 (새 리스트 반환 X, 반환값 None)
    # │ 입력 : key     - 정렬 기준값을 뽑는 함수. lambda pair: pair[1] 은 튜플의 두 번째 값(점수)
    # │        reverse - True 면 내림차순
    # └ Java : list.sort(Comparator.comparing((Pair p) -> p.score()).reversed())
    scored.sort(key=lambda pair: pair[1], reverse=True)

    print(f"질문: {query}\n")
    # 튜플 언패킹으로 문장과 점수를 각각 꺼내 출력
    for sentence, score in scored:
        # :.4f 는 소수점 4자리까지 출력하는 포맷 (Java 의 %.4f)
        print(f"{score:.4f}  {sentence}")

    # 관찰 포인트:
    # - "won" 이라는 단어가 없어도 'defeated ... to win' 문장이 가장 높은 점수를 받습니다 → 의미 기반 검색
    # - 북극곰 문장은 가장 낮은 점수 → 이 원리로 수백 개 청크 중 질문과 관련된 것만 골라낼 수 있습니다
    # - OpenAI 임베딩은 길이가 1로 정규화되어 있어서 내적(dot product) = 코사인 유사도 입니다
    #   (step04 에서 FAISS 의 IndexFlatIP(내적 인덱스)를 쓰는 이유)


if __name__ == "__main__":
    main()
