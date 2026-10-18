"""
[STEP 05] 키워드 기반 검색기: TF-IDF / BM25  -  원본: Chapters/Chapter-04/xtra_tfidf_bm25_retriever.ipynb

임베딩(의미 기반, dense) 검색과 대비되는 전통적인 "단어 일치 기반(sparse)" 검색입니다.
- TF-IDF : 질문 단어가 문서에 자주 나오고(TF), 전체 문서에는 드물수록(IDF) 높은 점수
- BM25   : TF-IDF 개선판. 단어가 너무 많이 반복되면 점수 증가가 둔화되고 문서 길이도 보정
           (Elasticsearch / Lucene 의 기본 랭킹 알고리즘이라 자바 개발자에게 익숙할 수 있습니다)

실행: python study/step05_keyword_retrievers.py   (API 키 불필요)
"""

# re: 정규식 (토크나이저 구현에 사용)
import re

# TF-IDF / BM25 검색기 (각각 scikit-learn, rank_bm25 라이브러리를 내부에서 사용)
from langchain_community.retrievers import TFIDFRetriever, BM25Retriever
from langchain_core.documents import Document

import common

# 실습용 짧은 문서 3개 (모듈 수준 상수)
TOY_TEXTS = [
    "Australia won the Cricket World Cup 2023",
    "India and Australia played in the finals",
    "Australia won the sixth time having last won in 2015",
]


def tokenize(text: str) -> list[str]:
    """
    BM25 용 단어 분리 함수: 소문자로 바꾸고 영문자/숫자 덩어리만 추출합니다.
    BM25Retriever 의 기본 분리 방식은 단순히 text.split() (공백 기준)이라서
    "Head" 와 "head", "final" 과 "final." 을 서로 다른 단어로 취급합니다. 그 차이를 보기 위한 함수입니다.
    """
    # ┌ [메서드] str.lower()               → 소문자로 변환한 새 문자열 (Java: toLowerCase())
    # └ [메서드] re.findall(pattern, text) → 정규식에 맞는 부분 문자열을 모두 찾아 list[str] 로 반환
    #                                       (Java: Matcher 로 while(m.find()) 돌며 group() 을 모으는 것)
    return re.findall(r"[a-z0-9]+", text.lower())


def show(retriever, query: str) -> None:
    """검색기에 질문을 던지고 결과를 순위대로 출력합니다. (파라미터 타입을 생략해도 동작합니다 - 덕 타이핑)"""
    # ┌ [메서드] retriever.invoke(query)
    # │ 역할 : 질문과 관련된 문서를 검색 (LangChain 의 모든 실행 객체가 공유하는 공통 메서드 "Runnable.invoke")
    # │ 입력 : 질문 문자열
    # │ 반환 : list[Document] - 관련도 높은 순서 (점수는 포함되지 않음)
    # └ 내부 : 각 Retriever 클래스가 구현한 _get_relevant_documents() 를 호출 → Java 의 인터페이스/다형성
    results = retriever.invoke(query)
    print(f"질문: {query}")
    for rank, doc in enumerate(results, start=1):
        # 줄바꿈을 정리하고 앞 150자만 출력
        print(f"  [{rank}] {common.clean_text(doc.page_content)[:150]}")
    print()   # 빈 줄


def main() -> None:
    common.print_title("1) 장난감 데이터로 TF-IDF vs BM25 비교")

    # ┌ [클래스 메서드] TFIDFRetriever.from_texts(texts, metadatas=None, tfidf_params=None, k=4)
    # │ 역할 : 문자열 리스트로 TF-IDF 검색기를 생성하는 정적 팩토리 메서드 (Java 의 XXX.of(...) 와 비슷)
    # │ 입력 : texts - 검색 대상 문서들, k - 검색 시 반환할 개수(기본 4)
    # │ 반환 : TFIDFRetriever 인스턴스
    # └ 내부 : scikit-learn 의 TfidfVectorizer 로 모든 문서를 TF-IDF 벡터로 변환해 둠
    #          검색 시 질문도 같은 방식으로 벡터화 → 코사인 유사도가 큰 순서로 k개 반환
    tfidf = TFIDFRetriever.from_texts(TOY_TEXTS)

    # ┌ [클래스 메서드] BM25Retriever.from_texts(texts, metadatas=None, bm25_params=None,
    # │                                          preprocess_func=<공백 split>, k=4)
    # │ 역할 : 문자열 리스트로 BM25 검색기 생성
    # │ 입력 : preprocess_func - 문장을 단어 리스트로 쪼개는 함수 (기본값: 공백 기준 split, 대소문자 구분)
    # └ 내부 : rank_bm25 라이브러리의 BM25Okapi 로 단어별 통계(빈도, 문서 길이)를 미리 계산해 둠
    #          검색 시 질문을 같은 함수로 쪼개 각 문서의 BM25 점수 계산 → 상위 k개 반환
    bm25 = BM25Retriever.from_texts(TOY_TEXTS, k=3)

    print("--- TF-IDF ---")
    # 'won' 이 두 번 등장하는 3번째 문장이 1위가 되는지 확인해 보세요
    show(tfidf, "won")
    print("--- BM25 ---")
    show(bm25, "Who won the 2023 Cricket World Cup?")

    common.print_title("2) 실제 청크(step02 결과)에 BM25 적용")

    # step02 에서 만든 청크 로드 → Document 리스트
    raw_chunks = common.load_json(common.CHUNKS_FILE)
    docs = [Document(page_content=c["content"], metadata=c["metadata"]) for c in raw_chunks]

    # ┌ [클래스 메서드] BM25Retriever.from_documents(documents, k=4, preprocess_func=...)
    # └ 역할 : from_texts 와 같지만 Document 리스트를 받음 (metadata 가 검색 결과에 유지됨)
    bm25_default = BM25Retriever.from_documents(docs, k=3)
    # preprocess_func 에 우리가 만든 tokenize 함수를 넘깁니다 (함수를 값처럼 전달 → Java 의 메서드 참조 Foo::tokenize)
    bm25_tokenized = BM25Retriever.from_documents(docs, k=3, preprocess_func=tokenize)

    # 고유명사/숫자처럼 "정확한 단어" 가 중요한 질문에서 키워드 검색이 강합니다
    question = "travis head final 137"
    print("--- 기본 분리(공백 split, 대소문자 구분) ---")
    show(bm25_default, question)
    print("--- tokenize 적용(소문자 + 구두점 제거) ---")
    show(bm25_tokenized, question)

    # 반대로 표현이 다른(동의어) 질문에서는 약합니다 → step04 의 임베딩 검색 결과와 비교해 보세요
    show(bm25_tokenized, "Which team lifted the trophy?")

    # 정리: 키워드 검색(정확한 단어)과 의미 검색(유사한 뜻)은 장단점이 서로 보완적입니다.
    #       step09 에서 두 방식을 합치는 "하이브리드 검색" 을 실습합니다.


if __name__ == "__main__":
    main()
