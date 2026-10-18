"""
[STEP 09] Advanced RAG ② 하이브리드 검색 + 검색 후 압축  -  원본: Chapters/Chapter-06/advanced_rag.ipynb

- Hybrid Retrieval : BM25(키워드, step05) + FAISS(의미, step04) 결과를 합칩니다.
                     합치는 방법으로 RRF(Reciprocal Rank Fusion)를 직접 구현합니다.
                     RRF 점수 = Σ 1 / (60 + 순위)  → 여러 검색기에서 상위권인 문서가 최종 상위로
                     (두 검색기의 점수 단위가 서로 달라서 점수 대신 "순위" 만 사용하는 것이 핵심)
- Compression      : 검색된 청크를 LLM 으로 요약/압축해서 프롬프트를 짧게 만듭니다 (Post-retrieval)
                     → 토큰 비용 절감, 불필요한 정보로 인한 혼란 감소

실행: python study/step09_hybrid_and_compression.py   (OpenAI API 키 필요, step04 를 먼저 실행)
"""

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

import common
from step05_keyword_retrievers import tokenize
from step06_generation import load_vector_store, PROMPT_TEMPLATE

# RRF 상수. 순위 간 점수 차이를 완만하게 해 주는 값으로, 관례적으로 60을 사용합니다
RRF_K = 60


def reciprocal_rank_fusion(result_lists: list[list[Document]], top_n: int = 3) -> list[Document]:
    """여러 검색 결과 리스트를 RRF 로 합쳐 상위 top_n 개를 반환합니다."""
    # 변수에도 타입 힌트를 붙일 수 있습니다. dict[int, float] 는 Java 의 Map<Integer, Double>
    scores: dict[int, float] = {}
    docs_by_id: dict[int, Document] = {}

    # 바깥 루프: 검색기별 결과 리스트, 안쪽 루프: 리스트 안의 문서와 순위(1부터)
    for results in result_lists:
        for rank, doc in enumerate(results, start=1):
            chunk_id = doc.metadata["chunk_id"]
            # ┌ [메서드] dict.get(key, default)
            # └ 역할 : key 의 값, 없으면 default 반환 (Java: map.getOrDefault(key, 0.0))
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
            docs_by_id[chunk_id] = doc

    # ┌ [내장 함수] sorted(컬렉션, key=함수, reverse=True)
    # │ 역할 : 정렬된 "새 리스트" 반환 (원본 유지). list.sort() 는 제자리 정렬이라는 점이 다름
    # │ 입력 : dict 를 넘기면 키(chunk_id)들을 정렬. key=lambda cid: scores[cid] → 점수 기준
    # └ Java : scores.keySet().stream().sorted(Comparator.comparing(scores::get).reversed()).toList()
    ranked_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    # 상위 top_n 개의 점수 출력 (ranked_ids[:top_n] 은 앞에서 top_n 개 슬라이싱)
    for cid in ranked_ids[:top_n]:
        print(f"  RRF score={scores[cid]:.5f} chunk_id={cid}")
    # 상위 top_n 개의 Document 반환
    return [docs_by_id[cid] for cid in ranked_ids[:top_n]]


def main() -> None:
    common.load_api_key()
    vector_store = load_vector_store()
    llm = ChatOpenAI(model=common.CHAT_MODEL, temperature=0)

    # BM25 검색기는 저장 기능이 없어 청크 JSON 으로부터 매번 새로 만듭니다 (계산이 가벼움)
    raw_chunks = common.load_json(common.CHUNKS_FILE)
    docs = [Document(page_content=c["content"], metadata=c["metadata"]) for c in raw_chunks]
    # step05 의 tokenize(소문자 + 구두점 제거)를 재사용해 단어 매칭 품질을 높입니다
    bm25 = BM25Retriever.from_documents(docs, k=5, preprocess_func=tokenize)

    question = "How many runs did Travis Head score in the final?"

    common.print_title("1) 각 검색기 단독 결과")

    # 키워드 검색 결과 (Runnable.invoke → list[Document])
    bm25_results = bm25.invoke(question)
    # 의미 검색 결과
    dense_results = vector_store.similarity_search(question, k=5)

    # 각 결과의 chunk_id 만 뽑아 리스트로 출력해 두 검색기의 순위를 비교
    print("BM25 :", [d.metadata["chunk_id"] for d in bm25_results])
    print("FAISS:", [d.metadata["chunk_id"] for d in dense_results])

    common.print_title("2) RRF 로 하이브리드 결과 만들기")

    hybrid_docs = reciprocal_rank_fusion([bm25_results, dense_results], top_n=3)

    common.print_title("3) 검색 후 압축: 각 청크에서 질문과 관련된 내용만 남기기")

    compressed_parts = []
    for doc in hybrid_docs:
        original = common.clean_text(doc.page_content)
        compress_prompt = (
            f"Extract only the sentences relevant to the question. "
            f"If nothing is relevant, return an empty string.\n\n"
            f"Question: {question}\n\nDocument:\n{original}"
        )
        # LLM 에게 압축을 시키고 응답 텍스트의 앞뒤 공백 제거
        compressed = llm.invoke([("human", compress_prompt)]).content.strip()
        # 압축 전후 길이 비교
        print(f"chunk_id={doc.metadata['chunk_id']}: {len(original)}자 → {len(compressed)}자")
        # 빈 문자열은 if 문에서 False 로 취급됩니다 (Python 의 truthy/falsy). 관련 내용이 있을 때만 추가
        if compressed:
            compressed_parts.append(compressed)

    common.print_title("4) 압축된 context 로 최종 답변 생성")

    # step06 의 프롬프트 템플릿을 그대로 재사용
    prompt = PROMPT_TEMPLATE.format(question=question, context="\n---\n".join(compressed_parts))
    print(prompt)
    print("A:", llm.invoke([("human", prompt)]).content)


if __name__ == "__main__":
    main()
