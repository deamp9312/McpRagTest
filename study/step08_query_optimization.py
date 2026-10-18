"""
[STEP 08] Advanced RAG ① 질의 최적화 (Pre-retrieval)  -  원본: Chapters/Chapter-06/advanced_rag.ipynb

사용자 질문은 짧거나 모호한 경우가 많아 검색이 잘 안 될 수 있습니다.
검색 "전에" 질문을 LLM 으로 손질해서 검색 품질을 높이는 기법들입니다.

- Multi-Query (질의 확장) : 질문을 여러 표현으로 바꿔 각각 검색 → 결과 합치기
- Step-back             : 더 넓은/추상적인 질문을 만들어 배경 지식까지 검색
- HyDE                  : LLM 이 "가상의 답변" 을 먼저 쓰고, 그 답변으로 검색
                          (질문보다 답변 문장이 실제 문서 청크와 더 비슷하게 생겼다는 아이디어)

실행: python study/step08_query_optimization.py   (OpenAI API 키 필요, step04 를 먼저 실행)
"""

# json: LLM 이 돌려준 JSON 문자열을 파싱하기 위해 사용
import json

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import common
from step06_generation import load_vector_store

# 확장 대상 원본 질문 (일부러 짧고 모호하게)
ORIGINAL_QUERY = "Kohli records?"


def multi_query(llm: ChatOpenAI, question: str, n: int = 3) -> list[str]:
    """LLM 으로 질문의 변형 n개를 JSON 으로 생성합니다."""
    # 괄호 안에 문자열 여러 개를 나란히 쓰면 자동으로 하나로 이어집니다 (긴 문자열을 여러 줄로 쓰는 방법)
    prompt = (
        f"Rewrite the following question in {n} different, more specific ways "
        f"for searching a document about the 2023 Cricket World Cup.\n"
        f"Question: {question}\n"
        'Respond in JSON: {"queries": ["...", "..."]}'   # 작은따옴표 문자열 안에서는 큰따옴표를 그대로 쓸 수 있습니다
    )
    # ┌ [메서드] llm.bind(**kwargs)
    # │ 역할 : 매 호출마다 추가로 넘길 API 파라미터를 "고정" 한 새 LLM 객체를 반환 (원본 llm 은 그대로)
    # │ 입력 : response_format={"type": "json_object"} → OpenAI 의 JSON 모드. 응답이 반드시 유효한 JSON
    # │        (JSON 모드는 프롬프트 안에 "JSON" 이라는 단어가 있어야 동작합니다)
    # └ Java 로 치면 불변 객체의 withXxx(...) 메서드 (예: HttpRequest.newBuilder(원본).header(...))
    json_llm = llm.bind(response_format={"type": "json_object"})
    # invoke 로 호출 → .content 는 JSON 형태의 문자열 → json.loads 로 dict 로 파싱
    data = json.loads(json_llm.invoke([("human", prompt)]).content)
    # dict 에서 "queries" 키의 값(리스트)을 반환
    return data["queries"]


def main() -> None:
    common.load_api_key()
    vector_store = load_vector_store()
    llm = ChatOpenAI(model=common.CHAT_MODEL, temperature=0)

    common.print_title("0) 기준선: 원본 질문으로 그냥 검색")

    # 비교 기준(baseline) 결과
    for doc in vector_store.similarity_search(ORIGINAL_QUERY, k=2):
        # print 에 인자를 여러 개 주면 공백으로 구분해서 출력합니다
        print("  -", common.clean_text(doc.page_content)[:150])

    common.print_title("1) Multi-Query: 질문을 여러 개로 확장해 각각 검색 후 합치기")

    queries = multi_query(llm, ORIGINAL_QUERY)
    # dict 는 삽입 순서를 유지하므로 "순서 있는 Set" 처럼 쓸 수 있습니다 (Java 의 LinkedHashMap)
    merged = {}
    for q in queries:
        print(f"확장 질의: {q}")
        for doc in vector_store.similarity_search(q, k=2):
            # ┌ [메서드] dict.setdefault(key, value)
            # │ 역할 : key 가 없을 때만 value 를 넣고, 있으면 아무것도 안 함 (기존 값 반환)
            # └ Java : map.putIfAbsent(key, value)
            # 같은 청크가 여러 질의에서 나와도 한 번만 저장 → 중복 제거
            merged.setdefault(doc.metadata["chunk_id"], doc)
    print(f"\n합쳐진 고유 청크 수: {len(merged)}")
    # dict.values(): 값들만 순회 (Java 의 map.values())
    for doc in merged.values():
        print("  -", common.clean_text(doc.page_content)[:150])

    common.print_title("2) Step-back: 한 단계 추상화된 질문 만들기")

    step_back_prompt = (
        f"Given the question '{ORIGINAL_QUERY}', write one broader, higher-level question "
        f"that would help retrieve useful background information. Return only the question."
    )
    # str.strip(): 앞뒤 공백/줄바꿈 제거 (Java 의 String.strip())
    step_back_query = llm.invoke([("human", step_back_prompt)]).content.strip()
    print(f"Step-back 질의: {step_back_query}")
    for doc in vector_store.similarity_search(step_back_query, k=2):
        print("  -", common.clean_text(doc.page_content)[:150])

    common.print_title("3) HyDE: 가상의 답변을 만들어 그 답변으로 검색")

    # system 메시지는 LLM 의 역할/규칙을 지정, human 메시지는 실제 요청
    hyde_messages = [
        ("system", "You are a cricket expert."),
        ("human", f"Write a short factual-sounding paragraph answering: {ORIGINAL_QUERY} "
                  f"(context: 2023 Cricket World Cup)"),
    ]
    hypothetical_answer = llm.invoke(hyde_messages).content
    # 가상의 답변은 틀릴 수 있습니다. 하지만 "답변처럼 생긴 문장" 이라 검색 벡터로는 유용합니다
    print(f"가상 답변: {hypothetical_answer[:300]}...\n")

    # 가상 답변을 직접 임베딩 (질문 대신 답변을 벡터로 만듦)
    embeddings = OpenAIEmbeddings(model=common.EMBEDDING_MODEL)
    hyde_vector = embeddings.embed_query(hypothetical_answer)
    # ┌ [메서드] vector_store.similarity_search_by_vector(embedding, k=4)
    # │ 역할 : 문자열 대신 "이미 만든 벡터" 로 검색 (내부에서 임베딩 단계를 건너뜀)
    # │ 입력 : embedding - list[float], k - 개수
    # └ 반환 : list[Document]
    for doc in vector_store.similarity_search_by_vector(hyde_vector, k=2):
        print("  -", common.clean_text(doc.page_content)[:150])

    # 관찰 포인트: 0) 기준선과 비교해 1)~3) 에서 Kohli 의 기록(765 runs, 50번째 ODI 센추리 등)이
    #              담긴 청크가 더 잘 검색되는지 확인해 보세요.


if __name__ == "__main__":
    main()
