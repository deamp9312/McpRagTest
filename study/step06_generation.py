"""
[STEP 06] 생성 파이프라인: Retrieve → Augment → Generate  -  원본: Chapters/Chapter-04/generation_pipeline.ipynb

드디어 RAG 의 완성형입니다.
1. Retrieve : 질문과 비슷한 청크를 벡터 저장소에서 찾는다
2. Augment  : 찾은 청크를 "참고 자료(context)" 로 프롬프트에 끼워 넣는다
3. Generate : LLM 이 그 참고 자료만 보고 답하게 한다 → 환각(hallucination) 감소, 최신/사내 지식 활용

실행: python study/step06_generation.py   (OpenAI API 키 필요, step04 를 먼저 실행)
"""

# FAISS 벡터 저장소 클래스
from langchain_community.vectorstores import FAISS
# OpenAIEmbeddings: 질문 임베딩용, ChatOpenAI: 답변 생성용 LLM 클래스
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

import common

# 프롬프트 템플릿. 세 개의 따옴표(""" """)는 여러 줄 문자열 (Java 15+ 의 텍스트 블록 """ 와 동일)
# {question}, {context} 자리는 나중에 .format(...) 으로 채웁니다
PROMPT_TEMPLATE = """Answer the question using ONLY the context below.
If the context does not contain the answer, reply exactly: I don't know.

Question: {question}

Context:
{context}
"""


def load_vector_store() -> FAISS:
    """step04 에서 저장한 FAISS 인덱스를 디스크에서 불러옵니다."""
    # 질문을 같은 모델로 임베딩해야 하므로 저장 시와 동일한 임베딩 모델을 넘겨 줍니다 (중요!)
    # 모델이 다르면 벡터 "좌표계" 가 달라서 검색 결과가 엉망이 됩니다
    embeddings = OpenAIEmbeddings(model=common.EMBEDDING_MODEL)
    # ┌ [클래스 메서드] FAISS.load_local(folder_path, embeddings, index_name="index",
    # │                                  allow_dangerous_deserialization=False)
    # │ 역할 : save_local 로 저장한 .faiss + .pkl 파일을 읽어 FAISS 저장소 객체를 복원
    # │ 반환 : FAISS 인스턴스 (바로 similarity_search 가능)
    # └ 주의 : .pkl 은 pickle 형식이라 악의적인 파일을 열면 임의 코드가 실행될 수 있어서,
    #          "내가 만든 파일이라 안전하다" 는 의미로 allow_dangerous_deserialization=True 를 명시해야 함
    #          (Java 의 ObjectInputStream 역직렬화 취약점과 같은 이슈)
    return FAISS.load_local(
        folder_path=str(common.DATA_DIR),
        index_name=common.INDEX_NAME,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )


# 반환 타입 tuple[list[str], str] 은 "문자열 리스트와 문자열 두 값을 묶어 반환" 한다는 뜻
# Java 라면 record RagResult(List<String> contexts, String answer) 를 만들어 반환하는 것과 같습니다
def rag(question: str, vector_store: FAISS, llm: ChatOpenAI, k: int = 2) -> tuple[list[str], str]:
    """RAG 한 사이클 실행. `k: int = 2` 는 기본값이 있는 파라미터 (Java 의 오버로딩 대신 사용)"""
    # ┌ [메서드] vector_store.similarity_search(query, k=4)
    # │ 역할 : 질문과 가장 비슷한 청크 k개를 Document 리스트로 반환 (점수 없이)
    # └ 내부 : 질문 임베딩(API 호출) → faiss 에서 가까운 벡터 k개 검색 → 원문 Document 로 변환 (step04 참고)
    retrieved_docs = vector_store.similarity_search(question, k=k)
    # 각 청크의 잡음 제거 → 문자열 리스트
    contexts = [common.clean_text(doc.page_content) for doc in retrieved_docs]

    # 2) Augment: 청크들을 구분선으로 이어 붙여 프롬프트의 {context} 에 넣습니다
    context_text = "\n---\n".join(contexts)
    # ┌ [메서드] str.format(이름=값, ...)
    # │ 역할 : 문자열 안의 {이름} 자리를 값으로 채운 새 문자열 반환
    # └ Java : String.format 과 비슷하지만 위치(%s) 대신 이름으로 채움
    prompt = PROMPT_TEMPLATE.format(question=question, context=context_text)

    # ┌ [메서드] llm.invoke(messages)
    # │ 역할 : LLM 에 대화 메시지를 보내고 응답을 받음 (OpenAI Chat Completions API 호출)
    # │ 입력 : 문자열 하나, 또는 (역할, 내용) 튜플의 리스트
    # │        역할: "system"(LLM 의 성격/규칙 지정), "human"(사용자 발화), "ai"(이전 LLM 응답)
    # │ 반환 : AIMessage 객체
    # │        .content          → 답변 텍스트 (str)
    # │        .usage_metadata   → 입력/출력 토큰 수 (비용 계산용)
    # └ 내부 : HTTP POST /v1/chat/completions, 실패 시 max_retries 만큼 재시도
    ai_message = llm.invoke([("human", prompt)])
    answer = ai_message.content

    # 두 값을 튜플로 반환 (호출부에서 contexts, answer = rag(...) 로 받을 수 있음)
    return contexts, answer


def main() -> None:
    common.load_api_key()

    # 벡터 저장소와 LLM 은 매번 만들 필요가 없으므로 한 번만 생성해서 재사용합니다
    # (노트북 원본은 함수 안에서 매번 로드하지만, 실무에서는 싱글톤 Bean 처럼 재사용하는 것이 일반적)
    vector_store = load_vector_store()

    # ┌ [생성자] ChatOpenAI(model, temperature, max_tokens=None, timeout=None, max_retries=2)
    # │ 역할 : OpenAI 채팅 모델 클라이언트 생성 (이 시점에는 API 호출 없음)
    # │ 입력 : model       - 모델 이름
    # │        temperature - 0~2. 0 = 가장 결정적(같은 질문에 거의 같은 답), 높을수록 창의적/무작위
    # │        max_tokens  - 답변 최대 길이(토큰 단위), None 이면 모델 기본값
    # │        max_retries - 네트워크/서버 오류 시 재시도 횟수
    # └ API 키는 환경 변수 OPENAI_API_KEY 에서 자동으로 읽음
    llm = ChatOpenAI(model=common.CHAT_MODEL, temperature=0, max_retries=2)

    common.print_title("1) 단계별로 뜯어보기: 프롬프트가 실제로 어떻게 생겼나")

    question = "Who won the world cup?"
    # 검색만 따로 실행해서 확인
    docs = vector_store.similarity_search(question, k=2)
    # join 에 리스트 대신 (표현식 for ...) 형태의 "제너레이터" 를 바로 넘길 수도 있습니다 (Java 의 Stream 과 유사)
    context_text = "\n---\n".join(common.clean_text(d.page_content) for d in docs)
    # LLM 에 실제로 전달될 최종 프롬프트 출력 → "RAG 는 결국 프롬프트에 자료를 붙여 넣는 것" 임을 확인
    print(PROMPT_TEMPLATE.format(question=question, context=context_text[:1500] + " ..."))

    common.print_title("2) rag() 함수로 여러 질문 실행")

    # 마지막 질문은 문서에 없는 내용 → "I don't know" 가 나오는지 확인 (환각 방지 확인)
    questions = [
        "Who won the world cup?",
        "What was Virat Kohli's achievement in the Cup?",
        "Which teams qualified for the semi-finals?",
        "What is the capital of France?",
    ]
    for q in questions:
        # 튜플 언패킹: 반환된 두 값을 각각 변수에 받습니다
        contexts, answer = rag(q, vector_store, llm)
        print(f"Q: {q}")
        print(f"A: {answer}")
        # 어떤 근거로 답했는지 확인하기 위해 첫 번째 청크 앞부분도 같이 출력
        print(f"   (근거 청크 앞부분: {contexts[0][:120]}...)\n")

    # 실험 아이디어:
    # - k 를 1, 4 로 바꿔 보기 → 답변 품질/길이 변화 관찰
    # - PROMPT_TEMPLATE 에서 "ONLY" 제약을 지우고 마지막 질문을 다시 해 보기 → LLM 이 자기 지식으로 답함


if __name__ == "__main__":
    main()
