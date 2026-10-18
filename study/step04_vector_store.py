"""
[STEP 04] 벡터 저장소 구축 (인덱싱 파이프라인 4단계: Store)  -  원본: Chapters/Chapter-03/indexing_pipeline.ipynb

step02 의 청크 전체를 임베딩하여 FAISS 벡터 인덱스에 저장하고 디스크에 보관합니다.
- FAISS: Meta 가 만든 벡터 유사도 검색 라이브러리. "벡터 전용 DB" 라고 생각하면 됩니다.
- 인덱스(벡터 배열) + 도큐먼트 저장소(원문 텍스트) + 둘을 잇는 ID 매핑 세 부분으로 구성됩니다.

    faiss index        index_to_docstore_id        docstore
    [0] [0.1, ...]  →  {0: "uuid-a",          →    {"uuid-a": Document(청크0 원문),
    [1] [0.3, ...]  →   1: "uuid-b", ...}           "uuid-b": Document(청크1 원문), ...}

실행: python study/step04_vector_store.py   (OpenAI API 키 필요, 청크 수십~수백 개 임베딩 → 비용 매우 적음)
결과: study/data/CWC_index.faiss, study/data/CWC_index.pkl
"""

# faiss: 벡터 인덱스 라이브러리 본체 (C++ 로 구현, 파이썬에서 호출)
import faiss
# Document: LangChain 의 문서 클래스 (page_content + metadata 를 담는 DTO)
from langchain_core.documents import Document
# InMemoryDocstore: ID → Document 를 메모리에 보관하는 저장소 (Java 의 HashMap<String, Document>)
from langchain_community.docstore.in_memory import InMemoryDocstore
# FAISS: faiss 인덱스를 LangChain 공통 인터페이스(VectorStore)로 감싼 클래스
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

import common


def main() -> None:
    common.load_api_key()

    common.print_title("1) step02 에서 저장한 청크 불러오기")

    # JSON → list[dict]
    raw_chunks = common.load_json(common.CHUNKS_FILE)
    # ┌ [생성자] Document(page_content, metadata)
    # └ 역할 : 텍스트와 부가정보를 담는 LangChain 표준 객체 생성 (Java 라면 record Document(String content, Map meta))
    # c["content"] 는 dict 조회로 Java 의 map.get("content") 와 같습니다 (단, 키가 없으면 KeyError 예외)
    documents = [Document(page_content=c["content"], metadata=c["metadata"]) for c in raw_chunks]
    print(f"청크 {len(documents)}개 로드")

    common.print_title("2) 빈 FAISS 인덱스 생성")

    # 임베딩 모델 준비 (step03 참고)
    embeddings = OpenAIEmbeddings(model=common.EMBEDDING_MODEL)
    # 벡터 차원 수를 알기 위해 짧은 문자열을 한 번 임베딩해 봅니다 (1536)
    dimension = len(embeddings.embed_query("hello"))

    # ┌ [생성자] faiss.IndexFlatIP(d)
    # │ 역할 : d 차원 벡터를 저장하고 "내적(Inner Product)" 이 큰 순서로 찾아 주는 인덱스 생성
    # │ 입력 : d - 벡터 차원 수 (임베딩 모델의 차원과 반드시 같아야 함)
    # └ 내부 : Flat = 근사 없이 저장된 모든 벡터와 하나하나 비교(완전 탐색) → 정확하지만 대용량엔 느림
    #          (비슷한 종류: IndexFlatL2 = 유클리드 거리 기준, IndexHNSWFlat/IndexIVFFlat = 대용량용 근사 검색)
    index = faiss.IndexFlatIP(dimension)
    # type(obj).__name__ : 객체의 클래스 이름 (Java: obj.getClass().getSimpleName())
    print(f"벡터 차원: {dimension}, 인덱스 타입: {type(index).__name__}")

    # ┌ [생성자] FAISS(embedding_function, index, docstore, index_to_docstore_id)
    # │ 역할 : faiss 인덱스 + 원문 저장소 + 임베딩 모델을 묶어 "검색 가능한 벡터 저장소" 생성
    # │ 입력 : embedding_function  - 텍스트를 벡터로 바꿀 객체 (문서 추가/질문 검색 시 자동 사용)
    # │        index               - 벡터가 실제로 저장될 faiss 인덱스
    # │        docstore            - 원문 Document 보관소
    # │        index_to_docstore_id- faiss 내부 번호(0,1,2...) → docstore ID 매핑 dict
    # └ Java 로 치면 Repository 에 DataSource, Mapper 등을 생성자 주입하는 것과 비슷
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),     # 빈 저장소로 시작
        index_to_docstore_id={},         # 빈 dict 로 시작
    )

    common.print_title("3) 청크를 임베딩하여 인덱스에 추가")

    # ┌ [메서드] vector_store.add_documents(documents, ids=None)
    # │ 역할 : Document 들을 벡터화하여 저장소에 추가
    # │ 입력 : documents - list[Document], ids - 직접 지정할 ID 리스트 (생략 시 UUID 자동 생성)
    # │ 반환 : list[str] - 각 문서에 부여된 ID
    # └ 내부 : ① embeddings.embed_documents(본문들) 로 벡터 생성 (API 호출)
    #          ② index.add(벡터 배열) 로 faiss 에 추가
    #          ③ docstore 에 {ID: Document} 저장, index_to_docstore_id 에 {번호: ID} 기록
    ids = vector_store.add_documents(documents=documents)
    # index.ntotal: faiss 인덱스에 저장된 벡터 개수 (Java 의 size())
    print(f"추가된 벡터 수: {vector_store.index.ntotal}, 첫 ID 예시: {ids[0]}")

    common.print_title("4) 디스크에 저장")

    # ┌ [메서드] vector_store.save_local(folder_path, index_name="index")
    # │ 역할 : 벡터 저장소를 파일 2개로 저장
    # │ 결과 : {index_name}.faiss - faiss 인덱스(벡터) 바이너리 (faiss.write_index 사용)
    # │        {index_name}.pkl   - docstore + ID 매핑을 pickle(파이썬 객체 직렬화)로 저장
    # └ Java 의 Serializable 객체를 ObjectOutputStream 으로 저장하는 것과 같은 개념
    # str(Path): Path 객체를 문자열 경로로 변환 (일부 라이브러리는 문자열만 받음)
    vector_store.save_local(folder_path=str(common.DATA_DIR), index_name=common.INDEX_NAME)
    print(f"저장 완료 → {common.DATA_DIR}/{common.INDEX_NAME}.faiss / .pkl")

    common.print_title("5) 저장한 인덱스로 검색 테스트")

    query = "Who won the World Cup final?"
    # ┌ [메서드] vector_store.similarity_search_with_score(query, k=4)
    # │ 역할 : 질문과 가장 비슷한 청크 k개를 점수와 함께 반환
    # │ 입력 : query - 질문 문자열, k - 가져올 개수
    # │ 반환 : list[tuple[Document, float]] - (문서, 점수) 쌍의 리스트, 비슷한 순서
    # └ 내부 : ① embed_query(query) 로 질문 벡터 생성 → ② index.search(질문벡터, k) 로 가까운 번호 k개
    #          → ③ index_to_docstore_id 와 docstore 로 번호를 Document 로 변환
    #          점수 의미는 인덱스 종류에 따라 다름: IndexFlatIP=내적(클수록 유사), IndexFlatL2=거리(작을수록 유사)
    #          (점수가 필요 없으면 similarity_search(query, k) → list[Document] 를 사용)
    results = vector_store.similarity_search_with_score(query, k=3)
    print(f"질문: {query}\n")
    # enumerate(results, start=1) 은 1부터 번호를 붙여 줍니다. (doc, score) 로 튜플을 바로 분해
    for rank, (doc, score) in enumerate(results, start=1):
        print(f"[{rank}] score={score:.4f} chunk_id={doc.metadata['chunk_id']}")
        # 본문 앞 300자를 정리해서 출력
        print("    " + common.clean_text(doc.page_content)[:300] + "...\n")


if __name__ == "__main__":
    main()
