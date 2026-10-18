"""
[STEP 07] RAG 평가 (RAGAS)  -  원본: Chapters/Chapter-05/rag_evaluations.ipynb

"잘 동작하는 것 같다" 를 숫자로 확인하는 단계입니다. (Java 로 치면 RAG 용 단위 테스트 + 품질 지표)
RAGAS 는 LLM 을 "채점자" 로 사용해 아래 지표를 0~1 점수로 계산합니다.

- context_recall      : 정답(reference)에 필요한 정보가 검색된 청크에 들어 있는가?  → 검색 품질
- faithfulness        : 답변의 내용이 검색된 청크에 근거하는가? (환각이 없는가)       → 생성 품질
- answer_relevancy    : 답변이 질문에 맞는 내용인가?                                 → 생성 품질
- factual_correctness : 답변이 정답과 사실적으로 일치하는가?                          → 전체 품질

원본 노트북은 RAGAS 로 테스트 질문까지 자동 생성하지만, 여기서는 비용과 이해를 위해
사람이 직접 작성한 질문/정답 쌍(골든 데이터셋)을 사용합니다.

실행: python study/step07_evaluation.py   (OpenAI API 키 필요, step04 를 먼저 실행)
"""

# RAGAS 의 평가 데이터셋 클래스와 평가 실행 함수
from ragas import EvaluationDataset, evaluate
# LangChain 의 LLM/임베딩 객체를 RAGAS 가 쓸 수 있게 감싸는 어댑터 (Adapter 패턴)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
# 사용할 평가 지표 클래스들
from ragas.metrics import LLMContextRecall, Faithfulness, ResponseRelevancy, FactualCorrectness
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import common
# step06 에서 만든 함수를 재사용 (다른 파일의 함수도 모듈 import 로 바로 가져다 씁니다)
from step06_generation import load_vector_store, rag

# 골든 데이터셋: (질문, 사람이 확인한 정답) 튜플의 리스트
# 정답은 위키 원문(study/data/raw_text.txt)과 대조해 확인 후 필요하면 수정하세요
GOLDEN_SET = [
    ("Who won the 2023 Cricket World Cup?",
     "Australia won the 2023 Cricket World Cup, beating India by six wickets in the final."),
    ("Where was the final of the 2023 Cricket World Cup played?",
     "The final was played at the Narendra Modi Stadium in Ahmedabad."),
    ("How many runs did Virat Kohli score in the tournament?",
     "Virat Kohli scored 765 runs, the most by any batter in a single Cricket World Cup."),
    ("Which teams reached the semi-finals?",
     "India, South Africa, Australia and New Zealand reached the semi-finals."),
]


def main() -> None:
    common.load_api_key()

    # step06 과 같은 RAG 구성 준비
    vector_store = load_vector_store()
    llm = ChatOpenAI(model=common.CHAT_MODEL, temperature=0)

    common.print_title("1) RAG 실행 결과를 평가용 데이터로 수집")

    # 평가용 레코드를 담을 빈 리스트 (Java 의 new ArrayList<>())
    records = []
    # 리스트 요소가 튜플이므로 for 문에서 바로 두 변수로 분해
    for question, reference in GOLDEN_SET:
        # 우리 RAG 시스템으로 실제 답변 생성
        contexts, answer = rag(question, vector_store, llm)
        # ┌ [메서드] list.append(x) → 리스트 끝에 요소 추가 (Java: list.add(x))
        # └ dict 의 키 이름은 RAGAS 가 정해 둔 이름을 정확히 써야 합니다
        records.append({
            "user_input": question,          # 질문
            "retrieved_contexts": contexts,  # 검색된 청크들 (list[str])
            "response": answer,              # RAG 의 답변
            "reference": reference,          # 정답
        })
        print(f"Q: {question}\nA: {answer}\n")

    # ┌ [클래스 메서드] EvaluationDataset.from_list(list[dict])
    # │ 역할 : dict 리스트를 RAGAS 평가 데이터셋으로 변환
    # └ 내부 : dict 하나하나를 SingleTurnSample(질문 1개 + 답변 1개짜리 샘플) 객체로 검증·변환
    dataset = EvaluationDataset.from_list(records)

    common.print_title("2) RAGAS 로 채점 (LLM 이 채점하므로 1~2분 걸릴 수 있습니다)")

    # ┌ [생성자] LangchainLLMWrapper(llm) / LangchainEmbeddingsWrapper(embeddings)
    # └ 역할 : LangChain 객체를 RAGAS 가 요구하는 인터페이스로 감싸는 어댑터 (Java 의 Adapter 패턴)
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model=common.CHAT_MODEL))
    evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=common.EMBEDDING_MODEL))

    # 지표 클래스별 채점 방식 (모두 LLM 이 채점 과정에 참여)
    # - LLMContextRecall   : 정답(reference)을 문장(주장) 단위로 나눔 → 각 주장이 검색된 청크로 뒷받침되는지 판정
    #                        → 뒷받침되는 비율
    # - Faithfulness       : 답변(response)을 주장 단위로 나눔 → 각 주장이 청크에서 유추 가능한지 판정 → 비율
    # - ResponseRelevancy  : 답변만 보고 LLM 이 "원래 질문" 을 역으로 추측 → 추측 질문과 실제 질문의
    #                        임베딩 유사도 평균 (그래서 embeddings 가 필요)
    # - FactualCorrectness : 답변과 정답을 각각 주장 단위로 나눠 서로 대조 → 정밀도/재현율 기반 F1 점수
    #
    # ┌ [함수] evaluate(dataset, metrics, llm, embeddings)
    # │ 역할 : 데이터셋의 모든 샘플에 대해 모든 지표를 계산
    # │ 반환 : EvaluationResult - print 하면 지표별 평균, to_pandas() 로 샘플별 상세 점수
    # └ 내부 : (샘플 수 × 지표 수) 만큼의 채점 작업을 비동기로 병렬 실행 → 진행률 바가 표시됨
    result = evaluate(
        dataset=dataset,
        metrics=[LLMContextRecall(), Faithfulness(), ResponseRelevancy(), FactualCorrectness()],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    # 전체 평균 점수 출력
    print(result)

    common.print_title("3) 질문별 상세 점수")

    # ┌ [메서드] result.to_pandas()
    # │ 역할 : 결과를 pandas DataFrame(엑셀 같은 표 객체)으로 변환. 행 = 샘플, 열 = 입력값 + 지표 점수
    # │ [메서드] df.select_dtypes("number") → 숫자형 열만 골라낸 새 표 (.columns 로 열 이름 목록)
    # │ [문법]   df[["열1", "열2"]]         → 지정한 열만 골라낸 새 표 (SQL 의 SELECT 열1, 열2)
    # └ [메서드] df.to_string()            → 표 전체를 콘솔 출력용 문자열로 변환
    df = result.to_pandas()
    score_columns = list(df.select_dtypes("number").columns)
    # 리스트 + 리스트 는 두 리스트를 이어 붙인 새 리스트 (Java: Stream.concat)
    print(df[["user_input"] + score_columns].to_string())

    # 해석 팁:
    # - context_recall 이 낮다 → 검색이 문제. chunk_size, k, 임베딩 모델, 하이브리드 검색(step09) 검토
    # - faithfulness 가 낮다   → 생성이 문제. 프롬프트 제약 강화, temperature 낮추기
    # - 설정을 바꿀 때마다 이 스크립트를 다시 돌려 점수를 비교하는 것이 "RAG 튜닝" 의 기본 루프입니다


if __name__ == "__main__":
    main()
