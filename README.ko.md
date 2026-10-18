# 검색 증강 생성(RAG) 간단 가이드

[English](./README.md) | **한국어**

 이 저장소는 [Manning Publications](https://www.manning.com/?utm_source=kimothi&utm_medium=affiliate&utm_campaign=affiliate&a_aid=kimothi)에서 출간한 책 [A Simple Guide to Retrieval Augmented Generation](https://mng.bz/8wdg)에서 다루는 예제와 그림의 소스 코드입니다.

<a href="https://mng.bz/8wdg" target="_blank">
    <img src="./Assets/Images/MEAP-HI.png" alt="New MEAP" style="width: 50%;" />
</a>

 검색 증강 생성(Retrieval Augmented Generation, RAG)은 응용 생성형 AI의 지형을 형성하는 핵심 기법입니다. Lewis 등이 기념비적인 논문 Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks(https://arxiv.org/abs/2005.11401)에서 처음 소개한 새로운 개념인 RAG는, 대규모 언어 모델(LLM) 출력의 신뢰성과 신뢰도를 높이는 핵심 기반으로 빠르게 자리 잡았습니다.

책 소개
---
이 책은 검색 증강 생성에 대해 쉽고도 포괄적인 입문서를 찾는 초보자를 위해 특별히 설계된 기초 가이드입니다. 이 책은 RAG의 세부적인 기술 사항을 깊이 파고들기보다는 전반적인 개요를 제공합니다. 생성형 AI 기반 애플리케이션 개발에 관심 있는 데이터 사이언티스트, 데이터 엔지니어, ML 엔지니어, 소프트웨어 개발자, 기술 리더, 학생 및 학계 연구자에게 유익한 책입니다. 이 책을 마치면 다음을 기대할 수 있습니다:

- RAG의 기본 원리, RAG 기반 시스템의 구성 요소 및 실제 활용 사례에 대한 탄탄한 이해

- RAG를 위한 비모수적(non-parametric) 지식 베이스가 무엇을 의미하며 어떻게 만들어지는지 이해

- 인덱싱 파이프라인과 생성 파이프라인의 세부 사항을 포함하여 RAG 기반 시스템 개발에 대한 지식 습득
- RAG 기반 시스템의 평가 및 모듈화된 평가 전략에 대한 깊은 통찰

- 고급 RAG 전략과 진화하는 생태계에 대한 이해

- 프로덕션 수준의 RAG 시스템을 구축하고 배포하기 위한 도구, 기술 및 프레임워크에 대한 지식 습득

- RAG의 현재 한계를 이해하고, 추가로 탐구할 만한 주목받는 최신 기법 접하기

__참고: 이 책은 아직 집필 중이며 향후 몇 달 내에 완성될 예정입니다__

[공식 소스 코드 저장소](https://github.com/abhinav-kimothi/A-Simple-Introduction-to-RAG) 링크

[manning.com에서 MEAP 참여하기](https://mng.bz/8wdg) 링크

이 저장소의 사본을 다운로드하려면 [Download ZIP](https://github.com/abhinav-kimothi/A-Simple-Introduction-to-RAG/archive/refs/heads/main.zip) 버튼을 클릭하거나 터미널에서 다음 명령을 실행하세요:

```
git clone https://github.com/abhinav-kimothi/A-Simple-Introduction-to-RAG.git
```


목차
---

책의 처음 세 장은 Manning 얼리 액세스 프로그램(MEAP)의 일부로 공개되었습니다. [여기에서 MEAP에 참여](https://mng.bz/8wdg)할 수 있습니다.

- 1장 - 대규모 언어 모델과 검색 증강 생성의 필요성 [초안 공개]

- 2장 - RAG 기반 시스템과 그 설계 [초안 공개]

- 3장 - 인덱싱 파이프라인: RAG 기반 애플리케이션을 위한 지식 베이스 구축 [초안 공개] [노트북](./Chapters/Chapter-03/indexing_pipeline.ipynb)

- 4장 - 생성 파이프라인: 맥락에 맞는 응답을 위한 실시간 상호작용 [노트북](./Chapters/Chapter-04/generation_pipeline.ipynb)

- 5장 - RAG 평가: 정확성, 관련성, 충실성 확인 [노트북](./Chapters/Chapter-05/rag_evaluations.ipynb)

- 6장 - RAG 시스템의 발전: Naive에서 Advanced, Modular로 [노트북](./Chapters/Chapter-06/advanced_rag.ipynb.ipynb)

- 7장 - 진화하는 RAGOps 스택: RAG를 가능하게 하는 기술들

- 8장 - 세부 주제: 파인튜닝과의 비교, 멀티모달 및 에이전틱 RAG

- 9장 - 최첨단: 모범 사례와 추가 탐구

MEAP에 참여해야 하는 이유
---
Manning 얼리 액세스 프로그램에 참여하면 다음 혜택을 받을 수 있습니다:
- 책의 현재 초안과 향후 모든 업데이트에 즉시 접근
- 피드백을 제공하여 최종 내용 형성에 참여할 기회
- 독점 할인 및 얼리버드 혜택

코드
---
코드 스니펫은 Chapters 디렉터리에 [장별로](./Chapters) 정리되어 있습니다.



- 1장 - 코드 없음

- 2장 - 코드 없음

- [3장 - 이 노트북](./Chapters/Chapter-03/indexing_pipeline.ipynb)은 인덱싱 파이프라인을 설명합니다.

    2023 크리켓 월드컵에 관한 Wikipedia 문서를 바탕으로 지식 베이스를 만듭니다. __AsyncHtmlLoader__와 __Html2TextTransformer__로 문서를 불러오고, __RecursiveCharacterTextSplitter__로 텍스트를 청크로 나누며, OpenAI의 __text-embedding-3-large__로 청크를 벡터로 변환하고, __FAISS__를 벡터 인덱스로 사용하여 임베딩을 저장합니다.

- [4장 - 이 노트북](./Chapters/Chapter-04/indexing_pipeline.ipynb)은 생성 파이프라인을 설명합니다.

    3장에서 2023 크리켓 월드컵 Wikipedia 문서로 만든 지식 베이스를 사용합니다. __FAISS__ 인덱스를 불러오고 __유사도 검색(similarity search)__ 함수로 청크를 검색합니다. 그런 다음 검색된 청크로 사용자 질의를 증강하고, OpenAI의 __GPT 4o__ 모델로 응답을 생성합니다.

    이 노트북에는 사용자가 물어볼 수 있는 다양한 질의에 대한 답변을 생성하는 데 사용할 수 있는 함수도 포함되어 있습니다.

    또한 이 장에는 TF-IDF와 BM25를 검색 알고리즘으로 사용하는 방법을 보여주는 [또 다른 노트북](./Chapters/Chapter-04/xtra_tfidf_bm25_retriever.ipynb)이 있습니다.

- [5장 - 이 노트북](./Chapters/Chapter-05/rag_evaluations.ipynb)은 3장과 4장에서 만든 RAG 파이프라인을 __RAGAS__ 프레임워크로 평가합니다.

    또한 이 장에는 LangChain Benchmarks를 사용해 LangChain QnA 문서에 대해 RAG 파이프라인을 벤치마킹하는 [노트북](./Chapters/Chapter-05/xtra_benchmarking.ipynb)이 포함되어 있습니다.

- [6장 - 이 노트북](./Chapters/Chapter-06/advanced_rag.ipynb)은 선별된 인덱스 최적화, 질의 최적화, 검색 전략, 검색 후 압축 기법을 보여줍니다.

__참고: 이 저장소는 작업 중(WIP)이며 이후 장들은 계속해서 공개될 예정입니다__

설정
---

이 저장소를 로컬 머신에 클론합니다:

```
git clone https://github.com/abhinav-kimothi/A-Simple-Introduction-to-RAG.git
```

클론한 저장소로 이동합니다:

    
    cd A-Simple-Introduction-to-RAG

다른 프로젝트나 시스템 전역 Python 패키지와의 충돌을 피하기 위해 가상 환경 사용을 권장합니다.

다음 명령을 실행하여 myenv라는 이름의 가상 환경을 만듭니다(원하는 이름으로 지정해도 됩니다):

```
python3 -m venv .myenv
```
가상 환경 활성화:

- Windows에서는 다음 명령으로 가상 환경을 활성화합니다:

```
    .myenv\Scripts\activate.bat
```

- macOS와 Linux에서는 다음 명령으로 활성화합니다:

```
    source venv/bin/activate
```

다음 pip 설치 명령을 실행하여 requirements.txt의 패키지를 설치합니다:


```
pip install -r requirements.txt
```

_권장_: API 키를 .env 파일에 저장하세요:
```
OPENAI_API_KEY=<YOUR API KEY>
LANGCHAIN_API_KEY=<YOUR API KEY>

### 구조는 이 저장소의 .\example_dot_env 파일을 참고하세요. 파일명을 .env로 바꾸는 것을 잊지 마세요
```

이 저장소의 노트북은 __Python 3.11.1 초과 버전__이 필요합니다.

피드백 및 기여
---
이 코드와 책 전반에 대한 여러분의 생각을 듣고 싶습니다. 모든 피드백을 감사히 환영합니다. 생각, 질문, 비평, 아이디어를 [이 저장소의 Discussion 포럼](https://github.com/abhinav-kimothi/A-Simple-Introduction-to-RAG/discussions)에 올리거나, [MEAP](https://mng.bz/8wdg)(https://mng.bz/8wdg)을 구매하셨다면 [manning.com의 라이브북](https://livebook.manning.com/book/a-simple-guide-to-retrieval-augmented-generation)에서 피드백을 남기실 수도 있습니다.

코드에서 오류나 문제를 발견하시면 [이 저장소에 이슈](https://github.com/abhinav-kimothi/A-Simple-Introduction-to-RAG/issues)를 등록해 주세요.

저자 소개
---
안녕하세요! 저는 Abhinav입니다! 저는 기업가이자 Yarnit의 인공지능 부문 부사장입니다. 데이터 사이언스, 머신러닝, AI 분야에서 15년 넘게 컨설팅과 리더십 역할을 맡아 왔습니다. 현재는 응용 생성형 AI 분야에 집중하며, 맥락 지능(contextual intelligence)을 통해 기업의 요구를 해결하는 데 주력하고 있습니다. AI의 발전에 열정을 가지고 있으며, 한계를 넓히고 세상에 긍정적인 영향을 만들기 위해 새로운 기술을 끊임없이 탐구하고 있습니다. 함께 미래를 만들어 갑시다!

#### 이야기 나누고 싶으시다면 언제든 연락 주세요

[![GitHub followers](https://img.shields.io/badge/Github-000000?style=for-the-badge&logo=github&logoColor=black&color=orange)](https://github.com/abhinav-kimothi)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-000000?style=for-the-badge&logo=linkedin&logoColor=orange&color=black)](https://www.linkedin.com/comm/mynetwork/discovery-see-all?usecase=PEOPLE_FOLLOWS&followMember=abhinav-kimothi)
[![Medium](https://img.shields.io/badge/Medium-000000?style=for-the-badge&logo=medium&logoColor=black&color=orange)](https://medium.com/@abhinavkimothi)
[![Insta](https://img.shields.io/badge/Instagram-000000?style=for-the-badge&logo=instagram&logoColor=orange&color=black)](https://www.instagram.com/akaiworks/)
[![Mail](https://img.shields.io/badge/email-000000?style=for-the-badge&logo=gmail&logoColor=black&color=orange)](mailto:abhinav.kimothi.ds@gmail.com)
[![X](https://img.shields.io/badge/Follow-000000?style=for-the-badge&logo=X&logoColor=orange&color=black)](https://twitter.com/abhinav_kimothi)
[![Linktree](https://img.shields.io/badge/Linktree-000000?style=for-the-badge&logo=linktree&logoColor=black&color=orange)](https://linktr.ee/abhinavkimothi)
[![Gumroad](https://img.shields.io/badge/Gumroad-000000?style=for-the-badge&logo=gumroad&logoColor=orange&color=black)](https://abhinavkimothi.gumroad.com/)


[아직 구독하지 않으셨다면, 여기에서 A Simple Guide to Retrieval Augmented Generation의 MEAP을 구독해 주세요](https://mng.bz/8wdg)

<a href="https://mng.bz/8wdg" target="_blank">
    <img src="./Assets/Images/NewMEAPFooter.png" alt="New MEAP" style="width: 100%;" />
</a>

---
