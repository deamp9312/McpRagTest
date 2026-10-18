"""
[STEP 01] 문서 로딩 (인덱싱 파이프라인 1단계: Load)  -  원본: Chapters/Chapter-03/indexing_pipeline.ipynb

RAG 의 첫 단계는 "지식의 원천" 을 가져오는 것입니다.
여기서는 위키백과 HTML 페이지를 내려받고, HTML 태그를 걷어 내 순수 텍스트로 바꿉니다.

터미널을 새로 열 때마다 가상환경을 한 번 켜 주세요. 켜고 나면 python이 이
  가상환경의 Python 3.12를 가리킵니다.
  cd /Users/dongchancho/Desktop/ai_study/A-Simple-Guide-to-RAG
  source .venv/bin/activate
  python study/step01_load_document.py
  Java로 치면 프로젝트별 JDK와 의존성을 따로 쓰는 것과 비슷합니다. 끌 때는 deactivate를 입력하면
  됩니다.

실행: python study/step01_load_document.py   (API 키 불필요)
결과: study/data/raw_text.txt
"""

# textwrap: 긴 문자열을 지정한 폭으로 줄바꿈해 주는 표준 라이브러리 (출력용)
import textwrap

# AsyncHtmlLoader: URL 목록을 받아 HTML 을 비동기로 내려받아 Document 객체로 만들어 주는 LangChain 로더
from langchain_community.document_loaders import AsyncHtmlLoader
# Html2TextTransformer: HTML 을 사람이 읽는 텍스트(마크다운 비슷한 형태)로 변환하는 변환기
from langchain_community.document_transformers import Html2TextTransformer

# 같은 폴더의 common.py 를 import (common.XXX 로 사용)
import common


# def 는 함수(메서드) 정의 키워드입니다. 클래스 밖에 있으므로 Java의 static 메서드라고 생각하면 됩니다
def main() -> None:
    common.print_title("1) 웹 페이지(HTML) 내려받기")

    # ┌ [생성자] AsyncHtmlLoader(web_path)
    # │ 역할 : 내려받을 URL 을 기억해 두는 로더 객체 생성 (이 시점엔 네트워크 요청 없음)
    # │ 입력 : URL 문자열 1개 또는 URL 리스트. 옵션으로 requests_per_second(초당 요청 수, 기본 2),
    # │        header_template(HTTP 헤더), verify_ssl 등이 있음
    # └ Python 은 `new` 없이 클래스명(...) 만으로 인스턴스를 만듭니다
    loader = AsyncHtmlLoader([common.SOURCE_URL])

    # ┌ [메서드] loader.load()
    # │ 역할 : 등록된 모든 URL 의 HTML 을 내려받아 Document 리스트로 반환
    # │ 반환 : list[Document] - URL 1개당 Document 1개 (여기선 길이 1)
    # │        Document.page_content = HTML 원문 전체 문자열
    # │        Document.metadata     = {"source": URL, "title": <title>, "language": ...} (dict)
    # └ 내부 : aiohttp 로 여러 URL 을 "동시에" 요청(비동기) → BeautifulSoup 로 title 등 메타데이터 추출
    #          Java 로 치면 HttpClient.sendAsync 여러 개를 CompletableFuture.allOf 로 기다리는 것과 비슷
    html_docs = loader.load()

    # 리스트의 첫 번째 요소를 꺼냅니다 (Java의 list.get(0))
    html_doc = html_docs[0]
    # len(...) : 길이를 구하는 내장 함수 (Java의 str.length() / list.size() 를 하나로 통합한 것)
    # :,       : f-string 포맷 지정자로 천 단위 콤마 출력
    print(f"HTML 길이: {len(html_doc.page_content):,} 글자")
    # metadata 는 dict(Java 의 Map<String,Object>) 입니다
    print(f"메타데이터: {html_doc.metadata}")
    # [:300] 은 슬라이싱 문법으로 앞 300글자를 잘라냅니다 (Java의 substring(0, 300), 길이 초과해도 예외 없음)
    print("HTML 앞부분 300자:\n", html_doc.page_content[:300])

    common.print_title("2) HTML → 순수 텍스트 변환")

    # ┌ [생성자] Html2TextTransformer(ignore_links=True, ignore_images=True)
    # └ 역할 : HTML→텍스트 변환기 생성. 기본값으로 링크 URL 과 이미지는 결과에서 제외
    html2text = Html2TextTransformer()

    # ┌ [메서드] html2text.transform_documents(documents)
    # │ 역할 : Document 들의 page_content(HTML)를 읽기 쉬운 텍스트로 바꾼 "새 Document 리스트" 반환
    # │ 입력 : list[Document]
    # │ 반환 : list[Document] - metadata 는 그대로 유지, page_content 만 텍스트로 교체
    # └ 내부 : html2text 라이브러리를 사용. <h2> → "## ", <li> → "* " 처럼 마크다운 비슷하게 변환하고
    #          <script>, <style> 같은 비본문 요소는 제거. (LangChain 의 "Transformer" = 문서 가공기)
    text_docs = html2text.transform_documents(html_docs)
    # 변환 결과의 본문 텍스트만 꺼냅니다
    text = text_docs[0].page_content

    # HTML 대비 텍스트가 얼마나 줄었는지 확인 (태그/스크립트가 빠지므로 크게 줄어듭니다)
    print(f"텍스트 길이: {len(text):,} 글자")
    # ┌ [메서드] textwrap.fill(text, width)
    # └ 역할 : 문자열을 width 글자 폭에 맞게 줄바꿈한 새 문자열 반환 (콘솔 출력 가독성용)
    print(textwrap.fill(text[:1000], width=100))

    common.print_title("3) 다음 단계를 위해 파일로 저장")

    # data 디렉터리가 없으면 생성 (Java: Files.createDirectories)
    common.DATA_DIR.mkdir(parents=True, exist_ok=True)
    # 텍스트를 UTF-8 로 파일에 기록 (Java: Files.writeString(path, text, UTF_8))
    common.RAW_TEXT_FILE.write_text(text, encoding="utf-8")
    print(f"저장 완료 → {common.RAW_TEXT_FILE}")


# 이 파일을 직접 실행했을 때만 main() 을 호출합니다 (다른 파일에서 import 할 때는 실행 안 됨)
# __name__ 은 모듈 이름이 담긴 내장 변수로, 직접 실행된 파일이면 "__main__" 이 됩니다
# Java의 `public static void main(String[] args)` 진입점과 같은 역할을 하는 Python 관용구입니다
if __name__ == "__main__":
    main()
