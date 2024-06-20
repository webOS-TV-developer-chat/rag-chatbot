# API 키를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv
import os
import bs4
from langchain import hub   # pip install langchainhub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS   # pip install faiss-cpu
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # Open AI  유료
from langchain.embeddings import HuggingFaceBgeEmbeddings  # Hugging Face Embedding 무료
from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.llms import Ollama

# API 키 정보 로드
load_dotenv()


# 디버깅을 위한 프로젝트명을 기입합니다.
os.environ["LANGCHAIN_PROJECT"] = "4일차 실습"

# tracing 을 위해서는 아래 코드의 주석을 해제하고 실행합니다.
# os.environ["LANGCHAIN_TRACING_V2"] = true

bs4.SoupStrainer(
    "div",
    attrs={"class": ["newsct_article _article_body", "media_end_head_title"]}, # 클래스 명을 입력
)
# 뉴스기사 내용을 로드하고, 청크로 나누고, 인덱싱합니다.
loader = WebBaseLoader(
    web_paths=("https://n.news.naver.com/article/437/0000378416",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            "div",
            attrs={"class": ["newsct_article _article_body",
                             "media_end_head_title"]},
        )
    ),
)

docs = loader.load()
print(f"문서의 수: {len(docs)}")
print(docs)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

splits = text_splitter.split_documents(docs)
print(len(splits))


# 벡터스토어를 생성합니다.
#vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings())
# OPEN AI를 쓰면 Embbeding 시 비용 발생으로... Hugging face 무료 버전으로 대체해서 사용해 본다. 

# pip install sentence_transformers
vectorstore = FAISS.from_documents(documents=splits, embedding=HuggingFaceBgeEmbeddings())
# vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings(model="ada", openai_api_key="sk-"))


# 뉴스에 포함되어 있는 정보를 검색하고 생성합니다.
retriever = vectorstore.as_retriever()

prompt = hub.pull("rlm/rag-prompt")

print(prompt.messages[0].prompt.template)



class StreamCallback(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs):
        print(token, end="", flush=True)


# llm = ChatOpenAI(
#     model_name="gpt-4-turbo-preview",
#     temperature=0,
#     streaming=True,
#     callbacks=[StreamCallback()],
# )

# llm = Ollama(
#     model="llama2",
#     temperature=0,
#     streaming=True,
#     callbacks=[StreamCallback()],
# )

llm =  Ollama(model="llama2",callbacks=[StreamCallback()],temperature=0)


def format_docs(docs):
    # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
    return "\n\n".join(doc.page_content for doc in docs)


# 체인을 생성합니다.
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

question = "부영그룹의 출산 장려 정책에 대해 설명해주세요"
response = rag_chain.invoke(question)

print(f"[HUMAN]\n{question}\n")
print(f"[AI]\n{response}")



# # 컬렉션을 삭제합니다.
# vectorstore.delete_collection()