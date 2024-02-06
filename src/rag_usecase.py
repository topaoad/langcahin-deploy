from dotenv import load_dotenv
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
FAISS_DB_DIR = "vectorstore"

load_dotenv()

# WebBaseLoaderは引数を取ることができるようです
# docs = loader.load()
# loader = WebBaseLoader(
#     web_paths=("https://js.langchain.com/doacs/use_cases/question_answering/",),
#     bs_kwargs=dict(
#         parse_only=bs4.SoupStrainer(
#             class_=("post-content", "post-title", "post-header")
#         )
#     ),
# )
# loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
loader = WebBaseLoader(
    "https://www.notion.so/gearaise/1b4ae48842ec4a709a7545fc9b194f22?v=f9df2c49ec1b4a28a36b6ab033f7dea2")
docs = loader.load()
# print("len(docs[0].page_content)", len(docs[0].page_content))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(
    documents=splits, embedding=OpenAIEmbeddings())

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ユーザーからの質問を受け取る
user_question = input("質問を入力してください: ")
# RAGチェーンを使用して質問に対する回答を生成
result = rag_chain.invoke(user_question)

if result:
    print(result)
else:
    print("エラーが発生しました。正しい回答が得られませんでした。")
