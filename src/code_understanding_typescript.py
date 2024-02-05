# from git import Repo
from langchain.text_splitter import Language
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationSummaryMemory
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp


load_dotenv()

# Clone ※うまくいかないので手動でクローンして使用
repo_path = "./"
# repo = Repo.clone_from(
#     "https://github.com/langchain-ai/langchain", to_path=repo_path)

# ドキュメントの読み取り
loader = GenericLoader.from_filesystem(
    # 対象のリポジトリのパス
    repo_path + "/repos/itk-develop-first",
    glob="**/*",
    suffixes=[".ts", ".tsx"],
    exclude=[""],
    parser=LanguageParser(),
)
documents = loader.load()
len(documents)

# 分割
python_splitter = RecursiveCharacterTextSplitter.from_language(
    chunk_size=2000, chunk_overlap=200
)
texts = python_splitter.split_documents(documents)
len(texts)

db = Chroma.from_documents(texts, OpenAIEmbeddings(disallowed_special=()))
retriever = db.as_retriever(
    search_type="mmr",  # Also test "similarity"
    search_kwargs={"k": 8},
)

llm = ChatOpenAI(model_name="gpt-4")
memory = ConversationSummaryMemory(
    llm=llm, memory_key="chat_history", return_messages=True
)
qa = ConversationalRetrievalChain.from_llm(
    llm, retriever=retriever, memory=memory)

# question = "ここには質問を直接打ち込む"
# こちらはターミナルから質問を入力する
question = input("質問を入力してください: ")
result = qa(question)
result["answer"]
print(result["answer"])
