from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader, CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

FAISS_DB_DIR = "vectorstore"
# ドキュメントをロードするだけならAPIキーは不要
load_dotenv()

# Document Loaders
loader = DirectoryLoader(path="data", loader_cls=CSVLoader, glob='*.csv')
raw_docs = loader.load()
# データ件数
print(len(raw_docs))

# Document Transformers
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100, chunk_overlap=50)
docs = text_splitter.split_documents(raw_docs)

# Embedding
embeddings = OpenAIEmbeddings()
faiss_db = FAISS.from_documents(documents=docs, embedding=embeddings)

# Vector Store
faiss_db.save_local(FAISS_DB_DIR)

# Retrievers
retriever = faiss_db.as_retriever()

query = "おくやま"
context_docs = retriever.get_relevant_documents(query)
print(f"len={len(context_docs)}")

first_doc = context_docs[0]
print(f"metadata={first_doc.metadata}")
print(first_doc.page_content)
