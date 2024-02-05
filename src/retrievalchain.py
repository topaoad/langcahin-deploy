# quick startの部分を複合的に実装
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are world class technical documentation writer."),
    ("user", "{input}")
])
# 取得の更新
prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
])

# load_dotenv()があれば引数でopenaiのapi keyを渡す必要はない
llm = ChatOpenAI()
output_parser = StrOutputParser()
chain = prompt | llm | output_parser
# chain.invoke({"input": "how can langsmith help with testing?"})

embeddings = OpenAIEmbeddings()
loader = WebBaseLoader("https://docs.smith.langchain.com/overview")

docs = loader.load()
# print(len(docs))
# print(docs[0].page_content)
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)


# prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

# <context>
# {context}
# </context>

# Question: {input}""")


# document_chain = create_stuff_documents_chain(llm, prompt)

# 取得の更新
# response = retrieval_chain.invoke(
#     {"input": "how can langsmith help with testing?",
#      "context": "hello world!"
#      })
# print("response", response["answer"])

retriever = vector.as_retriever()
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Answer the user's questions based on the below context:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])
retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)
chat_history = [HumanMessage(
    content="Next.jsの有効性について"), AIMessage(content="もちろん!")]
retriever_chain.invoke({
    "chat_history": chat_history,
    "input": "Tell me how"
})
