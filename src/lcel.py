from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


import os
from chatbot_engine import chat, create_index
from dotenv import load_dotenv
from langchain.memory import ChatMessageHistory
load_dotenv()


prompt = PromptTemplate.from_template("""料理のレシピを考えてください。
料理名: {dish}""")
model = ChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=0)


class Recipe(BaseModel):
    ingredients: list[str] = Field(description="ingredients of the dish")
    steps: list[str] = Field(description="steps to make the dish")


output_parser = PydanticOutputParser(pydantic_object=Recipe)

# chain = prompt | model
chain = prompt | model | output_parser

result = chain.invoke({"dish": "カレー"})
print(result.content)
