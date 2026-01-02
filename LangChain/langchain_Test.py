import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import dotenv
from pyexpat.errors import messages

dotenv.load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL")
llm = ChatOpenAI(model= "deepseek-chat")

prompt = ChatPromptTemplate.from_messages([
    ("system","你是世界级别的技术文档编写者，输出格式要求：{format_instructions}"),
    ("user","{input}")
])
output_parse = JsonOutputParser()

chain = prompt | llm | output_parse
message =  chain.invoke({"input":"大模型中的langchain是什么?","format_instructions":output_parse.get_format_instructions()})
print(message)


