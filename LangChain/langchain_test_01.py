
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from base_model import llm

prompt = ChatPromptTemplate.from_messages([
    ("system","你是世界级别的技术文档编写者，输出格式要求：{format_instructions}"),
    ("user","{input}")
])
output_parse = JsonOutputParser()

chain = prompt | llm | output_parse
message =  chain.invoke({"input":"大模型中的langchain是什么?","format_instructions":output_parse.get_format_instructions()})
print(message)


