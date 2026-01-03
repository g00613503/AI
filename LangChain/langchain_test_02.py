from langchain.chains.summarize.refine_prompts import prompt_template
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from base_model import *
import bs4


## 使用向量数据库存储


def load_content_from_url(url):
    print(f"正在加载网页内容: {url}")
    try:
        loader = WebBaseLoader(url)
        documents = loader.load()
        print(f"成功加载文档，总字符数: {len(documents[0].page_content)}")
    except Exception as e:
        print(f"网页加载失败: {e}")
        return None

    # 4. 分割文本为适合处理的块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # 每个块的大小
        chunk_overlap=200,  # 块之间的重叠部分
        length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )

    splits = text_splitter.split_documents(documents)
    print(f"文档被分割为 {len(splits)} 个文本块")
    return splits


if __name__ == "__main__":
    # 示例：分析维基百科页面
    url = "https://movie.douban.com/explore"
    documents = load_content_from_url(url)
    embedding = OpenAIEmbeddings(model="deepseek-chat")
    vector = FAISS.from_documents(documents, embedding)
    retriever = vector.as_retriever()
    retriever.search_kwargs = {"k": 3}
    docs = retriever.invoke("2025年美国的热门电影是什么？")
    prompt_template = ("""
    你是一个问答机器人。
    你的任务是根据下述给定的已知信息回答用户问题。
    确保你的回复完全依据下述已知信息，不要编造答案。
    如果下述已知信息不足以回答用户的问题，请直接回复"我无法回答你的问题"。
    
    已知信息：
    {info}
    
    用户问：
    {question}
    
    请用中文回答用户问题。
    """)
    template = PromptTemplate.from_template(prompt_template)
    prompt = template.format(info=docs, question="2025年美国的热门电影是什么？")
    response = llm.invoke(prompt)
    print(response.content)
