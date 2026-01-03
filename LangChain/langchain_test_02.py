from langchain.chains.summarize.refine_prompts import prompt_template
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
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
    url = "https://zh.wikipedia.org/wiki/%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD"
    splits = load_content_from_url(url)
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-zh",
        model_kwargs={'device': 'cpu'},  # 使用GPU可改为 'cuda'
        encode_kwargs={
            'normalize_embeddings': True,  # 归一化向量，便于相似度计算
            'batch_size': 32  # 批处理大小
        }
    )
    # 获取纯文本列表
    texts = [split.page_content for split in splits]
    # 生成嵌入向量
    vectors = embeddings.embed_documents(texts)
    print(f"  已生成 {len(vectors)} 个向量")
    print(f"  每个向量的维度: {len(vectors[0])}")
    # 6. 创建向量数据库并存储嵌入
    print("正在生成向量嵌入...")
    vectorstore = FAISS.from_documents(
        documents=splits,
        embedding=embeddings,
    )
    print("向量数据库创建完成")
    retriever = vectorstore.as_retriever(
        search_type="similarity",  # 相似度检索
        search_kwargs={"k": 4}     # 返回最相关的4个片段
    )

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
    prompt = template.format(info=splits, question="人工智能的历史发展是怎样的？")
    response = llm.invoke(prompt)
    print(response.content)