import os
from typing import List
import json
import argparse
import logging

from src.hipporag import HippoRAG

def main():

    # Prepare datasets and evaluation
    docs = [
        "俊佐是一个人工智能研究员，他住在北京。",
        "李明是俊佐的好朋友，他是一名软件工程师，住在深圳。",
        "王小红也是俊佐的朋友，她是数据科学家，在上海工作。",
        "ChromaDB是一个专为AI应用设计的向量数据库。",
        "HippoRAG结合了知识图谱和密集检索技术，提供更好的性能。"
    ]

    save_dir = 'outputs'  # Define save directory for HippoRAG objects (each LLM/Embedding model combination will create a new subdirectory)
    llm_model_name = 'deepseek-v3'  # Any OpenAI model name
    llm_base_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
    embedding_model_name = 'bge-m3'  # Embedding model name (NV-Embed, GritLM or Contriever for now)
    embedding_base_url='http://localhost:11434/v1'

    # Startup a HippoRAG instance
    hipporag = HippoRAG(save_dir=save_dir,
                        llm_model_name=llm_model_name,
                        llm_base_url=llm_base_url,
                        embedding_model_name=embedding_model_name,
                        embedding_base_url=embedding_base_url)

    # Run indexing
    hipporag.index(docs=docs)

    # Separate Retrieval & QA
    queries = [
        "俊佐有哪些好朋友？",
        "谁住在深圳？",
    ]

    print(hipporag.retrieve(queries=queries))

    # For Evaluation
    # answers = [
    #     ["Politician"],
    #     ["By going to the ball."],
    #     ["Rockland County"]
    # ]

    # gold_docs = [
    #     ["George Rankin is a politician."],
    #     ["Cinderella attended the royal ball.",
    #      "The prince used the lost glass slipper to search the kingdom.",
    #      "When the slipper fit perfectly, Cinderella was reunited with the prince."],
    #     ["Erik Hort's birthplace is Montebello.",
    #      "Montebello is a part of Rockland County."]
    # ]

    # print(hipporag.rag_qa(queries=queries,
    #                               gold_docs=gold_docs,
    #                               gold_answers=answers))

if __name__ == "__main__":
    main()
