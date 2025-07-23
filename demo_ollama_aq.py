import os
from typing import List
import json
import argparse
import logging

from src.hipporag import HippoRAG

def main():

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

    queries = [
        "俊佐有哪些好朋友？",
        "谁住在深圳？",
    ]
    retrieval_results = hipporag.retrieve(queries=queries, num_to_retrieve=3)
    print(retrieval_results)

    # print(hipporag.rag_qa(queries=queries))

if __name__ == "__main__":
    main()
