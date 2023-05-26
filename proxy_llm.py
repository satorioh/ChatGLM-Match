import os
from langchain.llms.base import LLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from typing import Optional, List
from utils import get_abs_path
from configs.global_config import (
    FAISS_INDEX_DIR
)

index_folder = get_abs_path(FAISS_INDEX_DIR)


class ProxyLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "ChatGLM"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return prompt


def init_knowledge_vector_store(embeddings):
    if os.path.exists(index_folder):
        try:
            print("start load faiss index")
            vector_store = FAISS.load_local(FAISS_INDEX_DIR, embeddings)
            print("load faiss index finished")
            return vector_store
        except Exception as err:
            print(f"load faiss index error: {err}")
    else:
        raise ValueError("faiss index not exist")


def init_chain_proxy(llm_proxy: LLM, vector_store, top_k=5):
    prompt_template = """你是一份有用的 PDF 文件。你的任务是提供信息并回答任何相关问题。你应该使用 PDF 文件的各个部分作为信息来源，尽可能简练准确地回答用户提出的任何问题。如果你在给定的部分中找不到相关信息，你需要告知用户该来源不包含相关信息，但仍然尝试根据你的一般知识提供答案。你需要在回答时引用相应的章节名称和页码，答案请使用中文。以下是有关该 PDF 文件的相关信息，这些信息将帮助你回答用户的问题。

相关信息:
{context}

请根据上述信息回答如下问题:
{question}"""
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    knowledge_chain = RetrievalQA.from_llm(
        llm=llm_proxy,
        retriever=vector_store.as_retriever(
            search_kwargs={"k": top_k}),
        prompt=prompt,
        return_source_documents=True
    )
    knowledge_chain.combine_documents_chain.document_prompt = PromptTemplate(
        input_variables=["page_content"], template="{page_content}"
    )
    return knowledge_chain
