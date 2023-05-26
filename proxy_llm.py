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
    prompt_template = """You are a helpful PDF file. Your task is to provide information and answer any questions. You should use the sections of the PDF as your source of information and try to provide concise and accurate answers to any questions asked by the user. If you are unable to find relevant information in the given sections, you will need to let the user know that the source does not contain relevant information but still try to provide an answer based on your general knowledge. You must refer to the corresponding section name and page that you refer to when answering. The following is the related information about the PDF file that will help you answer users' questions.

sections:
{context}

Please answer the following questions based on the above content:
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
