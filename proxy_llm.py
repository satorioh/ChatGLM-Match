import os
import pandas as pd
from langchain.llms.base import LLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from typing import Optional, List
from utils import get_abs_path
from configs.global_config import (
    PDF_DB_CACHE_PATH,
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


def init_knowledge_vector_store(path: str, embeddings):
    if os.path.exists(index_folder):
        try:
            print("start load faiss index")
            vector_store = FAISS.load_local(FAISS_INDEX_DIR, embeddings)
            print("load faiss index finished")
            return vector_store
        except Exception as err:
            print(f"load faiss index error: {err}")
    else:
        fold_path = path
        docs = []
        if not os.path.exists(fold_path):
            print(f"{fold_path} 路径不存在")
            return None
        elif os.path.isdir(fold_path):
            try:
                db_cache_path = get_abs_path(PDF_DB_CACHE_PATH)
                db_cache = pd.read_pickle(db_cache_path)
                for key, item in db_cache.items():
                    doc = Document(
                        page_content=item['sections'],
                        metadata={"title": item["title"], "authors": item["authors"], "pub_date": item["pub_date"],
                                  "abstract": item["abstract"], "source": item["title"] + ".pdf"}
                    )
                    print(doc)
                    docs.append(doc)
                print(f"{fold_path} 已成功加载")
            except Exception as err:
                print(err)
                print(f"{fold_path} 未能成功加载")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800,
                                                       separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
                                                       chunk_overlap=0, )
        # 切割加载的 document
        print("start split docs...")
        split_docs = text_splitter.split_documents(docs)
        print("split docs finished")
        vector_store = FAISS.from_documents(split_docs, embeddings)
        vector_store.save_local(FAISS_INDEX_DIR)
        return vector_store


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
