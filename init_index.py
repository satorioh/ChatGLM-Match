import pandas as pd
import os
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import TensorflowHubEmbeddings
from langchain.vectorstores import FAISS
from utils import (
    get_abs_path,
    tf_limit_memory
)
from configs.global_config import (
    PDF_DB_CACHE_PATH,
    FAISS_INDEX_DIR,
    EMBEDDING_MODEL_DIR
)

docs = []
try:
    db_cache_path = get_abs_path(PDF_DB_CACHE_PATH)
    db_cache = pd.read_pickle(db_cache_path)
    for key, item in db_cache.items():
        doc = Document(
            page_content=item['sections'],
            metadata={"title": item["title"], "authors": item["authors"], "pub_date": item["pub_date"],
                      "abstract": item["abstract"], "source": item["title"] + ".pdf"}
        )
        docs.append(doc)
    print(f"Document 已成功加载")
except Exception as err:
    print(err)
    print(f"Document 未能成功加载")

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                               separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
                                               chunk_overlap=200, )
embeddings = TensorflowHubEmbeddings(model_url=get_abs_path(EMBEDDING_MODEL_DIR))
# 切割加载的 document
print("start split docs...")
split_docs = text_splitter.split_documents(docs)
print("split docs finished")


def embed_documents(split_docs):
    vector_store = FAISS.load_local(FAISS_INDEX_DIR, embeddings)
    for index, split_doc in enumerate(split_docs):
        print(f"start faiss embedding {index}")
        vector_store.add_documents([split_doc])
        vector_store.save_local(FAISS_INDEX_DIR)
        print(f"faiss embedding {index} saved")
        yield split_doc


embed_documents(split_docs)
