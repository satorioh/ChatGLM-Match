import pandas as pd
from pdf_parser import GrobidSciPDFPaser
from utils import get_abs_path, get_file_list
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import TensorflowHubEmbeddings
from langchain.vectorstores import FAISS
from configs.global_config import (
    RAW_PDF_DIR,
    PDF_DB_CACHE_PATH,
    FAISS_INDEX_DIR,
    EMBEDDING_MODEL_DIR
)

file_list = get_file_list(get_abs_path(RAW_PDF_DIR))
for pdf_path in file_list:
    pdf = GrobidSciPDFPaser(
        pdf_link=pdf_path
    )
    print(pdf.metadata)

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

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                               separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
                                               chunk_overlap=200, )
embeddings = TensorflowHubEmbeddings(model_url=get_abs_path(EMBEDDING_MODEL_DIR))
# 切割加载的 document
print("start split docs...")
split_docs = text_splitter.split_documents(docs)
print("split docs finished")
vector_store = FAISS.from_documents(split_docs, embeddings)
vector_store.save_local(FAISS_INDEX_DIR)
