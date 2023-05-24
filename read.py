from helper import get_abs_path
import pandas as pd
from langchain.docstore.document import Document

db_cache_path = get_abs_path('pdf_db/pdf_parser_grobid_scipdf.pkl')
db_cache = pd.read_pickle(db_cache_path)
docs = []
for key, item in db_cache.items():
    doc = Document(
        page_content=item['sections'],
        metadata={"title": item["title"], "authors": item["authors"], "pub_date": item["pub_date"],
                  "abstract": item["abstract"], "source": item["title"] + ".pdf"}
    )
    print(doc)
    docs.append(doc)
