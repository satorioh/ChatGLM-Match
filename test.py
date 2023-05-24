from pdf_parser import GrobidSciPDFPaser
from utils import get_abs_path
import pandas as pd
import json


# pdf_path = get_abs_path('content/2.pdf')
# print(f"pdf path: {pdf_path}")
# pdf = GrobidSciPDFPaser(
#     pdf_link=pdf_path
# )
# print(pdf.metadata)
# db_cache_path = get_abs_path('pdf_db/pdf_parser_grobid_scipdf.pkl')
# db_cache = pd.read_pickle(db_cache_path)
# with open('pdf.json', 'w') as file:
#     file.write(json.dumps(db_cache[('/Users/robin/Git/ChatGLM-Match/content/2.pdf', 'grobid_scipdf')]))
