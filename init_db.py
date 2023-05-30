from pdf_parser import GrobidSciPDFPaser
from utils import get_abs_path, get_file_list
from configs.global_config import (
    RAW_PDF_DIR
)

file_list = get_file_list(get_abs_path(RAW_PDF_DIR))
for index, pdf_path in enumerate(file_list):
    print(f"{index} -----> {pdf_path}")
    pdf = GrobidSciPDFPaser(
        pdf_link=pdf_path
    )
    print(pdf.metadata)
