from pdf_parser import GrobidSciPDFPaser
from helper import get_abs_path, get_file_list

file_list = get_file_list(get_abs_path('content'))
# print(file_list)
#
for pdf_path in file_list:
    pdf = GrobidSciPDFPaser(
        pdf_link=pdf_path
    )
    print(pdf.metadata)
