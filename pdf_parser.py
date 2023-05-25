from base_class import AbstractPDFParser
from scipdf_utils import parse_pdf_to_dict
from configs.global_config import PDF_DB_NAME
import pickle


class GrobidSciPDFPaser(AbstractPDFParser):
    def __init__(self, pdf_link, db_name=PDF_DB_NAME) -> None:
        """Initialize the PDF parser

            Args:
                pdf_link: link to the PDF file, the pdf link can be a web link or local file path
                metadata: metadata of the PDF file, like authors, title, abstract, etc.
                paragraphs: list of paragraphs of the PDF file, all paragraphs are concatenated together
                split_paragraphs: dict of section name and corresponding list of split paragraphs
        """
        super().__init__(db_name=db_name)
        self.db_name = db_name
        self.pdf_link = pdf_link
        self.pdf = None
        self.metadata = {}
        self.parse_pdf()

    def _retrive_or_parse(self, ):
        """Return pdf dict from cache if present, otherwise parse the pdf"""
        print("Return pdf dict from cache")
        db_name = self.db_name
        if (self.pdf_link, db_name) not in self.db_cache.keys():
            print("build cache")
            self.db_cache[(self.pdf_link, db_name)
            ] = parse_pdf_to_dict(self.pdf_link)
            with open(self.db_cache_path, "wb") as db_cache_file:
                pickle.dump(self.db_cache, db_cache_file)
        return self.db_cache[(self.pdf_link, db_name)]

    def parse_pdf(self) -> None:
        print("start parse pdf")
        """Parse the PDF file
        """
        article_dict = self._retrive_or_parse()
        self.article_dict = article_dict
        self._get_metadata()
        print("finish parse pdf")

    def _get_metadata(self) -> None:
        print("get metadata")
        for meta in ['authors', "pub_date", "abstract", "references", "doi", 'title', ]:
            self.metadata[meta] = self.article_dict[meta]
