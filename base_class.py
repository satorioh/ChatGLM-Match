import os
import abc
import pandas as pd
import pickle
from utils import get_abs_path
from configs.global_config import (
    PDF_DB_DIR,
    PDF_DB_CACHE_PATH
)


class AbstractPDFParser(metaclass=abc.ABCMeta):
    """ PDF parser to parse a PDF file"""

    def __init__(self, db_name) -> None:
        """Initialize the pdf database"""
        db_path = get_abs_path(PDF_DB_DIR)
        if not os.path.exists(db_path):
            os.makedirs(db_path)
        db_cache_path = get_abs_path(PDF_DB_CACHE_PATH)
        self.db_cache_path = db_cache_path

        # load the cache if it exists, and save a copy to disk
        try:
            db_cache = pd.read_pickle(db_cache_path)
        except FileNotFoundError:
            db_cache = {}
        with open(db_cache_path, "wb") as cache_file:
            pickle.dump(db_cache, cache_file)
        self.db_cache = db_cache
        self.db_name = db_name

    @abc.abstractmethod
    def parse_pdf(self, ) -> None:
        """Parse the PDF file"""
        pass

    @abc.abstractmethod
    def _get_metadata(self, ) -> None:
        """Get the metadata of the PDF file"""
        pass
