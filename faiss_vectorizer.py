"""
faiss_vectorizer.py

This module provides functionality to create and use FAISS indexes for
Nginx access log data. It uses TF-IDF vectorization for different fields
and allows searching within the indexed vectors.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import faiss
from nginx_access_log_parser import NginxAccessLogParser


class FaissIndex:
    """
    A class to create and manage FAISS indexes for Nginx access log data.

    Attributes:
        log_file_path (str): The path to the Nginx access log file.
        df (pd.DataFrame): DataFrame containing the parsed log data.
        vectorizers (dict): Dictionary of TfidfVectorizer instances for each field.
        indices (dict): Dictionary of FAISS indexes for each field.
        vectors (dict): Dictionary of vector arrays for each field.
    """

    def __init__(self, log_file_path='nginx_access.log'):
        """
        Initializes the FaissIndex with the given log file path.

        Args:
            log_file_path (str): The path to the Nginx access log file.
        """
        self.df = NginxAccessLogParser(log_file_path).parse_log()
        self.vectorizers = {
            'url': TfidfVectorizer(),
            'timestamp': TfidfVectorizer(),
            'status': TfidfVectorizer(),
            'user_agent': TfidfVectorizer(),
            'ip_address': TfidfVectorizer(),
        }
        self.indices = {}
        self.vectors = {}

    def vectorize(self, text: str, field: str) -> np.ndarray:
        """
        Converts a text into a vector using the specified field's vectorizer.

        Args:
            text (str): The text to be vectorized.
            field (str): The field to use for vectorization.

        Returns:
            np.ndarray: The vectorized representation of the text.
        """
        return self.vectorizers[field].transform([text]).toarray()

    def build_index(self):
        """
        Builds FAISS indexes for each field using TF-IDF vectors.
        """
        for field, vectorizer in self.vectorizers.items():
            self.df[field] = self.df[field].astype(str)
            vectorizer.fit(self.df[field])
            self.vectors[field] = vectorizer.transform(self.df[field]).toarray().astype('float32')

            dimension = self.vectors[field].shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(self.vectors[field])
            self.indices[field] = index

    def search(self, query: str, field: str, k: int = 5) -> list:
        """
        Searches the FAISS index for the specified field using the query vector.

        Args:
            query (str): The query string to search.
            field (str): The field to search within.
            k (int): The number of nearest neighbors to return (default is 5).

        Returns:
            list: A list of indices of the nearest neighbors.
        """
        query_vector = self.vectorize(query, field).astype('float32')
        distances, indices = self.indices[field].search(query_vector, k)
        return indices.tolist()
