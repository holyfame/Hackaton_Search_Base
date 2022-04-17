import pickle, os, gdown
import numpy as np

from typing import List, Tuple
from .search import Base

import scann

class SearchSolution(Base):

    def __init__(self, data_file='./data/train_data.pickle',
                 data_url='https://drive.google.com/uc?id=1D_jPx7uIaCJiPb3pkxcrkbeFcEogdg2R') -> None:
        self.data_file = data_file
        self.data_url = data_url
        pass

    def set_base_from_pickle(self):
        if not os.path.isfile(self.data_file):
            if not os.path.isdir('./data'):
                os.mkdir('./data')
            gdown.download(self.data_url, self.data_file, quiet=False)

        with open(self.data_file, 'rb') as f:
            data = pickle.load(f)

        self.reg_matrix = [None] * len(data['reg'])
        self.ids = {}
        for i, key in enumerate(data['reg']):
            self.reg_matrix[i] = data['reg'][key][0][None]
            self.ids[i] = key

        self.reg_matrix = np.concatenate(self.reg_matrix, axis=0)
        self.pass_dict = data['pass']

        self.set_up_scann()

    # create indexer from reg_matrix
    def set_up_scann(self):
        self.searcher = scann.scann_ops_pybind.builder(self.reg_matrix, 10, "dot_product").tree(
            num_leaves=2000,
            num_leaves_to_search=100,
            training_sample_size=250000
        ).score_ah(
            2,
            anisotropic_quantization_threshold=0.2
        ).reorder(100).build()

    def search(self, query: np.array) -> List[Tuple]:
        neighbors, distances = self.searcher.search_batched(np.array([query]), final_num_neighbors=10)
        return [(self.ids[i], sim) for i, sim in zip(neighbors[0], distances)]

    def cos_sim(self, query: np.array) -> np.array:
        return np.dot(self.reg_matrix, query)

    def insert(self, feature: np.array) -> None:
        self.reg_matrix = np.concatenate(self.reg_matrix, feature, axis=0)
        self.set_up_scann()
