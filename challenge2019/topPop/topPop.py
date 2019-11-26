import numpy as np

from challenge2019.Base.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
from challenge2019.utils.run import Runner
from challenge2019.utils.utils import Utils

class TopPop():
    def __init__(self):
        self.URM = None

    def fit(self, URM, knn=100, shrink=5, similarity="cosine"):
        self.URM = URM
        self.occurrencies = np.array(np.zeros(self.URM.shape[1]))
        for i in range(0, self.URM.shape[1]):
            self.occurrencies[i] = len(self.URM[:, i].data)

    def recommend(self, user_id, at=10):
        expected_ratings = self.occurrencies
        recommended_items = np.flip(np.argsort(expected_ratings), 0)

        unseen_items_mask = np.in1d(recommended_items, self.URM[user_id].indices,
                                    assume_unique=True, invert=True)
        recommended_items = recommended_items[unseen_items_mask]
        return recommended_items[0:at]

#recommender = TopPop()
#Runner.run(recommender, True)