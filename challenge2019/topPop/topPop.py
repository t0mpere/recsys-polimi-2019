import numpy as np
import scipy.sparse as sps

from challenge2019.Base.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
from challenge2019.utils.run import Runner
from challenge2019.utils.utils import Utils


class TopPop():
    def __init__(self):
        self.URM = None

    def fit(self, URM):
        self.URM = URM
        self.occurrencies = np.array(np.zeros(self.URM.shape[1]))
        for i in range(0, self.URM.shape[1]):
            self.occurrencies[i] = len(self.URM[:, i].data)

    def get_expected_ratings(self, user_id):
        data = np.arange(0.1, 1.1, 0.1)
        data = np.flip(data, 0)

        recommended_items = list(self.recommend(user_id))
        expected_ratings = np.zeros(self.URM.shape[1])
        for item in recommended_items:
            expected_ratings[item] = data[recommended_items.index(item)]
        return expected_ratings


    def recommend(self, user_id, at=10):
        expected_ratings = self.occurrencies
        recommended_items = np.flip(np.argsort(expected_ratings), 0)

        unseen_items_mask = np.in1d(recommended_items, self.URM[user_id].indices,
                                    assume_unique=True, invert=True)
        recommended_items = recommended_items[unseen_items_mask]
        return recommended_items[0:at]


if __name__ == '__main__':
    recommender = TopPop()
    Runner.run(recommender, True, evaluate_different_type_of_users=True,batch_evaluation=True)
