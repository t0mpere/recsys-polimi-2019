import numpy as np
import scipy.sparse as sps

from challenge2019.Base.Similarity.Compute_Similarity_Cython import Compute_Similarity_Cython
from challenge2019.utils.run import Runner
from challenge2019.utils.utils import Utils


class hybridItemCF(object):

    def __init__(self):
        self.knn = None
        self.shrink = None
        self.similarity = None
        self.URM = None
        self.SM_item = None

    def create_similarity_matrices(self, params):
        res = list()
        for weight in params:
            similarity_object = Compute_Similarity_Cython(weight["URM"], topK=int(weight["knn"]),
                                                          shrink=int(weight["shrink"]), normalize=True,
                                                          similarity=weight["similarity"])

            res.append(similarity_object.compute_similarity())
        return res

    def fit(self, URM):
        print("Starting calculating similarity ITEM_CF")

        self.URM = URM
        utils = Utils()
        self.ICM = utils.get_icm()
        print(self.URM.shape, self.ICM.transpose().shape)
        self.URM_ICM = sps.vstack([self.URM, self.ICM.transpose()]).tocsr()

        # self.URM = utils.split_long_users(URM)
        # self.URM = utils.get_URM_BM_25(self.URM, K1=3, B=0.9) #<--- worst
        # self.URM = utils.get_URM_tfidf(self.URM) #<--- worst
        weights1 = {
            "URM": self.URM_ICM,
            "similarity": "tanimoto",
            "knn": 5,
            "shrink": 45
        }
        weights2 = {
            "URM": self.URM,
            "similarity": "tanimoto",
            "knn": 15,
            "shrink": 10
        }
        weights3 = {
            "URM": self.URM,
            "similarity": "tanimoto",
            "knn": 19,
            "shrink": 20
        }
        weights = [weights1, weights2, weights3]
        weights_sum = [0.9, 0.2, 0.2]

        self.results = self.create_similarity_matrices(weights)
        i = 0
        self.SM_item = None
        for res in self.results:
            if self.SM_item is None:
                self.SM_item = weights_sum[i] * res
            else:
                self.SM_item += weights_sum[i] * res
            i += 1
        self.RECS = self.URM.dot(self.SM_item)

    def get_similarity(self):
        return self.SM_item

    def get_expected_ratings(self, user_id, normalized_ratings=False):
        expected_ratings = self.RECS[user_id].todense()
        expected_ratings = np.squeeze(np.asarray(expected_ratings))

        # Normalize ratings
        if normalized_ratings and np.amax(expected_ratings) > 0:
            expected_ratings = expected_ratings / np.linalg.norm(expected_ratings)

        return expected_ratings

    def recommend(self, user_id, at=10):
        user_id = int(user_id)
        expected_ratings = self.get_expected_ratings(user_id)

        recommended_items = np.flip(np.argsort(expected_ratings), 0)

        unseen_items_mask = np.in1d(recommended_items, self.URM[user_id].indices,
                                    assume_unique=True, invert=True)
        recommended_items = recommended_items[unseen_items_mask]
        return recommended_items[0:at]


if __name__ == '__main__':
    recommender = hybridItemCF()
    Runner.run(recommender, True, find_hyper_parameters_cf=False, evaluate_different_type_of_users=True,
               batch_evaluation=True, split='2080')
