import numpy as np
import scipy.sparse as sps

from challenge2019.Base.Similarity.Compute_Similarity_Cython import \
    Compute_Similarity_Cython as Compute_Similarity_Python
from challenge2019.utils.run import Runner
from challenge2019.utils.utils import Utils


class ItemCollaborativeFiltering():
    def __init__(self):
        self.knn = None
        self.shrink = None
        self.similarity = None
        self.URM = None
        self.SM_item = None

    def create_similarity_matrix(self):
        similarity_object = Compute_Similarity_Python(self.URM, topK=self.knn, shrink=self.shrink, normalize=True,
                                                      similarity=self.similarity)
        return similarity_object.compute_similarity()

    def fit(self, URM, knn=5, shrink=45, similarity="tanimoto"):
        self.knn = knn
        self.shrink = shrink
        self.similarity = similarity
        print("Starting calculating similarity ITEM_CF")

        self.URM = URM
        utils = Utils()
        self.ICM = utils.get_icm()
        self.URM = sps.vstack([self.URM, self.ICM.transpose()]).tocsr()

        # self.URM = utils.split_long_users(URM)
        # self.URM = utils.get_URM_BM_25(self.URM, K1=3, B=0.9) #<--- worst
        # self.URM = utils.get_URM_tfidf(self.URM) #<--- worst

        self.SM_item = self.create_similarity_matrix()
        self.RECS = self.URM.dot(self.SM_item)

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
        if len(recommended_items) == 0: print("aia")
        return recommended_items[0:at]


if __name__ == '__main__':
    recommender = ItemCollaborativeFiltering()
    Runner.run(recommender, True, find_hyper_parameters_cf=False, evaluate_different_type_of_users=False,
               batch_evaluation=True, split='2080')

# 0.02888 with seed 69
# 0.19801094350783785 long seed 123
