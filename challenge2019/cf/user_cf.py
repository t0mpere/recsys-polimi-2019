import numpy as np
import scipy.sparse as sps

from challenge2019.Base.Similarity.Compute_Similarity_Cython import Compute_Similarity_Cython as Compute_Similarity_Python
from challenge2019.utils.run import Runner
from challenge2019.utils.utils import Utils


class UserCollaborativeFiltering():
    def __init__(self):
        self.knn = None
        self.shrink = None
        self.similarity = None
        self.URM = None
        self.SM_item = None

    def create_similarity_matrix(self, URM):
        similarity_object = Compute_Similarity_Python(URM.transpose(), topK=self.knn, shrink=self.shrink,
                                                      normalize=True, similarity=self.similarity)
        return similarity_object.compute_similarity()

    def fit(self, URM, knn=784, shrink=10, similarity="tversky"):
        self.knn = knn
        self.shrink = shrink
        self.similarity = similarity
        print("Starting calculating similarity USER_CF")

        self.URM = URM
        utils = Utils()
        self.URM = utils.get_URM_BM_25(self.URM)    #good
        # self.URM = utils.get_URM_tfidf(self.URM) <--- worst

        self.SM_user = self.create_similarity_matrix(self.URM)
        self.RECS = self.SM_user.dot(self.URM)

    def get_expected_ratings(self, user_id, normalized_ratings=False):
        user_id = int(user_id)
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
    recommender = UserCollaborativeFiltering()
    Runner.run(recommender, True, find_hyper_parameters_cf=False, evaluate_different_type_of_users=False, batch_evaluation=True, split='2080')

    # 0.02308 with seed 69