import numpy as np
import scipy.sparse as sps

from challenge2019.Base.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
from challenge2019.utils.run import Runner
from challenge2019.utils.utils import Utils


class UserContentBasedFiltering():
    def __init__(self):
        self.knn = None
        self.shrink = None
        self.knn_age = None
        self.knn_region = None
        self.similarity = None
        self.URM = None
        self.UCM_region = None
        self.UCM_age = None
        self.SM_region = None
        self.SM_age = None

    def create_similarity_matrix(self, UCM, knn=100):
        similarity_object = Compute_Similarity_Python(UCM.transpose(), topK=knn, shrink=self.shrink,
                                                      normalize=True, similarity=self.similarity)
        return similarity_object.compute_similarity()

    def fit(self, URM, knn_age=1000, knn_region=1000, shrink=20, similarity="dice"):
        utils = Utils()
        self.knn_age = knn_age
        self.knn_region = knn_region
        self.shrink = shrink
        self.similarity = similarity
        self.URM = URM
        self.UCM_region = utils.get_ucm_region_from_csv()
        self.UCM_age = utils.get_ucm_age_from_csv()

        self.combined_UCM = sps.hstack([self.UCM_age, self.UCM_region])
        # TODO: improve UCM (lezione 30/09)
        print("Starting calculating similarity USER_CBF")

        # self.SM_age = self.create_similarity_matrix(self.UCM_age, self.knn_age)
        # self.SM_region = self.create_similarity_matrix(self.UCM_region, self.knn_region)
        self.SM = self.create_similarity_matrix(self.combined_UCM, 1000)

        # self.RECS_region = self.SM_region.dot(self.URM)
        # self.RECS_age = self.SM_age.dot(self.URM)
        self.RECS = self.SM.dot(self.URM)


    def get_expected_ratings(self, user_id, i=0.5, normalized_ratings=True):
        user_id = int(user_id)
        # region_exp_ratings = self.RECS_region[user_id].todense()
        # age_exp_ratings = self.RECS_age[user_id].todense()
        expected_ratings = self.RECS[user_id].todense()

        # if np.amax(region_exp_ratings) > 0:
        #     region_exp_ratings = region_exp_ratings / np.linalg.norm(region_exp_ratings)
        # if np.amax(age_exp_ratings) > 0:
        #     age_exp_ratings = age_exp_ratings / np.linalg.norm(age_exp_ratings)

        # expected_ratings = (region_exp_ratings * i) \
        #                    + (age_exp_ratings * (1-i))

        expected_ratings = np.squeeze(np.asarray(expected_ratings))

        # Normalize ratings
        if normalized_ratings and np.amax(expected_ratings) > 0:
            expected_ratings = expected_ratings / np.linalg.norm(expected_ratings)

        return expected_ratings

    def recommend(self, user_id, i=0.5, at=10):
        user_id = int(user_id)
        expected_ratings = self.get_expected_ratings(user_id, i)
        recommended_items = np.flip(np.argsort(expected_ratings), 0)

        unseen_items_mask = np.in1d(recommended_items, self.URM[user_id].indices,
                                    assume_unique=True, invert=True)
        recommended_items = recommended_items[unseen_items_mask]
        return recommended_items[0:at]

if __name__ == '__main__':
    recommender = UserContentBasedFiltering()
    Runner.run(recommender, True, find_hyper_parameters_user_cbf=False, evaluate_different_region_of_users=True, batch_evaluation=True)
