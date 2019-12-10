from challenge2019.SLIM.SlimElasticNet import SLIMElasticNetRecommender
from challenge2019.cf.user_cf import *
from challenge2019.cf.item_cf import *
from challenge2019.cbf.item_cbf import *
from challenge2019.SLIM.SLIM_BPR_Cython import *
from challenge2019.topPop.topPop import *
from challenge2019.cbf.user_cbf import *
from challenge2019.utils.utils import Utils

class HybridUserCfCbf(object):

    def __init__(self, divide_recommendations=False):
        self.URM = None
        self.SM_item = None
        self.fitted = False

    def fit(self, URM, fit_once=False, alpha=0.8):
        self.alpha = alpha
        if not (fit_once and self.fitted):
            self.URM = URM
            utils = Utils()
            self.UCM_region = utils.get_ucm_region_from_csv()
            self.UCM_age = utils.get_ucm_age_from_csv()
            self.SM_age = self.create_similarity_matrix(self.UCM_age.transpose(), 1000, 20, similarity="dice")
            self.SM_region = self.create_similarity_matrix(self.UCM_region.transpose(), 1000, 20, similarity="dice")


            self.SM_cbf = self.SM_age + self.SM_region
            self.SM_cf = self.create_similarity_matrix(self.URM.transpose(), 784, 10, similarity="tversky")
            self.fitted = True

        self.SM = self.alpha * self.SM_cf + (1-self.alpha) * self.SM_cbf

        self.RECS = self.SM.dot(self.URM)


    def create_similarity_matrix(self, URM, knn, shrink, similarity="cosine"):
        similarity_object = Compute_Similarity_Python(URM, topK=knn, shrink=shrink, normalize=False,
                                                      similarity=similarity)
        return similarity_object.compute_similarity()

    def recommend(self, user_id, at=10):
        expected_ratings = self.get_expected_ratings(user_id)

        recommended_items = np.flip(np.argsort(expected_ratings), 0)

        unseen_items_mask = np.in1d(recommended_items, self.URM[user_id].indices,
                                    assume_unique=True, invert=True)
        recommended_items = recommended_items[unseen_items_mask]
        return recommended_items[0:at]

    def get_expected_ratings(self, user_id, normalized_ratings=False):
        expected_ratings = self.RECS[user_id].todense()
        expected_ratings = np.squeeze(np.asarray(expected_ratings))

        # Normalize ratings
        if normalized_ratings and np.amax(expected_ratings) > 0:
            expected_ratings = expected_ratings / np.linalg.norm(expected_ratings)

        return expected_ratings

if __name__ == '__main__':
    recommender = HybridUserCfCbf()
    Runner.run(recommender, True, find_weights_hybrid_item=True, evaluate_different_type_of_users=False, batch_evaluation=True)
