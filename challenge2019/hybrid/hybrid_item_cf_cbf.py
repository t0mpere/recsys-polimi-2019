from challenge2019.SLIM.SlimElasticNet import SLIMElasticNetRecommender
from challenge2019.cf.user_cf import *
from challenge2019.cf.item_cf import *
from challenge2019.cbf.item_cbf import *
from challenge2019.SLIM.SLIM_BPR_Cython import *
from challenge2019.topPop.topPop import *
from challenge2019.cbf.user_cbf import *
from challenge2019.utils.utils import Utils

class HybridItemCfCbf(object):

    def __init__(self, divide_recommendations=False):
        self.URM = None
        self.SM_item = None

    def fit(self, URM):
        self.URM = URM
        utils = Utils()
        self.alpha = 0.8
        self.ICM_asset = utils.get_icm_asset_from_csv()
        self.ICM_price = utils.get_icm_price_from_csv()
        self.ICM_sub_class = utils.get_icm_sub_class_from_csv()
        self.combined_ICM = sps.hstack([self.ICM_asset, self.ICM_sub_class, self.ICM_price])

        self.SM_cbf = self.create_similarity_matrix(self.combined_ICM.transpose(), 900, 2)
        self.SM_cf = self.create_similarity_matrix(URM, 12, 23, similarity="tanimoto")

        self.SM = self.alpha * self.SM_cf + (1-self.alpha) * self.SM_cbf

        self.RECS = self.URM.dot(self.SM)


    def create_similarity_matrix(self, URM, knn, shrink, similarity="cosine"):
        similarity_object = Compute_Similarity_Python(URM, topK=knn, shrink=shrink, normalize=True,
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
    recommender = HybridItemCfCbf()
    Runner.run(recommender, True, evaluate_different_type_of_users=True, batch_evaluation=True)
