from challenge2019.GraphBased.RP3beta import RP3betaRecommender
from challenge2019.cbf.user_cbf import *
from challenge2019.utils.utils import Utils
from challenge2019.SLIM.SlimElasticNet import *
from challenge2019.cbf.item_cbf import *

class HybridItemAllItemSim(object):

    def __init__(self, divide_recommendations=False):
        self.URM = None
        self.SM_item = None
        self.fitted = False
    # .4011
    # .24
    #
    def fit(self, URM, fit_once=False, weights=None):
        if weights is None:
            weights = {
                "RP3": 0.9,
                "SLIM_E": 0.7,
                "cbf": 0.7,
                "cf": 0.4
            }
        if not (fit_once and self.fitted):
            self.URM = URM
            RP3_beta = RP3betaRecommender()
            ElasticNet = SLIMElasticNetRecommender()
            item_cbf = ItemContentBasedFiltering()

            RP3_beta.fit(URM, normalize_similarity=True)
            ElasticNet.fit(URM)
            item_cbf.fit(URM)

            self.SM_cf = self.create_similarity_matrix(URM, 15, 19, similarity="tanimoto")
            self.SM_P3beta = RP3_beta.get_W()
            self.SM_EN = ElasticNet.W_sparse
            self.SM_cbf = item_cbf.SM

            self.fitted = True

        self.SM = weights["cf"] * self.SM_cf + weights["RP3"] * self.SM_P3beta + weights["SLIM_E"] * self.SM_EN + weights["cbf"] * self.SM_cbf

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
    for i in range(0, 10):
        recommender = HybridItemAllItemSim()
        Runner.run(recommender, True, find_weights_hybrid_all_item=True, batch_evaluation=False)
