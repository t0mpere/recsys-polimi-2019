from SLIM.SLIM_BPR_Cython import SLIM_BPR_Cython
from cf.item_cf import ItemCollaborativeFiltering
from challenge2019.GraphBased.RP3beta import RP3betaRecommender
from challenge2019.cbf.user_cbf import *
from challenge2019.utils.utils import Utils
from challenge2019.hybrid.hybrid_item_cf_RP3beta import HybridItemCfRP3Beta


class HybridOnRecs(object):

    def __init__(self, divide_recommendations=False):
        self.URM = None
        self.recommender_item_cf = ItemCollaborativeFiltering()
        self.recommender_rp3beta = HybridItemCfRP3Beta()

    # .4011
    # .24
    #
    def fit(self, URM, fit_once=False):
        self.recommender_item_cf.fit(URM)
        self.recommender_rp3beta.fit(URM)

    def recommend(self, user_id, at=10):

        recommended_items = []
        expected_items_rp3beta = self.recommender_rp3beta.recommend(user_id, at=40)
        expected_items_item_cf = self.recommender_item_cf.recommend(user_id, at=40)
        intersection = np.intersect1d(expected_items_item_cf, expected_items_rp3beta)
        if len(intersection) == 0:

            recommended_items = np.concatenate((expected_items_rp3beta[:5], expected_items_rp3beta[:5]))
        else:
            #print(len(intersection))
            #print(expected_items_rp3beta,intersection)
            expected_items_rp3beta = np.setdiff1d(expected_items_rp3beta, intersection)
            #print(expected_items_rp3beta)
            expected_items_item_cf = np.setdiff1d(expected_items_item_cf, intersection)
            recommended_items = np.concatenate(
                (intersection, expected_items_rp3beta[:5], expected_items_item_cf[:5]))
            # print(recommended_items)
        return recommended_items[:at]

    def get_expected_ratings(self, user_id, normalized_ratings=False):
        expected_ratings = self.RECS[user_id].todense()
        expected_ratings = np.squeeze(np.asarray(expected_ratings))


        # Normalize ratings
        if normalized_ratings and np.amax(expected_ratings) > 0:
            expected_ratings = expected_ratings / np.linalg.norm(expected_ratings)

        return expected_ratings

if __name__ == '__main__':
    recommender = HybridOnRecs()
    Runner.run(recommender, True, evaluate_different_type_of_users=True, find_weights_hybrid_item=False, batch_evaluation=True, split="2080")
