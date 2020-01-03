from challenge2019.GraphBased.RP3beta import RP3betaRecommender
from challenge2019.cbf.user_cbf import *
from challenge2019.utils.utils import Utils


class HybridItemCfRP3Beta(object):

    def __init__(self, divide_recommendations=False):
        self.URM = None
        self.RP3_beta_short = None
        self.RP3_beta_long = None

    def fit(self, URM):
        self.URM = URM

        self.RP3_beta_short = RP3betaRecommender()
        self.RP3_beta_long = RP3betaRecommender()
        self.RP3_beta_short.fit(self.URM, alpha=.528, beta=0.1592, topK=72, use_ICM=True)
        self.RP3_beta_long.fit(self.URM, alpha=.02069, beta=0.03782, topK=77, use_ICM=False)

    def get_expected_ratings(self, user_id):
        user_id = int(user_id)

        liked_items = self.URM[user_id]

        if len(liked_items.data) <= 32:
            expected_ratings = self.RP3_beta_short.get_expected_ratings(user_id)
        else:
            expected_ratings = self.RP3_beta_long.get_expected_ratings(user_id)

        return expected_ratings

    def recommend(self, user_id, at=10):
        expected_ratings = self.get_expected_ratings(user_id)

        recommended_items = np.flip(np.argsort(expected_ratings), 0)

        unseen_items_mask = np.in1d(recommended_items, self.URM[user_id].indices,
                                    assume_unique=True, invert=True)
        recommended_items = recommended_items[unseen_items_mask]
        return recommended_items[0:at]

if __name__ == '__main__':
    recommender = HybridItemCfRP3Beta()
    Runner.run(recommender, True, evaluate_different_type_of_users=True, find_weights_hybrid_item=False,
               batch_evaluation=True, split='2080')
