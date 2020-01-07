from challenge2019.GraphBased.RP3beta import RP3betaRecommender
from challenge2019.cbf.user_cbf import *
from challenge2019.utils.utils import Utils


class HybridRP3Beta(object):

    def __init__(self, divide_recommendations=False):
        self.URM = None
        self.RP3_beta_URM_ICM = None
        self.RP3_beta_URM = None
        self.alpha = None
        self.fitted = False

    def fit(self, URM, alpha=0.65, fit_once=False):
        self.URM = URM
        self.alpha = alpha

        if not (fit_once and self.fitted):
            self.RP3_beta_URM_ICM = RP3betaRecommender()
            self.RP3_beta_URM = RP3betaRecommender()
            self.RP3_beta_URM_ICM.fit(self.URM, alpha=.5, beta=0.1, topK=60, use_ICM=True)
            self.RP3_beta_URM.fit(self.URM, alpha=.3, beta=0.1, topK=90, use_ICM=False)
            self.fitted = True

    def get_expected_ratings(self, user_id):
        user_id = int(user_id)

        liked_items = self.URM[user_id]

        expected_ratings = self.RP3_beta_URM_ICM.get_expected_ratings(user_id) * self.alpha + self.RP3_beta_URM.get_expected_ratings(user_id) * (1 - self.alpha)

        return expected_ratings

    def recommend(self, user_id, at=10):
        expected_ratings = self.get_expected_ratings(user_id)

        recommended_items = np.flip(np.argsort(expected_ratings), 0)

        unseen_items_mask = np.in1d(recommended_items, self.URM[user_id].indices,
                                    assume_unique=True, invert=True)
        recommended_items = recommended_items[unseen_items_mask]
        return recommended_items[0:at]

if __name__ == '__main__':
    recommender = HybridRP3Beta()
    Runner.run(recommender, True, evaluate_different_type_of_users=True, find_weights_hybrid_item=False,
               batch_evaluation=True, split='2080')
