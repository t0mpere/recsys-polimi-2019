from challenge2019.MF.ALS import AlternatingLeastSquare
from challenge2019.cf.user_cf import *
from challenge2019.topPop.topPop import *
from challenge2019.featureWeighting.feature_weighting_on_item import *
from challenge2019.hybrid.hybrid_all_item_sim import *


class Hybrid(object):

    def __init__(self, divide_recommendations=False):
        self.URM = None
        self.SM_item = None
        self.recommenderUser = UserCollaborativeFiltering()
        self.recommenderHybridItem = HybridItemAllItemSim()
        self.recommenderTopPop = TopPop()
        self.recommender_ALS = AlternatingLeastSquare()
        self.divide_recommendations = divide_recommendations
        self.fitted = False
        self.weights = None

    def fit(self, URM, fit_once=False, weights=None):
        if weights is None:
            weights = {
                "MF": 0.005,
                "item": 0.7551,
                "user_cf": 0.006994
            }  # 0.05236
            weights_to_try_in_submit = {
                "MF": 0.005171,
                "item": 0.9976,
                "user_cf": 0.00531
            }  # 0.05236

        self.weights = weights

        if not (fit_once and self.fitted):
            self.URM = URM
            if self.divide_recommendations:
                True

            self.recommenderUser.fit(URM, knn=784, shrink=10)
            self.recommenderHybridItem.fit(URM)
            self.recommender_ALS.fit(URM)
            self.recommenderTopPop.fit(URM)
            self.fitted = True

    def recommend(self, user_id, at=10):

        normalized_ratings = False

        self.URM.eliminate_zeros()
        liked_items = self.URM[user_id]

        if len(liked_items.data) == 0:
            expected_ratings = self.recommenderTopPop.get_expected_ratings(user_id)
        else:
            expected_ratings = self.weights["user_cf"] * self.recommenderUser.get_expected_ratings(user_id,
                                                                                                   normalized_ratings=normalized_ratings) \
                               + self.weights["item"] * self.recommenderHybridItem.get_expected_ratings(user_id,
                                                                                                        normalized_ratings=normalized_ratings) \
                               + self.weights["MF"] * self.recommender_ALS.get_expected_ratings(user_id)

        recommended_items = np.flip(np.argsort(expected_ratings), 0)

        unseen_items_mask = np.in1d(recommended_items, self.URM[user_id].indices,
                                    assume_unique=True, invert=True)
        recommended_items = recommended_items[unseen_items_mask]
        return recommended_items[0:at]


if __name__ == '__main__':
    for i in range(0, 10):
        recommender = Hybrid(divide_recommendations=False)
        Runner.run(recommender, True, find_weights_hybrid_20=True, batch_evaluation=False)
