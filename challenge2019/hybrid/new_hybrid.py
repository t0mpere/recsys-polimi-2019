from challenge2019.MF.ALS import AlternatingLeastSquare
from challenge2019.SLIM.SlimElasticNet import *
from challenge2019.cbf.item_cbf import *
from challenge2019.cf.user_cf import *
from challenge2019.featureWeighting.feature_weighting_on_item import *
from challenge2019.hybrid.hybrid_RP3beta import RP3betaRecommender
from challenge2019.hybrid.hybrid_itemcf import hybridItemCF
from challenge2019.hybrid.hybrid_cold import HybridCold

"""
Best algorithm found
last sub: 0.03613
"""
class Hybrid(object):

    def __init__(self, divide_recommendations=False, only_cold=False):
        self.URM = None
        self.SM_item = None
        self.only_cold = only_cold
        self.recommenderUser = UserCollaborativeFiltering()
        self.recommenderItemCF = hybridItemCF()
        self.recommenderRP3Beta = RP3betaRecommender()
        self.recommenderItemCBF = ItemContentBasedFiltering()
        self.recommender_SLIM_E = SLIMElasticNetRecommender()
        self.recommender_ALS = AlternatingLeastSquare()
        self.hybrid_cold = HybridCold()
        self.divide_recommendations = divide_recommendations
        self.fitted = False
        self.weights = None

    def fit(self, URM, fit_once=False, weights=None):
        if weights is None:
            weights = {
                "MF": 0.08906,
                "RP3beta": 0.9,
                "SLIM_E": 0.3903,
                "item_cbf": 0.9087,
                "item_cf": 0.6,
                "user_cf": 0.002807
            }

        self.weights = weights

        if not (fit_once and self.fitted):
            self.URM = URM
            if self.divide_recommendations:
                True

            if not self.only_cold:
                self.recommenderUser.fit(URM, knn=784, shrink=10)
                self.recommenderItemCF.fit(URM)
                self.recommenderRP3Beta.fit(URM)
                self.recommender_SLIM_E.fit(URM)
                self.recommender_ALS.fit(URM)
                self.recommenderItemCBF.fit(URM)

            self.hybrid_cold.fit(URM)
            self.fitted = True

    def recommend(self, user_id, at=10):

        normalized_ratings = False

        self.URM.eliminate_zeros()
        liked_items = self.URM[user_id]

        if len(liked_items.data) == 0 or self.only_cold:
            recommended_items = self.hybrid_cold.recommend(user_id)

        else:
            er_item_cf = self.recommenderItemCF.get_expected_ratings(user_id, normalized_ratings=normalized_ratings)
            er_user_cf = self.recommenderUser.get_expected_ratings(user_id, normalized_ratings=normalized_ratings)
            er_SLIM_E = self.recommender_SLIM_E.get_expected_ratings(user_id, normalized_ratings=normalized_ratings)
            er_MF = self.recommender_ALS.get_expected_ratings(user_id)
            er_item_cbf = self.recommenderItemCBF.get_expected_ratings(user_id)
            er_RP3beta = self.recommenderRP3Beta.get_expected_ratings(user_id)

            # print("liked items: {} Sums: item_cf {}, user_cf {}, SLIM {}, MF {}, item_cbf {}".format(
            #    len(liked_items.data),
            #    er_item_cf.sum(), er_user_cf.sum(), er_SLIM_E.sum(), er_MF.sum(), er_item_cbf.sum()))

            expected_ratings = self.weights["item_cf"] * er_item_cf
            expected_ratings += self.weights["user_cf"] * er_user_cf
            expected_ratings += self.weights["SLIM_E"] * er_SLIM_E
            expected_ratings += self.weights["MF"] * er_MF
            expected_ratings += self.weights["item_cbf"] * er_item_cbf
            expected_ratings += self.weights["RP3beta"] * er_RP3beta



            if np.sum(expected_ratings) == 0:
                print("ah vettore vuoto")

            recommended_items = np.flip(np.argsort(expected_ratings), 0)

            unseen_items_mask = np.in1d(recommended_items, self.URM[user_id].indices,
                                        assume_unique=True, invert=True)
            recommended_items = recommended_items[unseen_items_mask]
        return recommended_items[0:at]


if __name__ == '__main__':
    recommender = Hybrid(divide_recommendations=False, only_cold=False)
    Runner.run(recommender, True, find_weights_new_hybrid=False, evaluate_different_type_of_users=False,
               batch_evaluation=True, split='2080')

    # best score on seed 69: MAP@10 : 0.03042666580147029
    # 0.03298346361837503
    # 0.03291286461327143
