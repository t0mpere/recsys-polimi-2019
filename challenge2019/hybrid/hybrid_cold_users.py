from challenge2019.GraphBased.RP3beta import RP3betaRecommender
from challenge2019.cbf.user_cbf import *
from challenge2019.utils.utils import Utils
from challenge2019.hybrid.hybrid_itemcf import hybridItemCF
from challenge2019.hybrid.hybrid_RP3beta import HybridRP3Beta
from challenge2019.cbf.user_cbf import *
from challenge2019.topPop.topPop import *


class HybridItemCfRP3Beta(object):

    def __init__(self, divide_recommendations=False):
        self.URM = None
        self.fitted = False
        self.at = None
        self.threshold = None

        self.recommenderUserCBF = UserContentBasedFiltering()
        self.recommenderTopPop = TopPop()

    def fit(self, URM, fit_once=False, at=20, threshold=0):
        self.at =  int(at)
        self.threshold = threshold
        if not (fit_once and self.fitted):
            self.URM = URM

            self.recommenderUserCBF.fit(URM)
            self.recommenderTopPop.fit(URM)

            self.fitted = True


    def recommend(self, user_id):

        liked_items = self.URM[user_id]
        recommended_items = []
        if len(liked_items.data) == 0:
            expected_items_top_pop = self.recommenderTopPop.recommend(user_id, at=self.at)
            expected_items_user_cbf = self.recommenderUserCBF.recommend(user_id, at=self.at)

            if np.flip(np.sort(self.recommenderUserCBF.get_expected_ratings(user_id)))[0] > self.threshold:
                recommended_items = list(set(expected_items_user_cbf).intersection(set(expected_items_top_pop)))

                i = 0
                while len(recommended_items) < 10:
                    if expected_items_user_cbf[i] not in recommended_items:
                        recommended_items.append(expected_items_user_cbf[i])
                    i += 1

            else:
                i = 0
                while len(recommended_items) < 10:
                    if expected_items_top_pop[i] not in recommended_items:
                        recommended_items.append(expected_items_top_pop[i])
                    i += 1
        else:
            recommended_items = self.recommenderTopPop.recommend(user_id, at=10)
        return recommended_items[0:10]

if __name__ == '__main__':
    recommender = HybridItemCfRP3Beta()
    Runner.run(recommender, True, evaluate_different_type_of_users=True, find_weights_hybrid_cold_users=False,
               batch_evaluation=True, split='2080')

# seed 123
