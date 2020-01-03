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
        self.SM_item = None
        self.hybrid_RP3_beta = None
        self.hybrid_itemcf = None
        self.fitted = False

        self.recommenderUserCBF = UserContentBasedFiltering()
        self.recommenderTopPop = TopPop()

    # .4011
    # .24
    #
    def fit(self, URM, fit_once=False, alpha=0.65):
        self.alpha = alpha
        if not (fit_once and self.fitted):
            self.URM = URM

            utils = Utils()

            self.ICM = utils.get_icm()
            print(self.URM.shape, self.ICM.transpose().shape)
            self.URM_ICM = sps.vstack([self.URM, self.ICM.transpose()]).tocsr()

            self.hybrid_RP3_beta = HybridRP3Beta()
            self.hybrid_RP3_beta.fit(self.URM)
            self.hybrid_itemcf = hybridItemCF()
            self.hybrid_itemcf.fit(self.URM)

            self.recommenderUserCBF.fit(URM)
            self.recommenderTopPop.fit(URM)

            self.fitted = True


    def recommend(self, user_id, at=10):

        liked_items = self.URM[user_id]

        if len(liked_items.data) == 0:
            recommended_items = []
            expected_items_top_pop = self.recommenderTopPop.recommend(user_id, at=20)
            expected_items_user_cbf = self.recommenderUserCBF.recommend(user_id, at=20)
            #    intersection = np.intersect1d(expected_items_user_cbf, expected_items_top_pop)
            #    if len(intersection) == 0:
            #        recommended_items = expected_items_top_pop[:10]
            #    else:
            #        expected_items_top_pop = np.setdiff1d(expected_items_top_pop, intersection)
            #        expected_items_user_cbf = np.setdiff1d(expected_items_user_cbf, intersection)
            #        recommended_items = np.concatenate(
            #            (expected_items_top_pop[:5], intersection, expected_items_user_cbf[:5]))
            if np.flip(np.sort(self.recommenderUserCBF.get_expected_ratings(user_id)))[0] > 0:
                recommended_items = list(set(expected_items_user_cbf).intersection(set(expected_items_top_pop)))

            i = 0
            while len(recommended_items) < 10:
                if expected_items_top_pop[i] not in recommended_items:
                    recommended_items.append(expected_items_top_pop[i])
                i += 1

        else:

            expected_ratings = self.get_expected_ratings(user_id)

            recommended_items = np.flip(np.argsort(expected_ratings), 0)

            unseen_items_mask = np.in1d(recommended_items, self.URM[user_id].indices,
                                        assume_unique=True, invert=True)
            recommended_items = recommended_items[unseen_items_mask]
        return recommended_items[0:at]

    def get_expected_ratings(self, user_id, normalized_ratings=False):
        expected_ratings = self.alpha * self.hybrid_RP3_beta.get_expected_ratings(user_id) + (1 - self.alpha) * self.hybrid_itemcf.get_expected_ratings(user_id)
        expected_ratings = np.squeeze(np.asarray(expected_ratings))

        # Normalize ratings
        if normalized_ratings and np.amax(expected_ratings) > 0:
            expected_ratings = expected_ratings / np.linalg.norm(expected_ratings)

        return expected_ratings


if __name__ == '__main__':
    recommender = HybridItemCfRP3Beta()
    Runner.run(recommender, False, evaluate_different_type_of_users=True, find_weights_hybrid_item=False,
               batch_evaluation=True, split='2080')

# seed 123
