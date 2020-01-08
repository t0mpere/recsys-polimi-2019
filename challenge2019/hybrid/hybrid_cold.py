from challenge2019.cbf.user_cbf import *
from challenge2019.topPop.topPop import *
from challenge2019.topPop.topPop_userClasses import *


class HybridCold(object):

    def __init__(self):
        self.alpha = None
        self.fitted = False

        self.UserContentBasedFilteringSenzaURM = UserContentBasedFiltering()
        self.UserContentBasedFilteringConURM = UserContentBasedFiltering()
        self.recommenderTopPop = TopPopUserClasses()

    def fit(self, URM, alpha=0.2, fit_once=False):
        self.alpha = alpha
        self.URM = URM
        if not (fit_once and self.fitted):

            self.UserContentBasedFilteringSenzaURM.fit(URM, use_URM=False)
            self.UserContentBasedFilteringConURM.fit(URM, use_URM=True)
            self.recommenderTopPop.fit(URM)

            self.fitted = True

    def recommend(self, user_id, at=10):
        liked_items = self.URM[user_id]
        expected_items_top_pop = self.recommenderTopPop.recommend(user_id, at=10)
        expected_items_user_cbf_senza_URM = self.UserContentBasedFilteringSenzaURM.recommend(user_id, at=10)
        expected_items_user_cbf_con_URM = self.UserContentBasedFilteringConURM.recommend(user_id, at=10)

        if np.flip(np.sort(self.UserContentBasedFilteringSenzaURM.get_expected_ratings(user_id)))[0] > 0:

            recommended_items = expected_items_user_cbf_con_URM

        else:
            recommended_items = expected_items_top_pop
        return recommended_items[0:at]


if __name__ == '__main__':
    recommender = HybridCold()
    Runner.run(recommender, True, evaluate_different_age_of_users=True, find_weights_hybrid_item=False,
               batch_evaluation=True, split='2080')

    # best score on seed 69: MAP@10 : 0.03042666580147029
    # 0.03298346361837503
    # 0.03291286461327143
