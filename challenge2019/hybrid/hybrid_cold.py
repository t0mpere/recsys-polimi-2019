from challenge2019.cbf.user_cbf import *
from challenge2019.topPop.topPop import *


class HybridCold(object):

    def __init__(self):
        self.URM = None

        self.recommenderUserCBF = UserContentBasedFiltering()
        self.recommenderTopPop = TopPop()

    def fit(self, URM):

        self.URM = URM

        self.recommenderUserCBF.fit(URM)
        self.recommenderTopPop.fit(URM)

    def recommend(self, user_id, at=10):
        self.URM.eliminate_zeros()

        recommended_items = []
        expected_items_top_pop = self.recommenderTopPop.recommend(user_id, at=20)
        expected_items_user_cbf = self.recommenderUserCBF.recommend(user_id, at=10)

        if np.flip(np.sort(self.recommenderUserCBF.get_expected_ratings(user_id)))[0] > 0:

            recommended_items = expected_items_user_cbf

        else:
            i = 0
            while len(recommended_items) < 10:
                if expected_items_top_pop[i] not in recommended_items:
                    recommended_items.append(expected_items_top_pop[i])
                i += 1

        return recommended_items[0:at]


if __name__ == '__main__':
    recommender = HybridCold()
    Runner.run(recommender, True,
               batch_evaluation=True, split='2080')

    # best score on seed 69: MAP@10 : 0.03042666580147029
    # 0.03298346361837503
    # 0.03291286461327143
