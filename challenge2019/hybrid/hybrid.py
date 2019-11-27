from challenge2019.cf.user_cf import *
from challenge2019.cf.item_cf import *
from challenge2019.cbf.item_cbf import *
from challenge2019.SLIM.SLIM_BPR_Cython import *
from challenge2019.topPop.topPop import *
from challenge2019.cbf.user_cbf import *


class Hybrid():

    def __init__(self):
        self.URM = None
        self.SM_item = None
        self.recommenderUser = UserCollaborativeFiltering()
        self.recommenderItem = ItemCollaborativeFiltering()
        self.recommender_SLIM_BPR = SLIM_BPR_Cython()
        self.recommenderItemCBF = ItemContentBasedFiltering()
        self.recommenderUserCBF = UserContentBasedFiltering()
        #self.recommenderTopPop = TopPop()

    def fit(self, URM):
        self.URM = URM
        #self.recommenderUser.fit(URM, knn=600, shrink=5)
        self.recommenderItem.fit(URM, knn=20, shrink=19)
        #self.recommender_SLIM_BPR.fit(URM, epochs=200, lambda_i=0.2, lambda_j=0.2, topk=200)
        #self.recommenderItemCBF.fit(URM, knn_asset=100, knn_price=100, knn_sub_class=300, shrink=10)
        self.recommenderUserCBF.fit(URM, knn_age=1000, knn_region=1000, shrink=20)
        #self.recommenderTopPop.fit(URM)

    def recommend(self, user_id, at=10):
        user_id = int(user_id)
        normalized_ratings = True
        # todo add weight and

        self.URM.eliminate_zeros()
        liked_items = self.URM[user_id]

        if len(liked_items.data) < 4:
            # Provare a usare cbf al posto di top pop
            # Using user_cbf
            #recommended_items = self.recommenderUserCBF.recommend(user_id)

            expected_ratings = 0.2 * self.recommenderItem.get_expected_ratings(user_id,
                                                                               normalized_ratings=normalized_ratings) \
                               + 0.8 * self.recommenderUserCBF.get_expected_ratings(user_id,
                                                                                    normalized_ratings=normalized_ratings)
            recommended_items = np.flip(np.argsort(expected_ratings), 0)

            # Using TopPop
            # recommended_items = self.recommenderTopPop.recommend(user_id)

        else:
            expected_ratings = 0.9 * self.recommenderItem.get_expected_ratings(user_id, normalized_ratings=normalized_ratings)\
                               + 0.1 * self.recommenderUserCBF.get_expected_ratings(user_id, normalized_ratings=normalized_ratings)
                               # 0.2 * self.recommenderUser.get_expected_ratings(user_id, normalized_ratings=normalized_ratings) \
                               #+ 0.1 * self.recommender_SLIM_BPR.get_expected_ratings(user_id, normalized_ratings=normalized_ratings) \
                               #+ 0.025 * self.recommenderItemCBF.get_expected_ratings(user_id, normalized_ratings=normalized_ratings) \

            recommended_items = np.flip(np.argsort(expected_ratings), 0)

        unseen_items_mask = np.in1d(recommended_items, self.URM[user_id].indices,
                                    assume_unique=True, invert=True)
        recommended_items = recommended_items[unseen_items_mask]
        return recommended_items[0:at]


recommender = Hybrid()
Runner.run(recommender, True)
