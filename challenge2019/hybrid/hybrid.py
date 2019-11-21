from challenge2019.cf.user_cf import *
from challenge2019.cf.item_cf import *
from challenge2019.utils.utils import *
from challenge2019.SLIM.SLIM_BPR_Cython import *

class Hybrid():

    def __init__(self, knn=100, shrink=5, similarity="cosine"):
        self.knn = knn
        self.shrink = shrink
        self.similarity = similarity
        self.URM = None
        self.SM_item = None
        self.recommenderUser = UserCollaborativeFiltering()
        self.recommenderItem = ItemCollaborativeFiltering()
        self.recommender_SLIM_BPR = SLIM_BPR_Cython()


    def fit(self,URM):
        self.URM = URM
        self.recommenderUser.fit(URM)
        self.recommenderItem.fit(URM)
        self.recommender_SLIM_BPR.fit(URM)

    def recommend(self, user_id, at=10):
        user_id = int(user_id)
        #todo add weight and
        expected_ratings = 0.1*self.recommenderUser.get_expected_ratings(user_id) + 0.3*self.recommenderItem.get_expected_ratings(user_id) + 0.9*self.recommender_SLIM_BPR.get_expected_ratings(user_id)
        recommended_items = np.flip(np.argsort(expected_ratings), 0)

        unseen_items_mask = np.in1d(recommended_items, self.URM[user_id].indices,
                                    assume_unique=True, invert=True)
        recommended_items = recommended_items[unseen_items_mask]
        return recommended_items[0:at]

recommender = Hybrid()
Runner.run(recommender, True)