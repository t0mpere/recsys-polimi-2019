import numpy as np

from challenge2019.Base.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
from challenge2019.utils.run import Runner
from challenge2019.utils.utils import Utils


class UserContentBasedFiltering():
    def __init__(self, knn=100, shrink=5, similarity="cosine"):
        self.knn = knn
        self.shrink = shrink
        self.similarity = similarity
        self.URM = None
        self.UCM_region = None
        self.UCM_age = None
        self.SM_region = None
        self.SM_age = None

    def create_similarity_matrix(self, UCM):
        similarity_object = Compute_Similarity_Python(UCM.transpose(), topK=self.knn, shrink=self.shrink,
                                                      normalize=True, similarity=self.similarity)
        return similarity_object.compute_similarity()

    def fit(self, URM):
        utils = Utils()
        self.URM = URM
        self.UCM_region = utils.get_ucm_region_from_csv()
        self.UCM_age = utils.get_ucm_age_from_csv()

        # TODO: improve UCM (lezione 30/09)
        print("Starting calculating similarity")

        self.SM_region = self.create_similarity_matrix(self.UCM_region)
        self.SM_age = self.create_similarity_matrix(self.UCM_age)

        self.RECS_region = self.SM_region.dot(self.URM)
        self.RECS_age = self.SM_age.dot(self.URM)

    def get_expected_ratings(self, user_id, i, j):
        user_id = int(user_id)

        expected_ratings = (self.RECS_region[user_id].todense() * i) \
                           + (self.RECS_age[user_id].todense() * j)

        expected_ratings = np.squeeze(np.asarray(expected_ratings))

        return expected_ratings

    def recommend(self, user_id, i=0.5, j=0.5, at=10):
        user_id = int(user_id)
        expected_ratings = self.get_expected_ratings(user_id, i, j)
        recommended_items = np.flip(np.argsort(expected_ratings), 0)

        unseen_items_mask = np.in1d(recommended_items, self.URM[user_id].indices,
                                    assume_unique=True, invert=True)
        recommended_items = recommended_items[unseen_items_mask]
        return recommended_items[0:at]


#recommender = UserContentBasedFiltering()
#Runner.run(recommender, False)
