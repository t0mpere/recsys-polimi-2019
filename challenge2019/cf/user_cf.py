import numpy as np

from challenge2019.Base.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
from challenge2019.utils.run import Runner
from challenge2019.utils.utils import Utils

class UserCollaborativeFiltering():
    def __init__(self):
        self.knn = None
        self.shrink = None
        self.similarity = None
        self.URM = None
        self.SM_item = None

    def create_similarity_matrix(self):
        similarity_object = Compute_Similarity_Python(self.URM.transpose(), topK=self.knn, shrink=self.shrink, normalize=True, similarity=self.similarity)
        return similarity_object.compute_similarity()

    def fit(self, URM, knn=100, shrink=5, similarity="cosine"):
        self.knn = knn
        self.shrink = shrink
        self.similarity = similarity
        print("Starting calculating similarity")

        self.URM = URM
        self.SM_user = self.create_similarity_matrix()
        self.RECS = self.SM_user.dot(self.URM)

    def get_expected_ratings(self, user_id, normalized_ratings=False):
        user_id = int(user_id)
        expected_ratings = self.RECS[user_id].todense()
        expected_ratings = np.squeeze(np.asarray(expected_ratings))

        # Normalize ratings
        if normalized_ratings and np.amax(expected_ratings) > 0:
            expected_ratings = expected_ratings / np.linalg.norm(expected_ratings)

        return expected_ratings

    def recommend(self, user_id, at=10):
        user_id = int(user_id)
        expected_ratings = self.get_expected_ratings(user_id)
        recommended_items = np.flip(np.argsort(expected_ratings), 0)

        unseen_items_mask = np.in1d(recommended_items, self.URM[user_id].indices,
                                    assume_unique=True, invert=True)
        recommended_items = recommended_items[unseen_items_mask]
        return recommended_items[0:at]

#recommender = UserCollaborativeFiltering()
#Runner.run(recommender, True, train_cf=True)