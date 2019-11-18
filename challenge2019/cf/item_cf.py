from challenge2019.lib.Compute_Similarity_Python import Compute_Similarity_Python
import numpy as np

class itemCollaborativeFiltering():
    def __init__(self, knn = 100, shrink = 5, similarity="cosine"):
        self.knn = knn
        self.shrink = shrink
        self.similarity = similarity
        self.URM = None
        self.SM_item = None

    def create_similarity_matrix(self):
        similarity_matrix = Compute_Similarity_Python(self, topK=self.knn, shrink=self.shrink, normalize=True, similarity=self.similarity)
        return similarity_matrix.compute_similarity()

    def fit(self, URM):
        self.SM_item = self.create_similarity_matrix()
        self.RECS = self.URM.dot(self.SM_item)

    def get_expected_ratings(self, user_id):
        user_id = int(user_id)
        expected_ratings = self.RECS[user_id].todense()
        return np.squeeze(np.asarray(expected_ratings))

    def recommend(self, user_id, at=10):
        user_id = int(user_id)
        expected_ratings = self.get_expected_ratings(user_id)
        recommended_items = np.flip(np.argsort(expected_ratings), 0)

        unseen_items_mask = np.in1d(recommended_items, self.URM[user_id].indices,
                                    assume_unique=True, invert=True)
        recommended_items = recommended_items[unseen_items_mask]
        return recommended_items[0:at]
