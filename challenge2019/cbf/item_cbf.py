import numpy as np

from challenge2019.Base.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
from challenge2019.utils.run import Runner
from challenge2019.utils.utils import Utils


class ItemContentBasedFiltering():
    def __init__(self):
        self.knn = None
        self.shrink = None
        self.similarity = None
        self.URM = None
        self.ICM_asset = None
        self.ICM_price = None
        self.ICM_sub_class = None
        self.SM_asset = None
        self.SM_price = None
        self.SM_sub_class = None

    def create_similarity_matrix(self, ICM):
        similarity_object = Compute_Similarity_Python(ICM.transpose(), topK=self.knn, shrink=self.shrink,
                                                      normalize=True, similarity=self.similarity)
        return similarity_object.compute_similarity()

    def fit(self, URM, knn=100, shrink=10, similarity="cosine"):
        utils = Utils()
        self.knn = knn
        self.shrink = shrink
        self.similarity = similarity
        self.URM = URM
        self.ICM_asset = utils.get_icm_asset_from_csv()
        self.ICM_price = utils.get_icm_price_from_csv()
        self.ICM_sub_class = utils.get_icm_sub_class_from_csv()

        # TODO: improve ICM (lezione 30/09)  + ICM DI UNA COLONNA PER TROVARE DISTANZA TRA I VALORI
        print("Starting calculating similarity")

        self.SM_asset = self.create_similarity_matrix(self.ICM_asset)
        self.SM_price = self.create_similarity_matrix(self.ICM_price)
        self.SM_sub_class = self.create_similarity_matrix(self.ICM_sub_class)
        self.RECS_asset = self.URM.dot(self.SM_asset)
        self.RECS_price = self.URM.dot(self.SM_price)
        self.RECS_sub_class = self.URM.dot(self.SM_sub_class)

    def get_expected_ratings(self, user_id, i=0.3, j=0.3, k=0.4, normalized_ratings=False):
        user_id = int(user_id)
        liked_items = self.URM[user_id]

        expected_ratings_assets = self.RECS_asset[user_id].todense()
        expected_ratings_price = self.RECS_price[user_id].todense()
        expected_ratings_sub_class = self.RECS_sub_class[user_id].todense()

        expected_ratings = + (expected_ratings_price * j) \
                           + (expected_ratings_assets * i) \
                           + (expected_ratings_sub_class * k)
        expected_ratings = np.squeeze(np.asarray(expected_ratings))

        # Normalize ratings
        if normalized_ratings and np.amax(expected_ratings) > 0:
            expected_ratings = expected_ratings / np.linalg.norm(expected_ratings)

        return expected_ratings

        return expected_ratings

    def recommend(self, user_id, i=0.3, j=0.3, k=0.4, at=10):
        user_id = int(user_id)
        expected_ratings = self.get_expected_ratings(user_id, i, j, k)
        recommended_items = np.flip(np.argsort(expected_ratings), 0)

        unseen_items_mask = np.in1d(recommended_items, self.URM[user_id].indices,
                                    assume_unique=True, invert=True)
        recommended_items = recommended_items[unseen_items_mask]
        return recommended_items[0:at]

#recommender = ItemContentBasedFiltering()
#Runner.run(recommender, True)
