import numpy as np
import scipy.sparse as sps
import torch
from challenge2019.Base.Similarity.Compute_Similarity_Cython import Compute_Similarity_Cython as Compute_Similarity_Python
from challenge2019.utils.run import Runner
from challenge2019.utils.utils import Utils


class ItemContentBasedFiltering():
    def __init__(self):
        self.knn = 900
        self.shrink = None
        self.similarity = None
        self.URM = None
        self.ICM_asset = None
        self.ICM_price = None
        self.ICM_sub_class = None
        self.SM_asset = None
        self.SM_price = None
        self.SM_sub_class = None
        self.knn_asset = None
        self.knn_price = None
        self.knn_sub_class = None
        self.weights = None

        self.ICM = None
        self.RECS = None
        self.SM = None

        self.swag_weights = {
            "price": 0.15,
            "asset": 0.35,
            "sub_class": 0.5
        }

    def create_similarity_matrix(self, ICM, knn=100, shrink=2):
        similarity_object = Compute_Similarity_Python(ICM.transpose(), topK=knn, shrink=shrink,
                                                      normalize=True, similarity=self.similarity)
        return similarity_object.compute_similarity()

    def fit(self, URM, weights=None, similarity="cosine"): #knn_asset=980, knn_price=68, knn_sub_class=1000, shrink=2):
        if weights is not None:
            self.weights = weights
        else:
            self.weights = self.swag_weights

        utils = Utils()
        # self.knn_asset = knn_asset
        # self.knn_price = knn_price
        # self.knn_sub_class = knn_sub_class
        # self.shrink = shrink
        self.similarity = similarity
        self.URM = URM
        self.ICM_asset = utils.get_icm_asset_from_csv()
        self.ICM_price = utils.get_icm_price_from_csv()
        self.ICM_sub_class = utils.get_icm_sub_class_from_csv()
        self.ICM = sps.hstack([self.ICM_asset, self.ICM_sub_class, self.ICM_price])

        print("Starting calculating similarity ITEM_CBF")

        # self.SM_asset = self.create_similarity_matrix(self.ICM_asset, knn=self.knn_asset)
        # self.SM_price = self.create_similarity_matrix(self.ICM_price, knn=self.knn_price)
        # self.SM_sub_class = self.create_similarity_matrix(self.ICM_sub_class, knn=self.knn_sub_class)
        self.SM = self.create_similarity_matrix(self.ICM, knn=5, shrink=120)

        # self.RECS_asset = self.URM.dot(self.SM_asset)
        # self.RECS_price = self.URM.dot(self.SM_price)
        # self.RECS_sub_class = self.URM.dot(self.SM_sub_class)
        self.RECS = self.URM.dot(self.SM)


    def get_expected_ratings(self, user_id, normalized_ratings=False):
        user_id = int(user_id)

        # expected_ratings_assets = self.RECS_asset[user_id].todense()
        # expected_ratings_price = self.RECS_price[user_id].todense()
        # expected_ratings_sub_class = self.RECS_sub_class[user_id].todense()

        expected_ratings = self.RECS[user_id].todense()

        # expected_ratings = + (expected_ratings_price * self.weights["price"]) \
        #                    + (expected_ratings_assets * self.weights["asset"]) \
        #                    + (expected_ratings_sub_class * self.weights["sub_class"])

        expected_ratings = np.squeeze(np.asarray(expected_ratings))

        # Normalize ratings
        if normalized_ratings and np.amax(expected_ratings) > 0:
            expected_ratings = expected_ratings / np.linalg.norm(expected_ratings)

        return expected_ratings

    # change how weights are handled
    def recommend(self, user_id, at=10):
        user_id = int(user_id)
        expected_ratings = self.get_expected_ratings(user_id)
        recommended_items = np.flip(np.argsort(expected_ratings), 0)

        unseen_items_mask = np.in1d(recommended_items, self.URM[user_id].indices,
                                    assume_unique=True, invert=True)
        recommended_items = recommended_items[unseen_items_mask]
        return recommended_items[0:at]


if __name__ == '__main__':
    recommender = ItemContentBasedFiltering()
    Runner.run(recommender, True, find_hyper_parameters_item_cbf=False, find_weights_item_cbf=False, batch_evaluation=True)
