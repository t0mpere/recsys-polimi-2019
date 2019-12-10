from lightfm import LightFM
import numpy as np

from challenge2019.utils.run import Runner
from challenge2019.utils.utils import Utils
import scipy.sparse as sps


class FM(object):
    def __init__(self):
        self.utils = Utils()
        self.URM = None
        self.model = LightFM(no_components=100, loss='warp-kos')
        self.ICM_asset = self.utils.get_icm_asset_from_csv()
        self.ICM_price = self.utils.get_icm_price_from_csv()
        self.ICM_sub_class = self.utils.get_icm_sub_class_from_csv()
        self.combined_ICM = sps.hstack([self.ICM_asset, self.ICM_sub_class, self.ICM_price])

    def fit(self, URM):
        self.URM = URM
        self.model.fit(URM.tocoo(), item_features=self.ICM_sub_class.tocsr(), epochs=30, num_threads=12, verbose=True)

    def get_expected_ratings(self, user_id, normalized_ratings=False):
        expected_ratings = self.model.predict(user_id, np.arange(self.ICM_sub_class.shape[0]),
                                              item_features=self.ICM_sub_class.tocsr())
        return expected_ratings

    def recommend(self, user_id, at=10):
        expected_ratings = self.get_expected_ratings(user_id)

        recommended_items = np.flip(np.argsort(expected_ratings), 0)

        unseen_items_mask = np.in1d(recommended_items, self.URM[user_id].indices,
                                    assume_unique=True, invert=True)
        recommended_items = recommended_items[unseen_items_mask]
        print(recommended_items)
        return recommended_items[0:at]


if __name__ == '__main__':
    recommender = FM()
    Runner.run(recommender, True, find_hyper_parameters_cf=False, evaluate_different_type_of_users=True,
               batch_evaluation=False)
