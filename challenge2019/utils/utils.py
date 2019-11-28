import time

import numpy as np
import pandas as pd
import scipy.sparse as sps
from sklearn import feature_extraction


class Utils(object):

    #TODO change URM to generic matrix
    def __init__(self):
        self.URM_csv = pd.read_csv("../dataset/data_train.csv")
        self.ICM_asset_csv = pd.read_csv("../dataset/data_ICM_asset.csv")
        self.ICM_price_csv = pd.read_csv("../dataset/data_ICM_price.csv")
        self.ICM_sub_class_csv = pd.read_csv("../dataset/data_ICM_sub_class.csv")
        self.UCM_region = pd.read_csv("../dataset/data_UCM_region.csv")
        self.UCM_age = pd.read_csv("../dataset/data_UCM_age.csv")

        self.user_list = np.asarray(list(self.URM_csv.row))
        self.item_list_urm = np.asarray(list(self.URM_csv.col))

        self.item_list_icm_asset = np.asarray(list(self.ICM_asset_csv.row))
        self.item_asset_list = np.asarray(list(self.ICM_asset_csv.data))
        self.item_list_icm_price = np.asarray(list(self.ICM_price_csv.row))
        self.item_price_list = np.asarray(list(self.ICM_price_csv.data))
        self.item_list_icm_sub_class = np.asarray(list(self.ICM_sub_class_csv.row))
        self.item_sub_class_list = np.asarray(list(self.ICM_sub_class_csv.col))

        self.user_list_ucm_region = np.asarray(list(self.UCM_region.row))
        self.user_region_list = np.asarray(list(self.UCM_region.col))
        self.user_list_ucm_age = np.asarray(list(self.UCM_age.row))
        self.user_age_list = np.asarray(list(self.UCM_age.col))

    def get_urm_from_csv(self):
        interaction_list = list(np.ones(len(self.item_list_urm)))
        URM = sps.coo_matrix((interaction_list, (self.user_list, self.item_list_urm)), dtype=np.float64)
        URM = URM.tocsr()
        return URM

    def get_icm_asset_from_csv(self):
        scale = 1000000
        data_list = list(np.ones(len(self.item_asset_list)))
        self.item_asset_list = self.item_asset_list * scale
        self.item_asset_list = self.item_asset_list.astype(int)

        ICM_asset = sps.coo_matrix((data_list, (self.item_list_icm_asset, self.item_asset_list)), dtype=np.float64)
        ICM_asset = ICM_asset.tocsr()
        return ICM_asset

    def get_URM_tfidf(self, URM):
        URM_tfidf = feature_extraction.text.TfidfTransformer().fit_transform(URM)
        return URM_tfidf.tocsr()

    def get_icm_price_from_csv(self):

        data_list = np.zeros(len(self.item_price_list))
        data_list = data_list.astype(int)

        ICM_price = sps.coo_matrix((self.item_price_list, (self.item_list_icm_price, data_list)), dtype=np.float64)
        ICM_price = ICM_price.tocsr()
        return ICM_price

    def get_icm_sub_class_from_csv(self):
        data_list = list(np.ones(len(self.item_sub_class_list)))
        ICM_sub_class = sps.coo_matrix((data_list, (self.item_list_icm_sub_class, self.item_sub_class_list)), dtype=np.float64)
        ICM_sub_class = ICM_sub_class.tocsr()
        return ICM_sub_class

    def get_ucm_age_from_csv(self):
        data_list = np.zeros(len(self.user_age_list))
        data_list = data_list.astype(int)

        UCM_age = sps.coo_matrix((self.user_age_list, (self.user_list_ucm_age, data_list)), dtype=np.float64)
        UCM_age = UCM_age.tocsr()
        return UCM_age

    def get_ucm_region_from_csv(self):
        data_list = list(np.ones(len(self.user_region_list)))

        UCM_region = sps.coo_matrix((data_list, (self.user_list_ucm_region, self.user_region_list)), dtype=np.float64)
        UCM_region= UCM_region.tocsr()
        return UCM_region

    def get_user_list(self):
        return set(self.user_list)

    def get_cold_user_list(self):
        cold_users = []
        URM = self.get_urm_from_csv()
        URM.eliminate_zeros()
        for i in range(0, URM.shape[0]):
            if len(URM[i].data) < 4:
                cold_users.append(i)

        return cold_users

    @staticmethod
    def get_target_user_list():
        target_users_dataset = pd.read_csv("../dataset/data_target_users_test.csv")
        return list(target_users_dataset.user_id)

