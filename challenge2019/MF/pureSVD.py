#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/06/18

@author: Maurizio Ferrari Dacrema
"""

from challenge2019.utils.run import Runner
from sklearn.utils.extmath import randomized_svd
import scipy.sparse as sps
import numpy as np


class PureSVDRecommender():
    """ PureSVDRecommender"""

    RECOMMENDER_NAME = "PureSVDRecommender"

    def __init__(self, verbose=True):
        self.URM_train = None
        self.USER_factors = None
        self.ITEM_factors = None

    def _get_cold_user_mask(self):
        self._cold_user_mask = np.ediff1d(self.URM_train.indptr) == 0
        return self._cold_user_mask

    def _get_cold_item_mask(self):
        self._cold_item_mask = np.ediff1d(self.URM_train.tocsc().indptr) == 0
        return self._cold_item_mask

    def fit(self, URM_train, num_factors=100, random_seed = None):
        self.URM_train = URM_train
        print("Computing SVD decomposition...")

        U, Sigma, VT = randomized_svd(self.URM_train,
                                      n_components=num_factors,
                                      #n_iter=5,
                                      random_state = random_seed)

        s_Vt = sps.diags(Sigma)*VT

        self.USER_factors = U
        self.ITEM_factors = s_Vt.T

        print("Computing SVD decomposition... Done!")

    def _compute_item_score_postprocess_for_cold_users(self, user_id_array, item_scores):
        """
        Remove cold users from the computed item scores, setting them to -inf
        Or estimate user factors with specified method
        :param user_id_array:
        :param item_scores:
        :return:
        """

        # todo: importa item_cbf (che fa schifo) in un'altra funzione e fanne il fit.

        cold_users_batch_mask = self._get_cold_user_mask()[user_id_array]

        # Set as -inf all cold user scores
        if cold_users_batch_mask.any():


            # Add KNN scores for users cold for MF but warm in KNN model
            cold_users_in_MF_warm_in_KNN_mask = np.logical_and(cold_users_batch_mask, self._warm_user_KNN_mask[user_id_array])

            item_scores[cold_users_in_MF_warm_in_KNN_mask, :] = self._ItemKNNRecommender._compute_item_score(user_id_array[cold_users_in_MF_warm_in_KNN_mask], items_to_compute=items_to_compute)

            # Set cold users as those neither in MF nor in KNN
            cold_users_batch_mask = np.logical_and(cold_users_batch_mask, np.logical_not(cold_users_in_MF_warm_in_KNN_mask))

        # Set as -inf all remaining cold user scores
        item_scores[cold_users_batch_mask, :] = - np.ones_like(item_scores[cold_users_batch_mask, :]) * np.inf

        return item_scores


    def get_expected_values(self, user_id_array, items_to_compute = None):
        """
        USER_factors is n_users x n_factors
        ITEM_factors is n_items x n_factors

        The prediction for cold users will always be -inf for ALL items

        :param user_id_array:
        :param items_to_compute:
        :return:
        """

        assert self.USER_factors.shape[1] == self.ITEM_factors.shape[1], \
            "{}: User and Item factors have inconsistent shape".format(self.RECOMMENDER_NAME)

        assert self.USER_factors.shape[0] > user_id_array,\
                "{}: Cold users not allowed. Users in trained model are {}, requested prediction for users up to {}".format(
                self.RECOMMENDER_NAME, self.USER_factors.shape[0], user_id_array)

        item_scores = np.dot(self.USER_factors[user_id_array], self.ITEM_factors.T)

        # item_scores = self._compute_item_score_postprocess_for_cold_users(user_id_array, item_scores)
        # item_scores = self._compute_item_score_postprocess_for_cold_items(item_scores)

        return item_scores

    def recommend(self, user_id, at=10):
        user_id = int(user_id)
        expected_ratings = self.get_expected_values(user_id)
        recommended_items = np.flip(np.argsort(expected_ratings), 0)

        unseen_items_mask = np.in1d(recommended_items, self.URM_train[user_id].indices,
                                    assume_unique=True, invert=True)
        recommended_items = recommended_items[unseen_items_mask]
        return recommended_items[0:at]

if __name__ == '__main__':
    recommender = PureSVDRecommender()
    Runner.run(recommender, False, find_hyper_parameters_pureSVD=True, evaluate_different_region_of_users=False, batch_evaluation=False)
