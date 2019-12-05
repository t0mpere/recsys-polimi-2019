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

    def fit(self, URM_train, num_factors=600, random_seed=None):
        self.URM_train = URM_train
        print("Computing SVD decomposition...")

        U, Sigma, VT = randomized_svd(self.URM_train,
                                      n_components=num_factors,
                                      # n_iter=5,
                                      random_state=random_seed)

        s_Vt = sps.diags(Sigma) * VT

        self.USER_factors = U
        self.ITEM_factors = s_Vt.T

        self.item_scores = np.dot(self.USER_factors, self.ITEM_factors.T)

        print("Computing SVD decomposition... Done!")

    def _compute_item_score_postprocess_for_cold_items(self, item_scores):
        """
        Remove cold items from the computed item scores, setting them to -inf
        :param item_scores:
        :return:
        """

        # Set as -inf all cold items scores
        if self._get_cold_item_mask().any():
            item_scores[:, self._get_cold_item_mask()] = - np.ones_like(
                item_scores[:, self._get_cold_item_mask()]) * np.inf

        return item_scores

    def get_expected_values(self, user_id, normalized_ratings=False):

        item_scores = self.item_scores[user_id]
        item_scores = np.squeeze(np.asarray(item_scores))

        print(max(item_scores), user_id, len(self.URM_train[user_id].data))

        # Normalize ratings
        if normalized_ratings and np.amax(item_scores) > 0:
            item_scores = item_scores / np.linalg.norm(item_scores)

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
    Runner.run(recommender, True, find_hyper_parameters_pureSVD=False, evaluate_different_region_of_users=True,
               batch_evaluation=False)
