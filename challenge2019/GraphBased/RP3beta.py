#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Cesare Bernardis
"""

import numpy as np
import scipy.sparse as sps

from sklearn.preprocessing import normalize
from challenge2019.Base.Recommender_utils import check_matrix, similarityMatrixTopK

import time, sys

from challenge2019.utils.run import Runner
from challenge2019.utils.utils import Utils


class RP3betaRecommender():
    """ RP3beta recommender """

    RECOMMENDER_NAME = "RP3betaRecommender"

    def __init__(self):
        self.URM_train = None

    def __str__(self):
        return "RP3beta(alpha={}, beta={}, min_rating={}, topk={}, implicit={}, normalize_similarity={})".format(
            self.alpha,
            self.beta, self.min_rating, self.topK,
            self.implicit, self.normalize_similarity)

    def fit(self, URM_train, alpha=.528, beta=0.1592, min_rating=0, topK=72, implicit=True, normalize_similarity=True, use_ICM=True):


        utils = Utils()
        self.URM_train = URM_train
        utils = Utils()
        self.ICM = utils.get_icm()
        if use_ICM:
            print("Using UCM + ICM")
            self.URM_train = sps.vstack([self.URM_train, self.ICM.transpose()]).tocsr()

        # self.URM_train = utils.get_URM_BM_25(self.URM_train) <--- worst
        # self.URM_train = utils.get_URM_tfidf(self.URM_train) <--- worst

        #self.URM_train = utils.weight_interactions(URM_train)

        self.alpha = alpha
        self.beta = beta
        self.min_rating = min_rating
        self.topK = topK
        self.implicit = implicit
        self.normalize_similarity = normalize_similarity

        # if X.dtype != np.float32:
        #     print("RP3beta fit: For memory usage reasons, we suggest to use np.float32 as dtype for the dataset")

        if self.min_rating > 0:
            self.URM_train.data[self.URM_train.data < self.min_rating] = 0
            self.URM_train.eliminate_zeros()
            if self.implicit:
                self.URM_train.data = np.ones(self.URM_train.data.size, dtype=np.float32)

        # Pui is the row-normalized urm
        Pui = normalize(self.URM_train, norm='l1', axis=1)

        # Piu is the column-normalized, "boolean" urm transposed
        X_bool = self.URM_train.transpose(copy=True)
        X_bool.data = np.ones(X_bool.data.size, np.float32)

        # Taking the degree of each item to penalize top popular
        # Some rows might be zero, make sure their degree remains zero
        X_bool_sum = np.array(X_bool.sum(axis=1)).ravel()

        degree = np.zeros(self.URM_train.shape[1])

        nonZeroMask = X_bool_sum != 0.0

        degree[nonZeroMask] = np.power(X_bool_sum[nonZeroMask], -self.beta)

        # ATTENTION: axis is still 1 because i transposed before the normalization
        Piu = normalize(X_bool, norm='l1', axis=1)
        del (X_bool)

        # Alfa power
        if self.alpha != 1.:
            Pui = Pui.power(self.alpha)
            Piu = Piu.power(self.alpha)

        # Final matrix is computed as Pui * Piu * Pui
        # Multiplication unpacked for memory usage reasons
        block_dim = 200
        d_t = Piu

        # Use array as it reduces memory requirements compared to lists
        dataBlock = 10000000

        rows = np.zeros(dataBlock, dtype=np.int32)
        cols = np.zeros(dataBlock, dtype=np.int32)
        values = np.zeros(dataBlock, dtype=np.float32)

        numCells = 0

        start_time = time.time()
        start_time_printBatch = start_time

        for current_block_start_row in range(0, Pui.shape[1], block_dim):

            if current_block_start_row + block_dim > Pui.shape[1]:
                block_dim = Pui.shape[1] - current_block_start_row

            similarity_block = d_t[current_block_start_row:current_block_start_row + block_dim, :] * Pui
            similarity_block = similarity_block.toarray()

            for row_in_block in range(block_dim):
                row_data = np.multiply(similarity_block[row_in_block, :], degree)
                row_data[current_block_start_row + row_in_block] = 0

                best = row_data.argsort()[::-1][:self.topK]

                notZerosMask = row_data[best] != 0.0

                values_to_add = row_data[best][notZerosMask]
                cols_to_add = best[notZerosMask]

                for index in range(len(values_to_add)):

                    if numCells == len(rows):
                        rows = np.concatenate((rows, np.zeros(dataBlock, dtype=np.int32)))
                        cols = np.concatenate((cols, np.zeros(dataBlock, dtype=np.int32)))
                        values = np.concatenate((values, np.zeros(dataBlock, dtype=np.float32)))

                    rows[numCells] = current_block_start_row + row_in_block
                    cols[numCells] = cols_to_add[index]
                    values[numCells] = values_to_add[index]

                    numCells += 1

            if time.time() - start_time_printBatch > 60:
                print("Processed {} ( {:.2f}% ) in {:.2f} minutes. Rows per second: {:.0f}".format(
                    current_block_start_row,
                    100.0 * float(current_block_start_row) / Pui.shape[1],
                    (time.time() - start_time) / 60,
                    float(current_block_start_row) / (time.time() - start_time)))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_printBatch = time.time()

        self.W_sparse = sps.csr_matrix((values[:numCells], (rows[:numCells], cols[:numCells])),
                                       shape=(Pui.shape[1], Pui.shape[1]))

        if self.normalize_similarity:
            self.W_sparse = normalize(self.W_sparse, norm='l1', axis=1)

        if self.topK != False:
            self.W_sparse = similarityMatrixTopK(self.W_sparse, k=self.topK)

        self.W_sparse = check_matrix(self.W_sparse, format='csr')
        self.RECS = self.URM_train.dot(self.W_sparse)
        self.fitted = True

    def get_W(self):
        if self.fitted:
            return self.W_sparse
        else:
            raise Exception("Te ga smenà")

    def get_expected_ratings(self, user_id, normalized_ratings=True):
        expected_ratings = self.RECS[user_id].todense()
        expected_ratings = np.squeeze(np.asarray(expected_ratings))
        return expected_ratings

    def recommend(self, playlist_id, at=10):

        # compute the scores using the dot product
        scores = self.get_expected_ratings(playlist_id)

        ranking = scores.argsort()[::-1]
        unseen_items_mask = np.in1d(ranking, self.URM_train[playlist_id].indices, assume_unique=True, invert=True)
        ranking = ranking[unseen_items_mask]
        ranking = ranking[:at]
        return ranking

if __name__ == '__main__':
    recommender = RP3betaRecommender()
    Runner.run(recommender, True, find_hyper_parameters_RP3beta=False, evaluate_different_type_of_users=True, batch_evaluation=True, split='2080')

# 0.02890 with seed 69
# 0.024040110720532526 long 123