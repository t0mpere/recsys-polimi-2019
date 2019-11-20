import numpy as np
import scipy as sps
import time
import sys
import matplotlib.pyplot as plt
import random
from scipy import stats
from scipy.optimize import fmin
from scipy.sparse import csc_matrix, csr_matrix
from sklearn.linear_model import ElasticNet
from tqdm import tqdm

from challenge2019.utils.run import  Runner

class SLIMElasticNetRecommender(object):

    def __init__(self,  l1_penalty=0.1, l2_penalty=0.1, positive_only=True, topK=100):
        super(SLIMElasticNetRecommender, self).__init__()
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.positive_only = positive_only
        self.topK = topK

        if self.l1_penalty + self.l2_penalty != 0:
            self.l1_ratio = self.l1_penalty / (self.l1_penalty + self.l2_penalty)
        else:
            print("SLIM_ElasticNet: l1_penalty+l2_penalty cannot be equal to zero, setting the ratio l1/(l1+l2) to 1.0")
            self.l1_ratio = 1.0

    def fit(self, URM):
        self.URM = URM

        # initialize the ElasticNet model
        print("porcodio0")
        self.model = ElasticNet(alpha=1.0,
                                l1_ratio=self.l1_ratio,
                                positive=self.positive_only,
                                fit_intercept=False,
                                copy_X=False,
                                precompute=True,
                                selection='random',
                                max_iter=10,
                                tol=1e-2)

        URM = csc_matrix(self.URM)

        n_items = URM.shape[1]

        # Use array as it reduces memory requirements compared to lists
        dataBlock = 10000000

        rows = np.zeros(dataBlock, dtype=np.int32)
        cols = np.zeros(dataBlock, dtype=np.int32)
        values = np.zeros(dataBlock, dtype=np.float32)

        numCells = 0

        start_time = time.time()
        start_time_printBatch = start_time

        # fit each item's factors sequentially (not in parallel)
        for currentItem in tqdm(range(n_items)):

            # get the target column
            y = URM[:, currentItem].toarray()

            # set the j-th column of X to zero
            start_pos = URM.indptr[currentItem]
            end_pos = URM.indptr[currentItem + 1]

            current_item_data_backup = URM.data[start_pos: end_pos].copy()
            URM.data[start_pos: end_pos] = 0.0

            # fit one ElasticNet model per column
            self.model.fit(URM, y)

            # self.model.coef_ contains the coefficient of the ElasticNet model
            # let's keep only the non-zero values

            # Select topK values
            # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
            # - Partition the data to extract the set of relevant items
            # - Sort only the relevant items
            # - Get the original item index

            nonzero_model_coef_index = self.model.sparse_coef_.indices
            nonzero_model_coef_value = self.model.sparse_coef_.data

            local_topK = min(len(nonzero_model_coef_value) - 1, self.topK)

            relevant_items_partition = (-nonzero_model_coef_value).argpartition(local_topK)[0:local_topK]
            relevant_items_partition_sorting = np.argsort(-nonzero_model_coef_value[relevant_items_partition])
            ranking = relevant_items_partition[relevant_items_partition_sorting]

            for index in range(len(ranking)):

                if numCells == len(rows):
                    rows = np.concatenate((rows, np.zeros(dataBlock, dtype=np.int32)))
                    cols = np.concatenate((cols, np.zeros(dataBlock, dtype=np.int32)))
                    values = np.concatenate((values, np.zeros(dataBlock, dtype=np.float32)))

                rows[numCells] = nonzero_model_coef_index[ranking[index]]
                cols[numCells] = currentItem
                values[numCells] = nonzero_model_coef_value[ranking[index]]

                numCells += 1

            # finally, replace the original values of the j-th column
            URM.data[start_pos:end_pos] = current_item_data_backup

            if time.time() - start_time_printBatch > 300 or currentItem == n_items - 1:
                print("Processed {} ( {:.2f}% ) in {:.2f} minutes. Items per second: {:.0f}".format(
                    currentItem + 1,
                    100.0 * float(currentItem + 1) / n_items,
                    (time.time() - start_time) / 60,
                    float(currentItem) / (time.time() - start_time)))
                sys.stdout.flush()
                sys.stderr.flush()

                start_time_printBatch = time.time()

        # generate the sparse weight matrix
        self.W_sparse = csr_matrix((values[:numCells], (rows[:numCells], cols[:numCells])),
                                       shape=(n_items, n_items), dtype=np.float32)


    def recommend(self, user_id, at=None, exclude_seen=True):
        # compute the scores using the dot product
        user_profile = self.URM[user_id]
        scores = user_profile.dot(self.W_sparse).toarray().ravel()

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        # rank items
        ranking = scores.argsort()[::-1]

        return ranking[:at]


    def filter_seen(self, user_id, scores):
        start_pos = self.URM.indptr[user_id]
        end_pos = self.URM.indptr[user_id + 1]

        user_profile = self.URM.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores

recommender = SLIMElasticNetRecommender()
Runner.run(recommender, True)