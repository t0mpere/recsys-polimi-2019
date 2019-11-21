#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 07/09/17

@author: Maurizio Ferrari Dacrema
"""
import numpy as np

from tqdm import tqdm

from challenge2019.Base.Evaluation.Evaluator import EvaluatorProf
from challenge2019.Base.Recommender_utils import check_matrix
from challenge2019.Base.Recommender_utils import similarityMatrixTopK

from challenge2019.CythonCompiler.run_compile_subprocess import run_compile_subprocess

from challenge2019.Base.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
import os, sys
from challenge2019.SLIM.SLIM_BPR_Cython_Epoch import SLIM_BPR_Cython_Epoch
from challenge2019.utils.run import Runner


def estimate_required_MB(n_items, symmetric):
    requiredMB = 8 * n_items ** 2 / 1e+06

    if symmetric:
        requiredMB /= 2

    return requiredMB


def get_RAM_status():
    try:
        data_list = os.popen('free -t -m').readlines()[1].split()
        tot_m = float(data_list[1])
        used_m = float(data_list[2])
        available_m = float(data_list[6])

    except Exception as exc:

        print("Unable to read memory status: {}".format(str(exc)))

        tot_m, used_m, available_m = None, None, None

    return tot_m, used_m, available_m


class SLIM_BPR_Cython(Incremental_Training_Early_Stopping):
    RECOMMENDER_NAME = "SLIM_BPR_Recommender"

    def __init__(self,
                 verbose=True,
                 free_mem_threshold=0.5,
                 recompile_cython=False):

        super().__init__()
        assert free_mem_threshold >= 0.0 and free_mem_threshold <= 1.0, "SLIM_BPR_Recommender: free_mem_threshold must be between 0.0 and 1.0, provided was '{}'".format(
            free_mem_threshold)
        self.test_dictionary = {}
        self.free_mem_threshold = free_mem_threshold

        if recompile_cython:
            print("Compiling in Cython")
            self.runCompilationScript()
            print("Compilation Complete")

    def fit(self,
            URM,
            epochs=300,
            positive_threshold_BPR=None,
            train_with_sparse_weights=None,
            symmetric=True,
            verbose=False,
            random_seed=None,
            batch_size=1000, lambda_i=0.2, lambda_j=0.2, learning_rate=1e-5, topK=100,
            sgd_mode='adagrad', gamma=0.995, beta_1=0.9, beta_2=0.999,
            **earlystopping_kwargs):

        self.URM_train = URM
        self.n_users, self.n_items = self.URM_train.shape

        # Import compiled module

        self.symmetric = symmetric
        self.train_with_sparse_weights = train_with_sparse_weights

        if self.train_with_sparse_weights is None:

            # auto select
            required_m = estimate_required_MB(self.n_items, self.symmetric)

            total_m, _, available_m = get_RAM_status()

            if total_m is not None:
                string = "SLIM_BPR_Cython: Automatic selection of fastest train mode. Available RAM is {:.2f} MB ({:.2f}%) of {:.2f} MB, required is {:.2f} MB. ".format(
                    available_m, available_m / total_m * 100, total_m, required_m)
            else:
                string = "SLIM_BPR_Cython: Automatic selection of fastest train mode. Unable to get current RAM status, you may be using a non-Linux operating system. "

            if total_m is None or required_m / available_m < self.free_mem_threshold:
                print(string + "Using dense matrix.")
                self.train_with_sparse_weights = False
            else:
                print(string + "Using sparse matrix.")
                self.train_with_sparse_weights = True

        # Select only positive interactions
        URM_train_positive = self.URM_train.copy()

        self.positive_threshold_BPR = positive_threshold_BPR
        self.sgd_mode = sgd_mode
        self.epochs = epochs

        if self.positive_threshold_BPR is not None:
            URM_train_positive.data = URM_train_positive.data >= self.positive_threshold_BPR
            URM_train_positive.eliminate_zeros()

            assert URM_train_positive.nnz > 0, "SLIM_BPR_Cython: URM_train_positive is empty, positive threshold is too high"

        self.cythonEpoch = SLIM_BPR_Cython_Epoch(URM_train_positive,
                                                 train_with_sparse_weights=self.train_with_sparse_weights,
                                                 final_model_sparse_weights=True,
                                                 topK=topK,
                                                 learning_rate=learning_rate,
                                                 li_reg=lambda_i,
                                                 lj_reg=lambda_j,
                                                 batch_size=1,
                                                 symmetric=self.symmetric,
                                                 sgd_mode=sgd_mode,
                                                 verbose=False,
                                                 random_seed=random_seed,
                                                 gamma=gamma,
                                                 beta_1=beta_1,
                                                 beta_2=beta_2,)

        if (topK != False and topK < 1):
            raise ValueError(
                "TopK not valid. Acceptable values are either False or a positive integer value. Provided value was '{}'".format(
                    topK))
        self.topK = topK

        self.batch_size = batch_size
        self.lambda_i = lambda_i
        self.lambda_j = lambda_j
        self.learning_rate = learning_rate

        self.S_incremental = self.cythonEpoch.get_S()
        self.S_best = self.S_incremental.copy()

        self.random_split(URM, None)
        self.evaluator = EvaluatorProf(URM_test_list = self.URM_train, cutoff_list = self.test_dictionary)

        self._train_with_early_stopping(epochs_max = 100,
                                        epochs_min = 0,
                                        evaluator_object = self.evaluator,
                                        stop_on_validation = True,
                                        validation_every_n =3,
                                        validation_metric = 'MAP',
                                        lower_validations_allowed = 5
                                        )

        self.get_S_incremental_and_set_W()
        self.RECS = self.URM_train.dot(self.W_sparse)
        self.cythonEpoch._dealloc()

        sys.stdout.flush()

    def _prepare_model_for_validation(self):
        self.get_S_incremental_and_set_W()

    def _update_best_model(self):
        self.S_best = self.S_incremental.copy()

    def _run_epoch(self, num_epoch):
        self.cythonEpoch.epochIteration_Cython()

    def get_S_incremental_and_set_W(self):

        self.S_incremental = self.cythonEpoch.get_S()

        if self.train_with_sparse_weights:
            self.W_sparse = self.S_incremental
            self.W_sparse = check_matrix(self.W_sparse, format='csr')
        else:
            self.W_sparse = similarityMatrixTopK(self.S_incremental, k=self.topK)
            self.W_sparse = check_matrix(self.W_sparse, format='csr')





    def writeCurrentConfig(self, currentEpoch, results_run, logFile):

        current_config = {'lambda_i': self.lambda_i,
                          'lambda_j': self.lambda_j,
                          'batch_size': self.batch_size,
                          'learn_rate': self.learning_rate,
                          'topK_similarity': self.topK,
                          'epoch': currentEpoch}

        print("Test case: {}\nResults {}\n".format(current_config, results_run))
        # print("Weights: {}\n".format(str(list(self.weights))))

        sys.stdout.flush()

        if (logFile != None):
            logFile.write("Test case: {}, Results {}\n".format(current_config, results_run))
            logFile.flush()

    def get_expected_ratings(self, playlist_id):
        expected_ratings = self.RECS[playlist_id].todense()
        return np.squeeze(np.asarray(expected_ratings))

    def recommend(self, playlist_id, at=10):

        # compute the scores using the dot product
        scores = self.get_expected_ratings(playlist_id)
        ranking = scores.argsort()[::-1]
        unseen_items_mask = np.in1d(ranking, self.URM_train[playlist_id].indices, assume_unique=True, invert=True)
        ranking = ranking[unseen_items_mask]
        return ranking[:at]

    def runCompilationScript(self):
        return None

        # Split random, 20% of each user

    def random_split(self, URM, URM_csv = None):
        user_indexes = np.arange(URM.shape[0])
        tmp = 0
        print("Splitting using random 20%\n---------------------")
        for user_index in tqdm(user_indexes, desc="Splitting dataset: "):
            # FOREACH USER
            item_left = len(URM[user_index].data)

            if item_left > 4:
                # If has more than 3 interactions

                # Array with the indexes of the non zero values
                non_zero = URM[user_index].indices
                # Shuffle array of indices
                np.random.shuffle(non_zero)
                # Select 20% of the array
                non_zero = non_zero[:min(int(len(non_zero) * .2), 9)]
                # Change values
                URM[user_index, non_zero] = 0
                URM.eliminate_zeros()
                self.test_dictionary[user_index] = non_zero
                tmp += len(self.test_dictionary[user_index])

            else:
                self.test_dictionary[user_index] = []

        self.URM_train = URM
        print('Number of element in test : {} \nNumber of elements in training : {}'.format(tmp,
                                                                                            len(URM.data)))


recommender = SLIM_BPR_Cython(recompile_cython=False)
Runner.run(recommender, False)
