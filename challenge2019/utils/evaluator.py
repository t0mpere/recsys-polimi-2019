from random import randint

import numpy as np
import pandas as pd
import scipy
import random

from scipy.sparse import csr_matrix, lil_matrix
from tqdm import tqdm

from .utils import Utils


class Evaluator(object):

    def __init__(self):
        self.URM_train = None
        self.test_dictionary = {}
        self.train_test_split = 0.7

    # Split random, 20% of each user
    def random_split(self, URM, URM_csv):
        user_indexes = np.arange(URM.shape[0])
        tmp = 0
        print(len(URM.data))
        for user_index in tqdm(user_indexes,desc="Splitting dataset: "):
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

    def MAP(self, recommended_items, relevant_items):
        # print(recommended_items)
        is_relevant = np.isin(recommended_items, relevant_items, assume_unique=True)

        # Cumulative sum: precision at 1, at 2, at 3 ...
        p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(len(is_relevant)))
        # print(recommended_items, relevant_items)
        map_score = np.sum(p_at_k) / np.min([len(relevant_items), len(is_relevant)])
        # if map_score == 0: print(recommended_items,relevant_items)
        return map_score

    def evaluate(self, user_id, recommended_items):
        relevant_items = self.test_dictionary[user_id]
        if(len(relevant_items) is not 0):
            map = self.MAP(recommended_items, relevant_items)
            return map
        else:
            return 0

    def eval_recommender(self, recommender):
        MAP_final = 0
        recommender.fit(self.URM_train)
        count = 0
        for user_id in tqdm(Utils.get_target_user_list()):
            recommended_items = recommender.recommend(user_id)
            MAP_final += self.evaluate(user_id, recommended_items)
            count += 1
        MAP_final /= len(Utils.get_target_user_list())
        return MAP_final
