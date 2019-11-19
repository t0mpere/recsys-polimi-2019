from random import randint

import numpy as np
import pandas as pd
import scipy
import random

from tqdm import tqdm

from .utils import Utils


class Evaluator(object):

    def __init__(self):
        self.URM_train = None
        self.URM_test = None
        self.test_dictionary = {}
        self.train_test_split = 0.7

    # Split totally randomic, no cluster
    def random_split(self, URM, URM_csv):
        user_indexes = np.arange(URM.shape[0])

        URM_test = URM.copy()
        URM_test.data[:] = 0
        for user_index in user_indexes:
            # FOREACH USER
            item_left = len(URM[user_index].data)
            if (item_left > 3):
                # If has more than 5 interactions

                # Array with the indexes of the non zero values
                non_zero = URM[user_index].indices
                # Shuffle array of indices
                np.random.shuffle(non_zero)
                # Select 20% of the array
                non_zero = non_zero[:int(len(non_zero) * .2)]
                # Change values
                URM[user_index, non_zero] = 0
                URM_test[user_index, non_zero] = 1
                if np.logical_and(URM[user_index].todense(), URM_test[user_index].todense()).any() == 1:
                    print("error")
            self.test_dictionary[user_index] = URM_test[user_index].indices

        self.URM_test = URM_test
        self.URM_train = URM
        print('Number of element in test : {} \nNumber of elements in training : {}'.format(URM_test.count_nonzero(),
                                                                                            URM.count_nonzero()))

    def MAP(self, recommended_items, relevant_items):

        is_relevant = np.isin(recommended_items, relevant_items, assume_unique=True)

        # Cumulative sum: precision at 1, at 2, at 3 ...
        p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(len(is_relevant)))
        print(p_at_k, is_relevant, recommended_items, relevant_items)

        map_score = np.sum(p_at_k) / np.min([len(relevant_items), len(is_relevant)])
        if map_score > 0: print(map_score)
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
