from random import randint

import numpy as np
import pandas as pd

class Evaluator(object):

    def __init__(self):
        self.URM_train = None
        self.URM_test = None
        self.train_test_split = 0.8




#Split totally randomic, no cluster
    def random_split(self, URM, URM_csv):

        numInteractions = np.arange(URM.shape[0])
        len_test_set = (1 - self.train_test_split) * len(numInteractions)

        test_set = np.array([])

        print(numInteractions)

        while len_test_set > 0:

            random_index = randint(0 , len(numInteractions) - 1)

            user_index = numInteractions[random_index]
            item_left = len(URM[user_index].data)
            if(item_left > 5):

                numInteractions = np.delete(numInteractions, np.where(numInteractions == user_index))
                test_set = np.append(test_set, user_index)

                len_test_set -= 1

        print(len(numInteractions))
        print('number of element in validation : {}'.format(len(test_set)) )
