import time

import numpy as np
import pandas as pd
import scipy.sparse as sps


class Utils(object):
    def __init__(self, URM):
        self.URM = URM
        self.user_list = np.asarray(list(self.URM.row))
        self.item_list = np.asarray(list(self.URM.col))

    def get_urm_from_csv(self):
        interaction_list = list(np.ones(len(self.item_list)))
        URM = sps.coo_matrix((interaction_list, (self.user_list, self.item_list)), dtype=np.float64)
        URM = URM.tocsr()
        return URM