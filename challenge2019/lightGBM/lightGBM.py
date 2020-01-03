import lightgbm as lgb
import numpy as np
import pandas as pd

from challenge2019.utils.run import Runner
from utils.utils import Utils


class LightGBMRecommender(object):

    def __init__(self):
        True

    def fit(self, URM):
        utils = Utils()
        param = {}
        param['metric'] = 'map'
        self.model = lgb.LGBMRanker(**param)
        X = np.array((URM.nonzero()[0], URM.nonzero()[1])).transpose()
        group = [329687]
        y = np.ones(len(URM.data))
        self.model.fit(X, y, verbose=True, group=group)




if __name__ == '__main__':
    recommender = LightGBMRecommender()
    Runner.run(recommender, True, split='random_all')
