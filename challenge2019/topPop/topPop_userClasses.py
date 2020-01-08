import numpy as np
import scipy.sparse as sps
from tqdm import tqdm

from challenge2019.Base.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
from challenge2019.utils.run import Runner
from challenge2019.utils.utils import Utils
from challenge2019.topPop.topPop import TopPop


class TopPopUserClasses():

    def __init__(self):
        self.URM = None
        self.TopPop = TopPop()

    def fit(self, URM):
        self.URM = URM
        self.utils = Utils()
        self.UCM_age = self.utils.get_ucm_age_from_csv_one_hot_encoding()
        self.UCM_region = self.utils.get_ucm_region_from_csv_one_hot_encoding()
        self.TopPop.fit(URM)

        self.recommenders_age = self.get_fitted_recommenders(self.UCM_age, URM.copy())
        self.recommenders_region = self.get_fitted_recommenders(self.UCM_age, URM.copy())



    def get_fitted_recommenders(self, UCM, URM):
        recommenders_list = list()
        for i in range(UCM.shape[1]):
            mask = self.UCM_age.getcol(i).todense()
            mask = mask.astype(dtype=bool)
            toppop = TopPop()
            indices = np.where(mask)[0]
            toppop.fit(URM[indices,:], verbose=False)
            recommenders_list.append(toppop)
        return recommenders_list

    def get_frequencies(self, dict):
        occurrencies = np.array(np.zeros(shape=(len(dict.keys()) + 1, self.URM.shape[1])))
        for k in dict.keys():
            # print(self.URM[age_dict[k], :].nonzero())
            for i in tqdm(set(self.URM[dict[k], :].nonzero()[0]), desc="Key: {}".format(k)):
                occurrencies[k][i] = len(self.URM[dict[k], i].data)
        return occurrencies

    def recommend(self, user_id, at=10):
        UCM_age = self.utils.get_ucm_age_from_csv()
        if len(UCM_age[user_id].data):
            age = UCM_age[user_id].data[0]
        else:
            age = 0

        expected_ratings = self.recommenders_age[int(age)].get_occurrencies()

        UCM_region = self.utils.get_ucm_region_from_csv()
        region = UCM_region[user_id].data
        if len(UCM_region[user_id].data):
            for i in UCM_region[user_id].data:
                expected_ratings += self.recommenders_region[int(region)].get_occurrencies()

        if sum(expected_ratings) == 0:
            expected_ratings = self.TopPop.get_occurrencies()

        recommended_items = np.flip(np.argsort(expected_ratings), 0)

        unseen_items_mask = np.in1d(recommended_items, self.URM[user_id].indices,
                                    assume_unique=True, invert=True)
        recommended_items = recommended_items[unseen_items_mask]
        return recommended_items[0:at]


if __name__ == '__main__':
    recommender = TopPopUserClasses()
    Runner.run(recommender, True, evaluate_different_type_of_users=True, batch_evaluation=True)
    recommender.get_expected_ratings(1)
