import numpy as np
import scipy.sparse as sps
from tqdm import tqdm

from challenge2019.Base.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
from challenge2019.utils.run import Runner
from challenge2019.utils.utils import Utils


class TopPopUserClasses():

    def __init__(self):
        self.URM = None

    def fit(self, URM, knn=100, shrink=5, similarity="cosine"):
        utils = Utils()
        self.UCM_age = utils.get_ucm_region_from_csv_one_hot_encoding()
        self.UCM_region = utils.get_ucm_age_from_csv_one_hot_encoding()
        age_dict = dict()
        reg_dict = dict()
        for i in utils.get_user_list():
            if len(self.UCM_age[i].data) is not 0:
                age_dict.setdefault(int(self.UCM_age[i].data[0]), []).append(i)
            if len(self.UCM_region[i].data) is not 0:
                non_zero_cols = self.UCM_region[i].nonzero()[1]
                if len(non_zero_cols) > 1:
                    for nzc in non_zero_cols:
                        reg_dict.setdefault(nzc, []).append(i)
                else: reg_dict.setdefault(non_zero_cols[0], []).append(i)
        self.URM = URM

        self.reg_occurrencies = self.get_frequencies(reg_dict)
        self.age_occurrencies = self.get_frequencies(age_dict)

    def get_expected_ratings(self, user_id):
        data = np.arange(0.1, 1.1, 0.1)
        data = np.flip(data, 0)

        recommended_items = list(self.recommend(user_id))
        expected_ratings = np.zeros(self.URM.shape[1])
        for item in recommended_items:
            expected_ratings[item] = data[recommended_items.index(item)]
        return expected_ratings

    def get_frequencies(self, dict):
        occurrencies = np.array(np.zeros(shape=(len(dict.keys()) + 1, self.URM.shape[1])))
        for k in dict.keys():
            # print(self.URM[age_dict[k], :].nonzero())
            for i in tqdm(set(self.URM[dict[k], :].nonzero()[0]), desc="Key: {}".format(k)):
                occurrencies[k][i] = len(self.URM[dict[k], i].data)
        return occurrencies

    def recommend(self, user_id, at=10):
        if len(self.UCM_age[user_id].data):
            age = self.UCM_age[user_id].data[0]
        else:
            age = 0
        if len(self.UCM_region[user_id].data):
            reg = self.UCM_region[user_id].data[0]
        else:
            reg = 0
        expected_ratings = self.age_occurrencies[int(age)] + self.reg_occurrencies[int(reg)]
        recommended_items = np.flip(np.argsort(expected_ratings), 0)

        unseen_items_mask = np.in1d(recommended_items, self.URM[user_id].indices,
                                    assume_unique=True, invert=True)
        recommended_items = recommended_items[unseen_items_mask]
        return recommended_items[0:at]


if __name__ == '__main__':
    recommender = TopPopUserClasses()
    Runner.run(recommender, True, evaluate_different_type_of_users=True, batch_evaluation=True)
    recommender.get_expected_ratings(1)
