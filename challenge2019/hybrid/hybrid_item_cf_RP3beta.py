from challenge2019.GraphBased.RP3beta import RP3betaRecommender
from challenge2019.cbf.user_cbf import *
from challenge2019.utils.utils import Utils

class HybridItemCfRP3Beta(object):

    def __init__(self, divide_recommendations=False):
        self.URM = None
        self.SM_item = None
        self.fitted = False
    # .4011
    # .24
    #
    def fit(self, URM, fit_once=False, alpha=.2842):
        self.alpha = alpha
        if not (fit_once and self.fitted):
            self.URM = URM

            utils = Utils()

            self.ICM = utils.get_icm()
            print(self.URM.shape, self.ICM.transpose().shape)
            self.URM = sps.vstack([self.URM, self.ICM.transpose()]).tocsr()

            RP3_beta = RP3betaRecommender()
            RP3_beta.fit(self.URM, normalize_similarity=True)

            self.SM_cf = self.create_similarity_matrix(self.URM, 15, 19, similarity="tanimoto")
            self.SM_RP3beta = RP3_beta.get_W()

            self.fitted = True

        self.SM = self.alpha * self.SM_cf + (1-self.alpha) * self.SM_RP3beta

        self.RECS = self.URM.dot(self.SM)


    def create_similarity_matrix(self, URM, knn, shrink, similarity="cosine"):
        similarity_object = Compute_Similarity_Python(URM, topK=knn, shrink=shrink, normalize=True,
                                                      similarity=similarity)
        return similarity_object.compute_similarity()

    def recommend(self, user_id, at=10):
        expected_ratings = self.get_expected_ratings(user_id)

        recommended_items = np.flip(np.argsort(expected_ratings), 0)

        unseen_items_mask = np.in1d(recommended_items, self.URM[user_id].indices,
                                    assume_unique=True, invert=True)
        recommended_items = recommended_items[unseen_items_mask]
        return recommended_items[0:at]

    def get_expected_ratings(self, user_id, normalized_ratings=False):
        expected_ratings = self.RECS[user_id].todense()
        expected_ratings = np.squeeze(np.asarray(expected_ratings))


        # Normalize ratings
        if normalized_ratings and np.amax(expected_ratings) > 0:
            expected_ratings = expected_ratings / np.linalg.norm(expected_ratings)

        return expected_ratings

if __name__ == '__main__':
    recommender = HybridItemCfRP3Beta()
    Runner.run(recommender, True, evaluate_different_type_of_users=True, find_weights_hybrid_item=False, batch_evaluation=True, split='random')
