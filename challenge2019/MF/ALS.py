import os

import numpy as np
import implicit

from challenge2019.utils.run import Runner
from challenge2019.utils.utils import Utils


class AlternatingLeastSquare:
    """
    ALS implemented with implicit following guideline of
    https://medium.com/radon-dev/als-implicit-collaborative-filtering-5ed653ba39fe

    IDEA:
    Recomputing x_{u} and y_i can be done with Stochastic Gradient Descent, but this is a non-convex optimization problem.
    We can convert it into a set of quadratic problems, by keeping either x_u or y_i fixed while optimizing the other.
    In that case, we can iteratively solve x and y by alternating between them until the algorithm converges.
    This is Alternating Least Squares.

    """

    def __init__(self):
        self.n_factors = None
        self.regularization = None
        self.iterations = None

    def fit(self, URM, n_factors=339, regularization=0.00187, iterations=50, alpha=24):
        self.URM = URM

        utils = Utils()
        # self.URM = utils.get_URM_BM_25(self.URM)
        self.URM = utils.get_URM_tfidf(self.URM)  # good

        self.n_factors = n_factors
        self.regularization = regularization
        self.iterations = iterations

        sparse_item_user = self.URM.T
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        # Initialize the als model and fit it using the sparse item-user matrix
        model = implicit.als.AlternatingLeastSquares(factors=self.n_factors, regularization=self.regularization,
                                                     iterations=self.iterations, use_gpu=False,
                                                     calculate_training_loss=True, use_cg=True)

        alpha_val = alpha
        # Calculate the confidence by multiplying it by our alpha value.

        data_conf = (sparse_item_user * alpha_val).astype('double')

        # Fit the model
        model.fit(data_conf, show_progress=True)

        # Get the user and item vectors from our trained model
        self.user_factors = model.user_factors
        self.item_factors = model.item_factors

    def get_expected_ratings(self, user_id):
        scores = np.dot(self.user_factors[user_id], self.item_factors.T)

        return np.squeeze(scores)

    def recommend(self, user_id, at=10):
        user_id = int(user_id)
        expected_ratings = self.get_expected_ratings(user_id)

        recommended_items = np.flip(np.argsort(expected_ratings), 0)

        unseen_items_mask = np.in1d(recommended_items, self.URM[user_id].indices,
                                    assume_unique=True, invert=True)
        recommended_items = recommended_items[unseen_items_mask]
        return recommended_items[0:at]


if __name__ == '__main__':
    recommender = AlternatingLeastSquare()
    Runner.run(recommender, True, find_hyper_parameters_ALS=True, evaluate_different_type_of_users=False,
               batch_evaluation=False, split='2080')

# 0.02331 with seed 69
