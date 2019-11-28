import multiprocessing
from functools import partial
import numpy as np
import scipy.sparse as sps
from sklearn.linear_model import ElasticNet
from tqdm import tqdm

from challenge2019.utils.run import Runner


class SLIMElasticNetRecommender(object):
    """
    Train a Sparse Linear Methods (SLIM) item similarity model.
    NOTE: ElasticNet solver is parallel, a single intance of SLIM_ElasticNet will
          make use of half the cores available

    See:
        Efficient Top-N Recommendation by Linear Regression,
        M. Levy and K. Jack, LSRS workshop at RecSys 2013.
        https://www.slideshare.net/MarkLevy/efficient-slides

        SLIM: Sparse linear methods for top-n recommender systems,
        X. Ning and G. Karypis, ICDM 2011.
        http://glaros.dtc.umn.edu/gkhome/fetch/papers/SLIM2011icdm.pdf
    """
    def __init__(self, fit_intercept=False, copy_X=False, precompute=False, selection='random',
                 positive_only=True, workers=multiprocessing.cpu_count()):

        self.analyzed_items = 0
        self.alpha = None
        self.l1_ratio = None
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.precompute = precompute
        self.selection = selection
        self.max_iter = None
        self.tol = None
        self.topK = None
        self.positive_only = positive_only
        self.workers = workers

    """ 
        Fit given to each pool thread, to fit the W_sparse 
    """
    def _partial_fit(self, currentItem, X, iterations=1):

        model = ElasticNet(alpha=self.alpha,
                           l1_ratio=self.l1_ratio,
                           positive=self.positive_only,
                           fit_intercept=self.fit_intercept,
                           copy_X=self.copy_X,
                           precompute=self.precompute,
                           selection=self.selection,
                           max_iter=self.max_iter,
                           tol=self.tol)

        # WARNING: make a copy of X to avoid race conditions on column j
        # TODO: We can probably come up with something better here.
        X_j = X.copy()
        # get the target column
        y = X_j[:, currentItem].toarray()
        # set the j-th column of X to zero
        X_j.data[X_j.indptr[currentItem]:X_j.indptr[currentItem + 1]] = 0.0
        # fit one ElasticNet model per column
        values = []
        rows = []
        cols = []
        if (currentItem % 3000) == 0:
            print(currentItem/iterations * 100)
        if np.max(y) > 0:

            model.fit(X_j, y)

            relevant_items_partition = (-model.coef_).argpartition(self.topK)[0:self.topK]
            relevant_items_partition_sorting = np.argsort(-model.coef_[relevant_items_partition])
            ranking = relevant_items_partition[relevant_items_partition_sorting]

            notZerosMask = model.coef_[ranking] > 0.0
            ranking = ranking[notZerosMask]

            values = model.coef_[ranking]
            rows = ranking
            cols = [currentItem] * len(ranking)

        return values, rows, cols

    def fit(self, URM, max_iter=200, tol=1e-5, topK=100, alpha=1e-3, l1_ratio=0.1):

        self.URM_train = URM
        self.max_iter = max_iter
        self.tol = tol
        self.topK = topK
        self.alpha = alpha
        self.l1_ratio = l1_ratio

        n_items = self.URM_train.shape[1]
        print("Iterating for " + str(n_items) + "times")
        warm_users = list(range(0, n_items))
        # something wrong here

        #create a copy of the URM since each _pfit will modify it

        _pfit = partial(self._partial_fit, X=self.URM_train, iterations=n_items)
        with multiprocessing.Pool(self.workers) as pool:
            res = pool.map(_pfit, warm_users)

        # res contains a vector of (values, rows, cols) tuples
        values, rows, cols = [], [], []
        for values_, rows_, cols_ in tqdm(res):
            values.extend(values_)
            rows.extend(rows_)
            cols.extend(cols_)

        self.W_sparse = sps.csc_matrix((values, (rows, cols)), shape=(n_items, n_items), dtype=np.float32)

    def get_expected_ratings(self, user_id):
        user_id = int(user_id)
        user_profile = self.URM_train[user_id]
        expected_ratings = user_profile.dot(self.W_sparse).toarray().ravel()

        # # EDIT
        return expected_ratings

    def recommend(self, user_id, at=10):
        user_id = int(user_id)
        # compute the scores using the dot product
        scores = self.get_expected_ratings(user_id)
        user_profile = self.URM_train[user_id].indices
        scores[user_profile] = 0

        # rank items
        recommended_items = np.flip(np.argsort(scores), 0)

        return recommended_items[:at]


if __name__ == '__main__':
    recommender = SLIMElasticNetRecommender()
    Runner.run(recommender, True,find_hyper_parameters_slim_elastic=True)
