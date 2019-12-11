from challenge2019.MF.ALS import AlternatingLeastSquare
from challenge2019.MF.pureSVD import PureSVDRecommender
from challenge2019.SLIM.SlimElasticNet import SLIMElasticNetRecommender
from challenge2019.cf.user_cf import *
from challenge2019.cf.item_cf import *
from challenge2019.cbf.item_cbf import *
from challenge2019.SLIM.SLIM_BPR_Cython import *
from challenge2019.topPop.topPop import *
from challenge2019.cbf.user_cbf import *
from challenge2019.SLIM.SlimElasticNet import *
from challenge2019.hybrid.hybrid_item_cf_P3alpha import HybridItemCfP3alpha
from challenge2019.hybrid.hybrid_item_cf_RP3beta import HybridItemCfRP3Beta
from challenge2019.hybrid.hybrid import Hybrid
from challenge2019.topPop.topPop_userClasses import TopPopUserClasses


class RunAllScripts():
    def run_all_scripts(self):
        utils = Utils()

        recommenders = [ItemCollaborativeFiltering(), ItemContentBasedFiltering(), SLIM_BPR_Cython(), TopPop(),
                        HybridItemCfRP3Beta(), HybridItemCfP3alpha(), UserContentBasedFiltering(),
                        UserCollaborativeFiltering(), Hybrid()]

        seeds = [1234, 9754, 7786, 1328, 6190]

        res = {}
        for seed in seeds:
            URM = utils.get_urm_from_csv()
            evaluator = Evaluator()
            evaluator.random_split_to_all_users(URM, seed)
            import matplotlib.pyplot as plt
            for recommender in recommenders:

                MAP, MAP_User = evaluator.fit_and_evaluate_recommender_on_different_length_of_user(recommender)
                res[recommender.__class__] = MAP_User
                plt.plot(MAP_User, label=recommender.__class__)


            plt.legend(loc=(1.04, 0))
            plt.rcParams['figure.dpi'] = 1000
            plt.show()


if __name__ == '__main__':
    r = RunAllScripts()
    r.run_all_scripts()