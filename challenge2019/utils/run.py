import pandas as pd
from tqdm import tqdm
import numpy as np

from challenge2019.utils.evaluator import Evaluator
from challenge2019.utils.utils import Utils


class Runner(object):

    @staticmethod
    def run(recommender, is_test=True, find_hyper_parameters_cf=False, find_hyper_parameters_item_cbf=False,
            find_hyper_parameters_user_cbf=False, find_epochs=False, find_hyper_parameters_slim_elastic=False,
            find_hyper_parameters_slim_bpr=False, evaluate_different_type_of_users=False,
            evaluate_different_age_of_users=False, evaluate_different_region_of_users=False, find_weights_hybrid=False,
            find_hyper_parameters_P3alpha=False, find_hyper_parameters_pureSVD=False, find_weights_hybrid_item = False,
            batch_evaluation=False, find_hyper_parameters_RP3beta=False, find_hyper_parameters_ALS=False,
            loo_split=False):
        # URM_csv = pd.read_csv("../dataset/data_train.csv")
        utils = Utils()
        # TODO: see if this line changes something

        if is_test:
            print("Starting testing phase..")

            if batch_evaluation:
                seeds = [69, 420, 666, 777, 619]
            else:
                seeds = [None]

            for seed in seeds:
                print("Seed: {}".format(seed))
                evaluator = None
                URM = utils.get_urm_from_csv()
                evaluator = Evaluator()

                if loo_split:
                    evaluator.leave_one_out(URM, seed)
                else:
                    evaluator.random_split(URM, seed)

                if find_hyper_parameters_cf:
                    tuning_params = {
                        "knn": (100, 1000),
                        "shrink": (10, 1000)
                    }
                    evaluator.set_recommender_to_tune(recommender)
                    evaluator.optimize_bo(tuning_params, evaluator.optimize_hyperparameters_bo_cf)

                elif find_hyper_parameters_item_cbf:
                    tuning_params = {
                        "shrink": (0, 30),
                        "knn_sub_class": (50, 1000),
                        "knn_price": (50, 1000),
                        "knn_asset": (50, 1000)
                    }
                    evaluator.set_recommender_to_tune(recommender)
                    evaluator.optimize_bo(tuning_params, evaluator.optimize_hyperparameters_bo_item_cbf)

                elif find_weights_hybrid_item:
                    tuning_params = {
                        "alpha": (0, 1),
                    }
                    evaluator.set_recommender_to_tune(recommender)
                    evaluator.optimize_bo(tuning_params, evaluator.optimize_weights_hybrid_item)

                elif find_hyper_parameters_P3alpha:
                    tuning_params = {
                        "topk": (1, 200),
                        "alpha": (0.01, 1)
                    }
                    evaluator.set_recommender_to_tune(recommender)
                    evaluator.optimize_bo(tuning_params, evaluator.optimize_hyperparameters_bo_P3alpha)

                elif find_hyper_parameters_RP3beta:
                    tuning_params = {
                        "topk": (1, 200),
                        "alpha": (0.01, 1),
                        "beta": (0.01, 1)
                    }
                    evaluator.set_recommender_to_tune(recommender)
                    evaluator.optimize_bo(tuning_params, evaluator.optimize_hyperparameters_bo_RP3beta)
                elif find_weights_hybrid:

                    weights = {
                        "SLIM_E": (0.8, 1.1),
                        "item_cf": (0.8, 1.5),
                        "user_cf": (0, 0.01),
                        # "MF": (0, 1),
                        # "user_cbf": (0, 1)
                    }
                    evaluator.set_recommender_to_tune(recommender)
                    evaluator.optimize_bo(weights, evaluator.optimize_weights_hybrid)
                    # print("MAP@10 : {}".format(evaluator.find_weight_item_cbf(recommender)))

                elif find_hyper_parameters_user_cbf:
                    tuning_params = {
                        "shrink": (0, 30),
                        "knn_age": (50, 1000),
                        "knn_region": (50, 1000),
                    }
                    evaluator.set_recommender_to_tune(recommender)
                    evaluator.optimize_bo(tuning_params, evaluator.optimize_hyperparameters_bo_user_cbf)

                elif find_hyper_parameters_pureSVD:
                    tuning_params = {
                        "num_factors": (500, 1000)
                    }
                    evaluator.set_recommender_to_tune(recommender)
                    evaluator.optimize_bo(tuning_params, evaluator.optimize_hyperparameters_bo_pure_svd)

                elif find_hyper_parameters_ALS:
                    tuning_params = {
                        "n_factors": (400, 800),
                        "regularization": (0.01, 0.1),
                        "iterations" : (45, 100)
                    }
                    evaluator.set_recommender_to_tune(recommender)
                    evaluator.optimize_bo(tuning_params, evaluator.optimize_hyperparameters_bo_ALS)

                elif find_hyper_parameters_slim_bpr:

                    tuning_params = {
                        "topK": (100, 1),
                        "learning_rate": (1e-3, 1e-7),
                        "li_reg": (0.1, 0.0001),
                        "lj_reg": (0.1, 0.0001)
                    }

                    evaluator.set_recommender_to_tune(recommender)
                    evaluator.optimize_bo(tuning_params, evaluator.optimize_hyperparameters_bo_SLIM_bpr)
                elif evaluate_different_type_of_users:
                    print("MAP@10 : {}".format((evaluator.fit_and_evaluate_recommender_on_different_length_of_user(recommender))))
                elif evaluate_different_age_of_users:
                    print("MAP@10 : {}".format((evaluator.fit_and_evaluate_recommender_on_different_age_of_user(recommender))))
                elif evaluate_different_region_of_users:
                    print("MAP@10 : {}".format(
                        (evaluator.fit_and_evaluate_recommender_on_different_region_of_user(recommender))))
                elif find_hyper_parameters_slim_elastic:
                    tuning_params = {
                        "topK": (100, 100),
                        "alpha": (1e-2, 1e-4),
                        "l1_ratio": (0.2, 0.05),
                        "tol": (1e-4, 1e-6)
                    }
                    evaluator.set_recommender_to_tune(recommender)
                    evaluator.optimize_bo(tuning_params, evaluator.optimize_hyperparameters_bo_SLIM_el)
                elif find_epochs:
                    print("MAP@10 : {}".format(evaluator.find_epochs(recommender)))
                else:
                    print("MAP@10 : {}".format(evaluator.fit_and_evaluate_recommender(recommender)))


        else:
            URM = utils.get_urm_from_csv()
            recommender.fit(URM)
            submission_file = open('../dataset/submission.csv', 'w')
            print("Starting recommend ")
            submission_file.write('user_id,item_list\n')
            for user in tqdm(utils.get_target_user_list()):
                user_recommendations = recommender.recommend(user)
                user_recommendations = " ".join(str(x) for x in user_recommendations)
                submission_file.write(str(user) + ',' + user_recommendations + '\n')
            submission_file.close()
