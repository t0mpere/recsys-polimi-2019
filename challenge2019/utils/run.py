import pandas as pd
from tqdm import tqdm
import numpy as np

from challenge2019.utils.evaluator import Evaluator, bcolors
from challenge2019.utils.utils import Utils


class Runner(object):

    """
    Runner is the class responsible for the handling of a recommender object
    :parameter is_test :decides weather to run the test suite or to create a csv for submission
    """
    @staticmethod
    def run(recommender,

            is_test=True,

            #
            # PARAMETER FOR BAYESIAN OPTIMIZATION
            #
            find_hyper_parameters_cf=False,
            find_hyper_parameters_item_cbf=False,
            find_hyper_parameters_user_cbf=False,
            find_hyper_parameters_slim_elastic=False,
            find_weights_hybrid_cold_users=False,
            find_hyper_parameters_slim_bpr=False,
            find_weights_hybrid=False,
            find_hyper_parameters_RP3beta=False,
            find_hyper_parameters_ALS=False,
            find_hyper_parameters_fw=False,
            find_weights_hybrid_all_item=False,
            find_hyper_parameters_P3alpha=False,
            find_hyper_parameters_pureSVD=False,
            find_weights_hybrid_item=False,
            find_weights_new_hybrid=False,
            find_weights_hybrid_20=False,
            find_weights_item_cbf=False,

            #
            # PARAMETERS FOR EVALUATION METHODS
            #
            evaluate_different_type_of_users=False,
            evaluate_different_age_of_users=False,
            evaluate_different_region_of_users=False,
            batch_evaluation=False,

            #
            # SPLIT SELECT
            # 2080: Random selection of 20% of the dataset
            # random: Random selection of 20% of interactions for all users with more than 4
            # random_all: Random selection of 20% of interactions for all users
            #
            split='2080'):
        utils = Utils()

        if is_test:
            print("Starting testing phase..")

            if batch_evaluation:
                seeds = [123, 9754, 7786, 1328, 6190]
            else:
                seeds = [None]

            for seed in seeds:
                print("Seed: {}".format(seed))
                evaluator = None
                URM = utils.get_urm_from_csv()
                evaluator = Evaluator()

                assert split in ['loo', 'random', 'random_all', '2080']
                if split is 'loo':
                    evaluator.leave_one_out(URM, seed)
                elif split is 'random_all':
                    evaluator.random_split_to_all_users(URM, seed)
                elif split is 'random':
                    evaluator.random_split(URM, seed)
                elif split is '2080':
                    evaluator.train_test_holdout(URM, seed)

                if find_hyper_parameters_cf:
                    tuning_params = {
                        "knn": (1, 100),
                        "shrink": (1, 100)
                    }
                    evaluator.set_recommender_to_tune(recommender)
                    evaluator.optimize_bo(tuning_params, evaluator.optimize_long_item_cf)

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
                        "alpha": (0.1, 0.9),
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
                        "topk": (70, 150),
                        "alpha": (0.01, 0.9),
                        "beta": (0.01, 0.9)
                    }
                    evaluator.set_recommender_to_tune(recommender)
                    evaluator.optimize_bo(tuning_params, evaluator.optimize_hyperparameters_bo_RP3beta)

                elif find_weights_hybrid:
                    weights = {
                        "MF": (0, 1),
                        "SLIM_E": (0, 1),
                        "item_cf": (0, 1),
                        "user_cf": (0, 1),
                        "item_cbf": (0, 1)
                    }
                    evaluator.set_recommender_to_tune(recommender)
                    evaluator.optimize_bo(weights, evaluator.optimize_weights_hybrid)

                elif find_weights_new_hybrid:
                    weights = {
                        "MF": (0, 2),
                        "RP3beta": (0, 10),
                        "SLIM_E": (0, 5),
                        "item_cbf": (0, 10),
                        "item_cf": (0, 10),
                        "user_cf": (0, 0.1)
                    }
                    evaluator.set_recommender_to_tune(recommender)
                    evaluator.optimize_bo(weights, evaluator.optimize_weights_new_hybrid)

                elif find_weights_hybrid_20:
                    weights = {
                        "MF": (0, 1),
                        "item": (0, 1),
                        "user_cf": (0.004, 0.007),
                    }
                    evaluator.set_recommender_to_tune(recommender)
                    evaluator.optimize_bo(weights, evaluator.optimize_weights_hybrid_20)

                elif find_weights_hybrid_all_item:
                    weights = {
                        "SLIM_E": (0, 1),
                        "cf": (0, 1),
                        "RP3": (0, 1),
                        "cbf": (0, 1)
                    }
                    evaluator.set_recommender_to_tune(recommender)
                    evaluator.optimize_bo(weights, evaluator.optimize_weights_hybrid_all_item)

                elif find_weights_item_cbf:
                    weights = {
                        "asset": (0, 1),
                        "price": (0, 1),
                        "sub_class": (0, 1)
                    }
                    evaluator.set_recommender_to_tune(recommender)
                    evaluator.optimize_bo(weights, evaluator.optimize_weights_item_cbf)

                elif find_hyper_parameters_user_cbf:
                    tuning_params = {
                        "shrink": (0, 50),
                        "knn": (100, 2000)
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
                        "n_factors": (100, 500),
                        "regularization": (0.001, 0.1),
                        "iterations": (50, 50),
                        "alpha": (5, 50)
                    }
                    evaluator.set_recommender_to_tune(recommender)
                    evaluator.optimize_bo(tuning_params, evaluator.optimize_hyperparameters_bo_ALS)

                elif find_hyper_parameters_fw:
                    tuning_params = {
                        "loss_tolerance": (1e-5, 1e-7),
                        "iteration_limit": (10000, 50000),
                        "damp_coeff": (0, 0.01),
                        "topK": (20,200),
                        "add_zeros_quota": (0, 0.01)
                    }
                    evaluator.set_recommender_to_tune(recommender)
                    evaluator.optimize_bo(tuning_params, evaluator.optimize_hyperparameters_bo_fw)

                elif find_hyper_parameters_slim_bpr:
                    tuning_params = {
                        "topK": (100, 1),
                        "learning_rate": (1e-3, 1e-7),
                        "li_reg": (0.1, 0.0001),
                        "lj_reg": (0.1, 0.0001)
                    }

                    evaluator.set_recommender_to_tune(recommender)
                    evaluator.optimize_bo(tuning_params, evaluator.optimize_hyperparameters_bo_SLIM_bpr)

                elif find_weights_hybrid_cold_users:
                    tuning_params = {
                        "at": (10, 100),
                        "threshold": (0, 15)
                    }

                    evaluator.set_recommender_to_tune(recommender)
                    evaluator.optimize_bo(tuning_params, evaluator.optimize_weights_hybrid_cold_users)

                elif find_hyper_parameters_slim_elastic:
                    tuning_params = {
                        "max_iter": (50, 300),
                        "topK": (50, 200),
                        "alpha": (1e-2, 1e-4),
                        "l1_ratio": (0.2, 0.05),
                        "tol": (1e-4, 1e-6)
                    }
                    evaluator.set_recommender_to_tune(recommender)
                    evaluator.optimize_bo(tuning_params, evaluator.optimize_hyperparameters_bo_SLIM_el)

                elif evaluate_different_type_of_users:
                    MAP, MAP_user = evaluator.evaluate_recommender_on_different_length_of_user(recommender)
                    print("MAP@10 : {}".format(MAP))
                elif evaluate_different_age_of_users:
                    print("MAP@10 : {}".format((evaluator.fit_and_evaluate_recommender_on_different_age_of_user(recommender))))
                elif evaluate_different_region_of_users:
                    print("MAP@10 : {}".format((evaluator.fit_and_evaluate_recommender_on_different_region_of_user(recommender))))
                else:
                    print(bcolors.color("MAP@10 : {}".format(evaluator.fit_and_evaluate_recommender(recommender)), 3))


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
