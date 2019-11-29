import pandas as pd
from tqdm import tqdm

from challenge2019.utils.evaluator import Evaluator
from challenge2019.utils.utils import Utils


class Runner(object):

    @staticmethod
    def run(recommender, is_test=True, find_hyper_parameters_cf=False, find_hyper_parameters_item_cbf=False,
            find_hyper_parameters_user_cbf=False, evaluate_cold_users=False, find_hyper_parameters_slim_elastic=False,
            find_hyper_parameters_slim_bpr=False, evaluate_different_type_of_users=False):
        # URM_csv = pd.read_csv("../dataset/data_train.csv")
        utils = Utils()
        URM = utils.get_urm_from_csv()
        # TODO: see if this line changes something

        if is_test:
            print("Starting testing phase..")
            evaluator = Evaluator()

            evaluator.random_split_to_all_users(URM, None)

            if find_hyper_parameters_cf:
                tuning_params = {
                    "knn": (100, 1000),
                    "shrink": (10, 1000)
                }
                evaluator.set_recommender_to_tune(recommender)
                evaluator.optimize_bo(tuning_params, evaluator.optimize_hyperparameters_bo_cf)
                #print("MAP@10 : {}".format(evaluator.find_hyper_parameters_cf(recommender)))

            elif find_hyper_parameters_item_cbf:
                tuning_params = {
                    "shrink": (0, 30),
                    "knn_sub_class": (50, 1000),
                    "knn_price": (50, 1000),
                    "knn_asset": (50, 1000)
                }
                evaluator.set_recommender_to_tune(recommender)
                evaluator.optimize_bo(tuning_params, evaluator.optimize_hyperparameters_bo_item_cbf)
                #print("MAP@10 : {}".format(evaluator.find_weight_item_cbf(recommender)))
            elif find_hyper_parameters_user_cbf:
                tuning_params = {
                    "shrink": (0, 30),
                    "knn_age": (50, 1000),
                    "knn_region": (50, 1000),
                }
                evaluator.set_recommender_to_tune(recommender)
                evaluator.optimize_bo(tuning_params, evaluator.optimize_hyperparameters_bo_user_cbf)
                #print("MAP@10 : {}".format((evaluator.find_hyper_parameters_user_cbf(recommender))))
            elif evaluate_cold_users:
                print("MAP@10 : {}".format((evaluator.eval_recommender_cold_users(recommender))))
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
                print("MAP@10 : {}".format((evaluator.fit_and_evaluate_recommender_on_different_type_of_user(recommender))))
            elif find_hyper_parameters_slim_elastic:
                tuning_params = {
                    "topK": (100, 100),
                    "alpha": (1e-2, 1e-4),
                    "l1_ratio": (0.2, 0.05),
                    "tol": (1e-4, 1e-6)
                }
                evaluator.set_recommender_to_tune(recommender)
                evaluator.optimize_bo(tuning_params, evaluator.optimize_hyperparameters_bo_SLIM_el)
            else:
                print("MAP@10 : {}".format(evaluator.fit_and_evaluate_recommender(recommender)))


        else:

            recommender.fit(URM)
            submission_file = open('../dataset/submission.csv', 'w')
            print("Starting recommend ")
            submission_file.write('user_id,item_list\n')
            for user in tqdm(utils.get_target_user_list()):
                user_recommendations = recommender.recommend(user)
                user_recommendations = " ".join(str(x) for x in user_recommendations)
                submission_file.write(str(user) + ',' + user_recommendations + '\n')
            submission_file.close()
