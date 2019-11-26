import pandas as pd
from tqdm import tqdm

from challenge2019.utils.evaluator import Evaluator
from challenge2019.utils.utils import Utils




class Runner(object):

    @staticmethod
    def run(recommender, is_test=True, train_knn_shrink=False, train_cbf_weights=False, early_stopping=False):
        # URM_csv = pd.read_csv("../dataset/data_train.csv")
        utils = Utils()
        URM = utils.get_urm_from_csv()

        if is_test:
            print("Starting testing phase..")
            evaluator = Evaluator()

            evaluator.random_split(URM, None)
            if train_knn_shrink:
                print("MAP@10 : {}".format(evaluator.find_hyper_parameters_cf(recommender)))
            elif train_cbf_weights:
                print("MAP@10 : {}".format(evaluator.find_weight_item_cbf(recommender)))
            else:
                print("MAP@10 : {}".format(evaluator.fit_and_evaluate_recommender(recommender)))


        else:

            recommender.fit(URM)
            submission_file = open('../dataset/submission.csv','w')
            print("Starting recommend ")
            submission_file.write('user_id,item_list\n')
            for user in tqdm(utils.get_target_user_list()):
                user_recommendations = recommender.recommend(user)
                user_recommendations = " ".join(str(x) for x in user_recommendations)
                submission_file.write(str(user) + ',' + user_recommendations + '\n')
            submission_file.close()



