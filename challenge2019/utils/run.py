import pandas as pd
from tqdm import tqdm

from .evaluator import Evaluator
from .utils import Utils




class Runner(object):

    @staticmethod
    def run(recommender, is_test=True):
        URM_csv = pd.read_csv("../dataset/data_train.csv")
        utils = Utils(URM_csv)
        URM = utils.get_urm_from_csv()

        evaluator = Evaluator()
        evaluator.random_split(URM, URM_csv)

        if is_test:
            print("Starting testing phase..")
            print("MAP@10 : {}".format(evaluator.eval_recommender(recommender)))

        else:
            recommender.fit(URM)
            submission_file = open('../dataset/submission.csv','w')

            submission_file.write('user_id,item_list\n')
            for user in tqdm(utils.get_target_user_list()):
                user_recommendations = recommender.recommend(user)
                user_recommendations = " ".join(str(x) for x in user_recommendations)
                submission_file.write(str(user) + ',' + user_recommendations + '\n')
            submission_file.close()



