import pandas as pd
from tqdm import tqdm

from challenge2019.utils.evaluator import Evaluator
from challenge2019.utils.utils import Utils




class Runner(object):

    @staticmethod
    def run(is_test=True):
        URM_csv = pd.read_csv("../dataset/data_train.csv")
        util = Utils(URM_csv)
        URM = util.get_urm_from_csv()

        dio = Evaluator()

        if is_test:
            print("Starting testing phase..")

        dio.random_split(URM, URM_csv)

Runner.run()