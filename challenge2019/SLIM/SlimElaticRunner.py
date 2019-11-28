from SLIM.SlimElasticNet import SLIMElasticNetRecommender
from utils.run import Runner

if __name__ == '__main__':
    recommender = SLIMElasticNetRecommender()
    Runner.run(recommender, True)
