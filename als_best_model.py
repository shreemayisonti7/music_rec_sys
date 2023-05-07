import os
import time

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import desc, row_number, monotonically_increasing_id
from pyspark.sql.window import Window
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql import Row
from pyspark.ml.tuning import ParamGridBuilder,CrossValidator
from pyspark.ml import Pipeline

def main(spark, userID):
    print("---------------------------Converting recording_msids to integer for train---------------------------------")
    best_model = ALSModel.load(f'hdfs:/user/ss16270_nyu_edu/als_model')
    print(f"Rank: {best_model.rank}")
    print(f"Alpha:{best_model.getRegParam()}")
    print(f"Reg param:{best_model.getAlpha()}")





if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName('checkpoint').getOrCreate()

    # Get user userID from the command line to access HDFS folder
    userID = os.environ['USER']
    print(userID)

    # Calling main
    main(spark, userID)





