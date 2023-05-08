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
    best_model = spark.read.parquet(f'hdfs:/user/ss16270_nyu_edu/best_recs.parquet')

    best_model.show()
    # val_data = spark.read.parquet(f'hdfs:/user/ss16270_nyu_edu/val_full_als.parquet')
    #
    # val_data_f = val_data.groupBy('user_id').agg(F.collect_list('recording_index').alias('ground_truth_songs'))
    #
    # final_data = val_data.join

if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName('checkpoint').getOrCreate()

    # Get user userID from the command line to access HDFS folder
    userID = os.environ['USER']
    print(userID)

    # Calling main
    main(spark, userID)





