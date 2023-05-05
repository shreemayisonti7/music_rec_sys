import os
import time

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import desc, row_number, monotonically_increasing_id
from pyspark.sql.window import Window
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row


def main(spark, userID):
    print("---------------------------Converting recording_msids to integer for train---------------------------------")
    start = time.time()
    train_data = spark.read.parquet(f'hdfs:/user/ss16270_nyu_edu/train_full_als.parquet')
    val_data = spark.read.parquet(f'hdfs:/user/ss16270_nyu_edu/val_full_als.parquet')

    train_data_f = train_data.select("user_id","recording_index")
    grouped_data = train_data_f.groupBy("user_id","recording_index").agg(F.count("recording_index").
                                                                         alias("rec_frequency"))

    val_data_f = val_data.select("user_id","recording_index")
    grouped_data_v = train_data_f.groupBy("user_id", "recording_index").agg(F.count("recording_index").
                                                                          alias("rec_frequency"))

    als = ALS(maxIter=5, regParam=0.01, userCol="user_id", itemCol="recording_index", ratingCol="rec_frequency",
              coldStartStrategy="drop",implicitPrefs=True)
    model = als.fit(training)

    # Evaluate the model by computing the RMSE on the test data
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rec_frequency",
                                    predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print("Root-mean-square error = " + str(rmse))

    # Generate top 10 movie recommendations for each user
    # userRecs = model.recommendForAllUsers(10)
    # # Generate top 10 user recommendations for each movie
    # movieRecs = model.recommendForAllItems(10)
    #
    # # Generate top 10 movie recommendations for a specified set of users
    # users = ratings.select(als.getUserCol()).distinct().limit(3)
    # userSubsetRecs = model.recommendForUserSubset(users, 10)
    # # Generate top 10 user recommendations for a specified set of movies
    # movies = ratings.select(als.getItemCol()).distinct().limit(3)
    # movieSubSetRecs = model.recommendForItemSubset(movies, 10)

    end=time.time()

    print(f"Total time for execution:{end-start}")





if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName('checkpoint').getOrCreate()

    # Get user userID from the command line to access HDFS folder
    userID = os.environ['USER']
    print(userID)

    # Calling main
    main(spark, userID)





