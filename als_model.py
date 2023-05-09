import os
import time

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import desc, row_number, monotonically_increasing_id
from pyspark.sql.window import Window
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml import Pipeline


def main(spark):
    # train hdfs:/user/ss16270_nyu_edu/als_train_set.parquet
    # val hdfs:/user/ss16270_nyu_edu/als_val_set.parquet
    start = time.time()
    train_data = spark.read.parquet(f'hdfs:/user/ss16270_nyu_edu/als_train_set.parquet')
    val_data = spark.read.parquet(f'hdfs:/user/ss16270_nyu_edu/als_val_set.parquet')
    test_data = spark.read.parquet(f'hdfs:/user/ss16270_nyu_edu/als_test_set.parquet ')

    als = ALS(maxIter=5, regParam=0.001, rank=10, alpha=50, userCol="user_id", itemCol="rmsid_int", ratingCol="ratings",
              coldStartStrategy="drop", implicitPrefs=True)
    model = als.fit(train_data)

    # Evaluate the model by computing the RMSE on the val data
    pred_val = model.transform(val_data)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="ratings", predictionCol="prediction")
    rmse_val = evaluator.evaluate(pred_val)
    print("Root-mean-square error = " + str(rmse_val))

    # Evaluate the model by computing the RMSE on the test data
    pred_test = model.transform(test_data)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="ratings", predictionCol="prediction")
    rmse_test = evaluator.evaluate(pred_test)
    print("Root-mean-square error = " + str(rmse_test))

    # Generate top 10 movie recommendations for each user
    userRecs = model.recommendForAllUsers(100)
    userRecs.write.parquet(f'hdfs:/user/ss16270_nyu_edu/best_recs.parquet', mode="overwrite")

    end = time.time()

    print(f"Total time for execution:{end - start}")


if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName('checkpoint').getOrCreate()

    main(spark, userID)
    spark.stop()
