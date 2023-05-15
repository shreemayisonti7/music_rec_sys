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
    print("---------------------------Tuning hyperparameters---------------------------------")
    train_data = spark.read.parquet(f'hdfs:/user/ss16270_nyu_edu/als_train_set.parquet')
    val_data = spark.read.parquet(f'hdfs:/user/ss16270_nyu_edu/als_val_set.parquet')

    lambda_val = 0.0001
    rank_val = [5,10,15,20,25]
    alpha_val = 2

    for i in range(len(rank_val)):
        als = ALS(maxIter=10, regParam=lambda_val, rank=rank_val[i], alpha= alpha_val, userCol="user_id", itemCol="rmsid_int", ratingCol="ratings",
           coldStartStrategy="drop", implicitPrefs=True)
        model = als.fit(train_data)
        pred_val = model.transform(val_data)
        print("Printing model transformed validation data")
        pred_val.show()
        evaluator = RegressionEvaluator(metricName="rmse", labelCol="ratings", predictionCol="prediction")
        rmse_val = evaluator.evaluate(pred_val)
        print(f'RMSE={rmse_val}, rank:{rank_val[i]}')

if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName('checkpoint').getOrCreate()

    # Get user userID from the command line to access HDFS folder
    userID = os.environ['USER']
    print(userID)

    # Calling main
    main(spark, userID)
    spark.stop()





