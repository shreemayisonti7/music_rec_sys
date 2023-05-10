import time

from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql.window import Window
import pyspark.sql.functions as F

def main(spark):
    start = time.time()
    #train_data = spark.read.parquet(f'hdfs:/user/ss16270_nyu_edu/als_train_set.parquet')
    val_data = spark.read.parquet(f'hdfs:/user/ss16270_nyu_edu/als_val_set.parquet')
    #test_data = spark.read.parquet(f'hdfs:/user/ss16270_nyu_edu/als_test_set.parquet')

    #This is to handle the case where an item is in val/test but not in train.
    val_data = val_data.select('user_id').distinct()
    val_data = val_data.dropna()
    #test_data = test_data.dropna()

    # als = ALS(maxIter=5, regParam=0.001, rank=10, alpha=50, userCol="user_id", itemCol="rmsid_int", ratingCol="ratings",
    #           coldStartStrategy="drop", implicitPrefs=True)
    # model = als.fit(train_data)
    # model.write().overwrite().save(f'hdfs:/user/ss16270_nyu_edu/als_model')
    # Evaluate the model by computing the RMSE on the val data
    # pred_val = model.transform(val_data)
    # print("Printing model transformed validation data")
    # pred_val.show()
    # evaluator = RegressionEvaluator(metricName="rmse", labelCol="ratings", predictionCol="prediction")
    # rmse_val = evaluator.evaluate(pred_val)
    # print("Root-mean-square val error = " + str(rmse_val))

    # Evaluate the model by computing the RMSE on the test data
    # pred_test = model.transform(test_data)
    # pred_test.show()
    # evaluator = RegressionEvaluator(metricName="rmse", labelCol="ratings", predictionCol="prediction")
    # rmse_test = evaluator.evaluate(pred_test)
    # print("Root-mean-square test error = " + str(rmse_test))

    # Generate top 10 movie recommendations for each user
    print("Loading model")
    model = ALSModel.load(f'hdfs:/user/ss16270_nyu_edu/als_model')

    print("Making recommendations")
    user_recs = model.recommendForUserSubset(val_data,100)

    user_recs.repartition(50,"user_id")

    print("Saving recs")
    user_recs.write.parquet(f'hdfs:/user/ss16270_nyu_edu/val_recs.parquet', mode="overwrite")


    end = time.time()

    print(f"Total time for execution:{end - start}")


if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName('checkpoint').getOrCreate()

    main(spark)
    spark.stop()
