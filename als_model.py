import time

from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql.window import Window
import pyspark.sql.functions as F

def main(spark):
    start = time.time()
    #train_data = spark.read.parquet(f'hdfs:/user/ss16270_nyu_edu/als_train_set.parquet')
    #val_data = spark.read.parquet(f'hdfs:/user/ss16270_nyu_edu/als_val_set.parquet')
    #test_data = spark.read.parquet(f'hdfs:/user/ss16270_nyu_edu/als_test_set.parquet')

    #This is to handle the case where an item is in val/test but not in train.
    #val_data = val_data.dropna()
    #val_data = val_data.groupBy('user_id').agg(F.collect_set('rmsid_int').alias('ground_truth_songs'))
    val_data = spark.read.parquet(f'hdfs:/user/ss16270_nyu_edu/val_data_eval.parquet')

    val_data_1 = val_data.select("user_id")
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
    user_recs = model.recommendForUserSubset(val_data_1,100)

    #user_recs.repartition(50,"user_id")
    print("Mapping")
    user_recs = user_recs.rdd.map(lambda x: (x[0],[list(i)[0] for i in x[1]]))

    print("Converting to DF")
    #user_recs = user_recs.repartition(50, "user_id")
    user_f = user_recs.toDF(["user_id","recs"])
    user_f.write.parquet(f'hdfs:/user/ss16270_nyu_edu/val_recs_als.parquet', mode="overwrite")

    #val_data.repartition(50,"user_id")

    print("Joining")
    user_final = val_data.join(user_f,on="user_id",how="left")

    #user_final.repartition(50,"user_id")
    user_final.write.parquet(f'hdfs:/user/ss16270_nyu_edu/val_eval_f.parquet', mode="overwrite")

    end = time.time()

    print(f"Total time for execution:{end - start}")


if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName('checkpoint').getOrCreate()

    main(spark)
    spark.stop()
