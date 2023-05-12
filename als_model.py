import time

from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql.window import Window
import pyspark.sql.functions as F


def average_precision_calculator(pred_songs, true_songs):
    pred_songs_1 = set_to_list(pred_songs)
    if len(true_songs) <= 0 or len(pred_songs_1) <= 0:
        return 0
    cumulative_average_precision = 0
    positives = 0
    for i in range(len(pred_songs_1)):
        if pred_songs_1[i] in true_songs:
            positives += 1
            cumulative_average_precision += positives / (i + 1)
    if positives == 0:
        return 0
    return cumulative_average_precision / positives

# def reciprocal_rank_calculator(pred_songs, true_songs):
#     if len(true_songs) <= 0 or len(pred_songs) <= 0:
#         return 0
#     for i in range(len(pred_songs)):
#         if pred_songs[i] in true_songs:
#             return 1 / (i + 1)
#     return 0

def set_to_list(predicted_recs):
    new_recs = []
    for i in predicted_recs:
        new_i = list(i)
        new_recs.append(new_i[0])

    return new_recs

def evaluator(test_set):
    udf_ap = F.udf(lambda recs,ground_truth_songs:
                   average_precision_calculator(recs, ground_truth_songs))
    test_set = test_set.withColumn('average_precision', udf_ap(F.col('recs','ground_truth_songs')))

    # udf_rr = F.udf(lambda recs,ground_truth_songs:
    #                reciprocal_rank_calculator(recs, ground_truth_songs))
    #
    # test_set = test_set.withColumn('reciprocal_rank', udf_rr(F.col('recs','ground_truth_songs')))

    mean_average_precision = test_set.agg(F.mean(F.col("average_precision")).alias("mean_average_precision")
                                          ).collect()[0]['mean_average_precision']
    # mean_reciprocal_rank = test_set.agg(F.mean(F.col("reciprocal_rank")).alias("mean_reciprocal_rank")
    #                                     ).collect()[0]['mean_reciprocal_rank']
    return mean_average_precision

def main(spark):
    start = time.time()
    #train_data = spark.read.parquet(f'hdfs:/user/ss16270_nyu_edu/als_train_set.parquet')
    #val_data = spark.read.parquet(f'hdfs:/user/ss16270_nyu_edu/als_val_set.parquet')
    #test_data = spark.read.parquet(f'hdfs:/user/ss16270_nyu_edu/als_test_set.parquet')

    #This is to handle the case where an item is in val/test but not in train.
    #val_data = val_data.dropna()
    #val_data = val_data.groupBy('user_id').agg(F.collect_set('rmsid_int').alias('ground_truth_songs'))
    val_data = spark.read.parquet(f'hdfs:/user/ss16270_nyu_edu/val_data_eval.parquet')

    val_data_1 = val_data.select("user_id").limit(1)
    #test_data = test_data.dropna()

    # als = ALS(maxIter=5, regParam=0.01, rank=15, alpha=10, userCol="user_id", itemCol="rmsid_int", ratingCol="ratings",
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
    #
    print("Making recommendations")
    user_recs = model.recommendForUserSubset(val_data_1,100)
    #
    # print("Converting to DF")
    print("Printing user")
    print(user_recs.take(1))
    # user_f = user_recs.toDF("user_id","recs")
    # user_recs_1 = user_f.repartition(50, "user_id")
    #
    # val_data_1 = val_data.repartition(50, "user_id")
    #
    # print("Joining")
    # user_final = val_data_1.join(user_recs_1,on="user_id",how="left")
    # user_final_1 = user_final.repartition(50, "user_id")
    #
    # #user_final.repartition(50,"user_id")
    # user_final_1.write.parquet(f'hdfs:/user/ss16270_nyu_edu/val_eval_f.parquet', mode="overwrite")
    #
    # current_map= evaluator(user_final_1)
    # print(f"MAP:{current_map}")

    end = time.time()

    print(f"Total time for execution:{end - start}")


if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName('checkpoint').getOrCreate()

    main(spark)
    spark.stop()
