import time

from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql.window import Window
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.mllib.evaluation import RankingMetrics

def average_precision_calculator(pred_songs, true_songs):
    if len(true_songs) <= 0 or len(pred_songs) <= 0:
        return 0
    cumulative_average_precision = 0
    positives = 0
    for i in range(len(pred_songs)):
        if pred_songs[i] in true_songs:
            positives += 1
            cumulative_average_precision += positives / (i + 1)
    if positives == 0:
        return 0
    return cumulative_average_precision / positives

def reciprocal_rank_calculator(pred_songs, true_songs):
    if len(true_songs) <= 0 or len(pred_songs) <= 0:
        return 0
    for i in range(len(pred_songs)):
        if pred_songs[i] in true_songs:
            return 1 / (i + 1)
    return 0


def evaluator(test_set):
    udf_ap = F.udf(lambda recommendations, ground_truth_songs:
                   average_precision_calculator(recommendations, ground_truth_songs))
    test_set = test_set.withColumn('average_precision', udf_ap(F.col('recommendations'),F.col('ground_truth_songs')))

    udf_rr = F.udf(lambda recommendations,ground_truth_songs:
                   reciprocal_rank_calculator(recommendations, ground_truth_songs))

    test_set = test_set.withColumn('reciprocal_rank', udf_rr(F.col('recommendations'),F.col('ground_truth_songs')))

    mean_average_precision = test_set.agg(F.mean(F.col("average_precision")).alias("mean_average_precision")
                                          ).collect()[0]['mean_average_precision']
    mean_reciprocal_rank = test_set.agg(F.mean(F.col("reciprocal_rank")).alias("mean_reciprocal_rank")
                                        ).collect()[0]['mean_reciprocal_rank']
    return mean_average_precision, mean_reciprocal_rank

def main(spark):
    start = time.time()
    #train_data = spark.read.parquet(f'hdfs:/user/ss16270_nyu_edu/als_train_set.parquet')
    #val_data = spark.read.parquet(f'hdfs:/user/ss16270_nyu_edu/als_val_set.parquet')
    test_data = spark.read.parquet(f'hdfs:/user/ss16270_nyu_edu/als_test_set.parquet')

    #This is to handle the case where an item is in val/test but not in train.
    #test_data = test_data.dropna()
    #test_data = test_data.groupBy('user_id').agg(F.collect_set('rmsid_int').alias('ground_truth_songs'))
    #test_data.write.parquet(f'hdfs:/user/ss16270_nyu_edu/test_data_eval.parquet', mode="overwrite")

    #test_data = spark.read.parquet(f'hdfs:/user/ss16270_nyu_edu/test_data_eval.parquet')

    #test_data_1 = test_data.select("user_id")


    # als = ALS(maxIter=10, regParam=0.0001, rank=30, alpha=5, userCol="user_id", itemCol="rmsid_int", ratingCol="ratings",
    #            coldStartStrategy="drop", implicitPrefs=True)
    # model = als.fit(train_data)
    # model.write().overwrite().save(f'hdfs:/user/ss16270_nyu_edu/als_model_r30_l0001_a5_i10')
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

    # # Generate top 10 movie recommendations for each user
    print("Loading model")
    model = ALSModel.load(f'hdfs:/user/ss16270_nyu_edu/als_model_r25_l0001_a2_i10')

    pred_val = model.transform(test_data)
    print("Printing model transformed validation data")
    pred_val.show()
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="ratings", predictionCol="prediction")
    rmse_val = evaluator.evaluate(pred_val)
    print("Root-mean-square val error = " + str(rmse_val))
    #
    # print("Making recommendations")
    # test_data_1 = test_data_1.repartition(50, "user_id")
    # user_recs = model.recommendForUserSubset(test_data_1,100)
    # # #
    # print("Converting recs")
    # user_recs = user_recs.repartition(50,"user_id")
    # user_recs = user_recs.withColumn("recommendations", col("recommendations").getField("rmsid_int"))
    #
    # print("Joining")
    # user_final = test_data.join(user_recs,on="user_id",how="left")
    # user_final = user_final.repartition(50,"user_id")
    #
    # print("Mapping")
    # user_final_1 = user_final.rdd.map(lambda x:(x[1],x[2]))
    #
    # #map_val, mrr_val = evaluator(user_final)
    # print("Metrics")
    # metric = RankingMetrics(user_final_1)
    # print(f"MAP is {metric.meanAveragePrecision}")
    # #print(f"MAP is:{map_val}, MRR is:{mrr_val}")
    end = time.time()

    print(f"Total time for execution:{end - start}")


if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName('checkpoint').getOrCreate()

    main(spark)
    spark.stop()
