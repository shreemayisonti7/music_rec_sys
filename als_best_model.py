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
    print("---------------------------Tuning hyperparameters---------------------------------")
    train_data = spark.read.parquet(f'hdfs:/user/ss16270_nyu_edu/als_train_set.parquet')
    val_data = spark.read.parquet(f'hdfs:/user/ss16270_nyu_edu/als_val_set.parquet')

    # val_data_new = spark.read.parquet(f'hdfs:/user/ss16270_nyu_edu/val_data_eval.parquet')
    # val_data_1 = val_data_new.select("user_id")
    # val_data_1 = val_data_1.repartition(50, "user_id")
    #
    # lambda_val = 0.0001
    # rank_val = [20,25]
    # alpha_val = 2
    #
    # for i in range(len(rank_val)):
    #     als = ALS(maxIter=10, regParam=lambda_val, rank=rank_val[i], alpha= alpha_val, userCol="user_id", itemCol="rmsid_int", ratingCol="ratings",
    #        coldStartStrategy="drop", implicitPrefs=True)
    #     model = als.fit(train_data)
    #     # pred_val = model.transform(val_data)
    #     # evaluator = RegressionEvaluator(metricName="rmse", labelCol="ratings", predictionCol="prediction")
    #     # rmse_val = evaluator.evaluate(pred_val)
    #     # print(f'RMSE={rmse_val}, rank:{rank_val[i]}')
    #
    #     user_recs = model.recommendForUserSubset(val_data_1, 100)
    #     print("Converting recs")
    #     user_recs = user_recs.repartition(50, "user_id")
    #     user_recs = user_recs.withColumn("recommendations", col("recommendations").getField("rmsid_int"))
    #
    #     print("Joining")
    #     user_final = val_data_new.join(user_recs, on="user_id", how="left")
    #     user_final = user_final.repartition(50, "user_id")
    #
    #     map_val, mrr_val = evaluator(user_final)
    #     print(f"MAP is:{map_val}, MRR is:{mrr_val}")




    lambda_val = [0.001, 0.01, 0.05, 0.1, 1]
    rank_val = 25
    alpha_val = 2

    for i in range(len(lambda_val)):
        als = ALS(maxIter=10, regParam=lambda_val[i], rank=rank_val, alpha= alpha_val, userCol="user_id", itemCol="rmsid_int", ratingCol="ratings",
           coldStartStrategy="drop", implicitPrefs=True)
        model = als.fit(train_data)
        pred_val = model.transform(val_data)
        evaluator = RegressionEvaluator(metricName="rmse", labelCol="ratings", predictionCol="prediction")
        rmse_val = evaluator.evaluate(pred_val)
        print(f'RMSE={rmse_val}, rank:{lambda_val[i]}')

if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName('checkpoint').getOrCreate()

    main(spark)
    spark.stop()





