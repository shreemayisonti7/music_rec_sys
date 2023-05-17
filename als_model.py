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
    train_data = spark.read.parquet(f'hdfs:/user/ss16270_nyu_edu/als_train_set.parquet')
    val_data = spark.read.parquet(f'hdfs:/user/ss16270_nyu_edu/als_val_set.parquet')
    test_data = spark.read.parquet(f'hdfs:/user/ss16270_nyu_edu/als_test_set.parquet')

    #Group by user_id to get final table for evaluation
    val_data_eval = val_data.groupBy('user_id').agg(F.collect_set('rmsid_int').alias('ground_truth_songs'))
    test_data_eval = test_data.groupBy('user_id').agg(F.collect_set('rmsid_int').alias('ground_truth_songs'))

    val_data_1 = val_data_eval.select("user_id")
    val_data_1 = val_data_1.repartition(50, "user_id")

    test_data_1 = test_data_eval.select("user_id")
    test_data_1 = test_data_1.repartition(50, "user_id")

    #Hyperparameter tuning over values
    reg_val = [0.001, 0.01, 0.1]
    rank_val = [5, 10, 15, 20, 25, 30]
    alpha_val = [5,10,25,50]

    #Looping through rank values for optimal rank
    best_map = float('inf')
    best_mrr = 0
    best_rank = 0
    for i in range(len(rank_val)):
        print("Tuning rank")
        print(f'Param_val:{rank_val[i]}')
        als = ALS(maxIter=10, regParam=0.0001, rank=rank_val[i], alpha=2, userCol="user_id", itemCol="rmsid_int", ratingCol="ratings",
               coldStartStrategy="drop", implicitPrefs=True)
        model = als.fit(train_data)

        #making recommendations
        user_recs_v = model.recommendForUserSubset(val_data_1, 100)
        user_recs_v = user_recs_v.withColumn("recommendations", col("recommendations").getField("rmsid_int"))
        user_final_v = val_data.join(user_recs_v, on="user_id", how="left")
        map_val, mrr_val = evaluator(user_final_v)
        print(f'MAP:{map_val}, MRR:{mrr_val}')
        if map_val<= best_map:
            best_map = map_val
            best_mrr = mrr_val
            best_rank = rank_val[i]
            model.write().overwrite().save(f'hdfs:/user/ss16270_nyu_edu/als_model')

    #Looping through alpha values for optimal value
    best_alpha = 0
    for i in range(len(alpha_val)):
        print("Tuning slphs")
        print(f'Param_val:{alpha_val[i]}')
        als = ALS(maxIter=10, regParam=0.0001, rank=best_rank, alpha=alpha_val[i], userCol="user_id", itemCol="rmsid_int",
                  ratingCol="ratings",
                  coldStartStrategy="drop", implicitPrefs=True)
        model = als.fit(train_data)

        # making recommendations
        user_recs_v = model.recommendForUserSubset(val_data_1, 100)
        user_recs_v = user_recs_v.withColumn("recommendations", col("recommendations").getField("rmsid_int"))
        user_final_v = val_data.join(user_recs_v, on="user_id", how="left")
        map_val, mrr_val = evaluator(user_final_v)
        print(f'MAP:{map_val}, MRR:{mrr_val}')
        if map_val <= best_map:
            best_map = map_val
            best_mrr = mrr_val
            best_alpha = alpha_val[i]
            model.write().overwrite().save(f'hdfs:/user/ss16270_nyu_edu/als_model')

    #Looping through values of regularization parameters
    best_reg = 0
    for i in range(len(reg_val)):
        print("Tuning reg val")
        print(f'Param_val:{reg_val[i]}')
        als = ALS(maxIter=10, regParam=reg_val[i], rank=best_rank, alpha=best_alpha, userCol="user_id",
                  itemCol="rmsid_int",
                  ratingCol="ratings",
                  coldStartStrategy="drop", implicitPrefs=True)
        model = als.fit(train_data)

        # making recommendations
        user_recs_v = model.recommendForUserSubset(val_data_1, 100)
        user_recs_v = user_recs_v.withColumn("recommendations", col("recommendations").getField("rmsid_int"))
        user_final_v = val_data.join(user_recs_v, on="user_id", how="left")
        map_val, mrr_val = evaluator(user_final_v)
        print(f'MAP:{map_val}, MRR:{mrr_val}')
        if map_val <= best_map:
            best_map_r = map_val
            best_mrr = mrr_val
            best_reg = reg_val[i]
            model.write().overwrite().save(f'hdfs:/user/ss16270_nyu_edu/als_model')

    #Load the best model
    model = ALSModel.load(f'hdfs:/user/ss16270_nyu_edu/als_model')

    #Transforming test data and making predictions
    test_data_1 = test_data_1.repartition(50, "user_id")
    user_recs = model.recommendForUserSubset(test_data_1,100)

    user_recs = user_recs.repartition(50,"user_id")
    user_recs = user_recs.withColumn("recommendations", col("recommendations").getField("rmsid_int"))

    user_final = test_data.join(user_recs,on="user_id",how="left")
    user_final = user_final.repartition(50,"user_id")

    map_val, mrr_val = evaluator(user_final)
    print(f"MAP is:{map_val}, MRR is:{mrr_val}")
    end = time.time()

    print(f"Total time for execution:{end - start}")


if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName('checkpoint').getOrCreate()

    main(spark)
    spark.stop()
