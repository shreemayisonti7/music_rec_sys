#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.mllib.evaluation import RankingMetrics

import time


def baseline_evaluation(spark, baseline_predictions, test_set):
    udf_metrics = F.udf(lambda ground_truth_songs: RankingMetrics(
        spark.sparkContext.parallelize([(baseline_predictions, ground_truth_songs)])).meanAveragePrecision)
    test_set = test_set.groupBy('user_id').agg(F.collect_list('recording_msid').alias('ground_truth_songs'))
    test_set.show()
    test_set = test_set.withColumn('average_precision', udf_metrics(F.col('ground_truth_songs')))
    test_set.show()
    # print("Counting distinct user_id in val set")
    # test_set.agg(F.countDistinct('user_id')).show()
    # users = test_set.select('user_id').distinct().rdd.flatMap(lambda x: x).collect()
    # print(len(users))
    # map_list = []
    # for user in users:
    #     current_user_rmsids = test_set.filter(test_set.user_id == user).select('recording_msid').distinct().rdd.flatMap(
    #         lambda x: x).collect()
    #     prediction_Labels = spark.sparkContext.parallelize([(baseline_predictions, current_user_rmsids)])
    #     metrics = RankingMetrics(prediction_Labels)
    #     map_list.append(metrics.meanAveragePrecision)
    # MAP_final = sum(map_list)/len(users)
    # print("Final MAP:", MAP_final)
    # MAP_final
    return


def main(spark, userID):
    '''
    Parameters
    spark : SparkSession object
    userID : string, userID to find files in HDFS
    '''
    # #################################################################################################################
    # Baseline model
    start = time.time()
    train_file = f'hdfs:/user/ss16270_nyu_edu/train_small.parquet'
    val_file = f'hdfs:/user/ss16270_nyu_edu/val_small.parquet'

    train_set = spark.read.parquet(train_file)
    train_set = train_set.repartition("recording_msid")
    train_set.createOrReplaceTempView('train_set')

    val_set = spark.read.parquet(val_file)
    val_set = val_set.repartition("user_id")

    # beta_g = [0.01, 0.1, 1, 10, 100, 1000, 10000]
    # beta_i = [0.01, 0.1, 1, 10, 100, 1000, 10000]
    beta_g = [1000]
    beta_i = [1000]

    for i in range(len(beta_g)):
        mu = 1/(1+beta_g[i])

        grouped_result = train_set.groupBy('recording_msid').agg(F.count('user_id').alias('listens'), F.countDistinct(
            'user_id').alias('num_users'))

        baseline_output = grouped_result.orderBy((F.col('listens') - mu) / (F.col('num_users') + beta_i[i]),
                                                                            ascending=False).limit(100)
        baseline_output.show()
        prediction = baseline_output.select('recording_msid').rdd.flatMap(lambda x: x).collect()
        print("Printing top 100 recording msids")
        print(prediction)

        baseline_evaluation(spark, prediction, val_set)
        # print(map, beta_g[i], beta_i[i])

    ###################################################################################################################
    # evaluation
    # val_f = val_set.groupby('user_id').agg(F.collect_set('recording_msid').alias('unique_recordings'))
    # print("printing validation recording ids' lists for various users")
    # val_f.show()
    # val_f.createOrReplaceTempView('val_f')
    # ground_truth = val_f.select('unique_recordings').rdd.flatMap(lambda x: x).collect()
    # print("Ground truth")
    # print(ground_truth)

    end = time.time()
    print(f"Total time for evaluation:{end - start}")
    print("The end")


# Only enter this block if we're in main
if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName('checkpoint').getOrCreate()

    # Get user userID from the command line to access HDFS folder
    userID = os.environ['USER']
    print(userID)

    # Calling main
    main(spark, userID)

    spark.stop()


