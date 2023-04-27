#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.mllib.evaluation import RankingMetrics

import time


def baseline_evaluation(spark, baseline_predictions, test_set):
    print("Counting distinct user_id in val set")
    test_set.agg(F.countDistinct('user_id')).show()
    users = test_set.select('user_id').distinct().rdd.flatMap(lambda x: x).collect()
    print(len(users))
    list_tuples = []
    for user in users:
        current_user_rmsids = test_set.filter(test_set.user_id == user).select('recording_msid').distinct().rdd.flatMap(
            lambda x: x).collect()
        list_tuples.append((baseline_predictions, current_user_rmsids))
    prediction_Labels = spark.sparkContext.parallelize(list_tuples)
    metrics = RankingMetrics(prediction_Labels)
    print(metrics.meanAveragePrecision)
    print("precision at 100", metrics.precisionAt(100))
    print("precision at 15", metrics.precisionAt(15))
    return metrics.meanAveragePrecision


def main(spark, userID):
    '''
    Parameters
    spark : SparkSession object
    userID : string, userID to find files in HDFS
    '''
    # #################################################################################################################
    # Baseline model
    # start = time.time()

    train_set = spark.read.parquet(f'hdfs:/user/ss16270_nyu_edu/train_full.parquet')
    train_set = train_set.repartition("recording_msid")
    train_set.createOrReplaceTempView('train_set')

    val_set = spark.read.parquet(f'hdfs:/user/ss16270_nyu_edu/val_full.parquet')
    val_set = val_set.repartition("user_id")
    # val_set.createOrReplaceTempView('val_set')

    # beta_g = [0.01, 0.1, 1, 10, 100, 1000, 10000]
    # beta_i = [0.01, 0.1, 1, 10, 100, 1000, 10000]
    beta_g = [1000]
    beta_i = [1000]

    for i in range(len(beta_g)):
        mu = 1/(1+beta_g[i])
        grouped_result = spark.sql(
            'SELECT recording_msid, COUNT(user_id) as cum_rating, COUNT(DISTINCT(user_id)) as num_users FROM train_set '
            'GROUP BY recording_msid ORDER BY cum_rating DESC LIMIT 100')

        grouped_result.createOrReplaceTempView("grouped_result")
        # baseline_output = spark.sql(
        #     f'SELECT recording_msid, (cum_rating - {mu[0]})/(num_users + {beta_i[0]}) as avg_rating FROM grouped_result '
        #     f'ORDER BY avg_rating DESC LIMIT 100')
        baseline_output = spark.sql(
            f'SELECT recording_msid FROM grouped_result '
            f'ORDER BY (cum_rating - {mu})/(num_users + {beta_i[i]}) DESC LIMIT 100')
        # baseline_output.show()
        prediction = baseline_output.select('recording_msid').rdd.flatMap(lambda x: x).collect()
        print("Printing top 100 recording msids")
        print(prediction)

        map = baseline_evaluation(spark, prediction, val_set)
        print(map, beta_g[i], beta_i[i])

    ###################################################################################################################
    # evaluation
    # val_f = val_set.groupby('user_id').agg(F.collect_set('recording_msid').alias('unique_recordings'))
    # print("printing validation recording ids' lists for various users")
    # val_f.show()
    # val_f.createOrReplaceTempView('val_f')
    # ground_truth = val_f.select('unique_recordings').rdd.flatMap(lambda x: x).collect()
    # print("Ground truth")
    # print(ground_truth)

    # end = time.time()
    # print(f"Total time for evaluation:{end - start}")
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
