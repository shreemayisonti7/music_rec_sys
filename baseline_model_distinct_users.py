#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
# from pyspark.mllib.evaluation import RankingMetrics

import time


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


def evaluator(baseline_predictions, test_set):
    test_set = test_set.groupBy('user_id').agg(F.collect_list('recording_msid').alias('ground_truth_songs'))
    udf_ap = F.udf(lambda ground_truth_songs:
                   average_precision_calculator(baseline_predictions, ground_truth_songs))
    test_set = test_set.withColumn('average_precision', udf_ap(F.col('ground_truth_songs')))

    udf_rr = F.udf(lambda ground_truth_songs:
                   reciprocal_rank_calculator(baseline_predictions, ground_truth_songs))

    test_set = test_set.withColumn('reciprocal_rank', udf_rr(F.col('ground_truth_songs')))

    mean_average_precision = test_set.agg(F.mean(F.col("average_precision")).alias("mean_average_precision")
                                          ).collect()[0]['mean_average_precision']
    mean_reciprocal_rank = test_set.agg(F.mean(F.col("reciprocal_rank")).alias("mean_reciprocal_rank")
                                        ).collect()[0]['mean_reciprocal_rank']
    return mean_average_precision, mean_reciprocal_rank


def main(spark):
    """
    Parameters
    spark : SparkSession object
    userID : string, userID to find files in HDFS
    """
    start = time.time()

    # train_file = f'hdfs:/user/ss16270_nyu_edu/train_small.parquet'
    # val_file = f'hdfs:/user/ss16270_nyu_edu/val_small.parquet'

    train_file = f'hdfs:/user/ss16270_nyu_edu/train_full.parquet'
    val_file = f'hdfs:/user/ss16270_nyu_edu/val_full.parquet'
    test_file = f'hdfs:/user/bm106_nyu_edu/1004-project-2023/interactions_test.parquet'

    train_set = spark.read.parquet(train_file)
    val_set = spark.read.parquet(val_file)
    test_set = spark.read.parquet(test_file)

    grouped_result = train_set.groupBy('recording_msid').agg(F.countDistinct('user_id').alias('num_users'))
    baseline_output = grouped_result.orderBy(F.col('num_users'), ascending=False).limit(100)
    prediction = baseline_output.select('recording_msid').rdd.flatMap(lambda x: x).collect()

    val_map, val_mrr = evaluator(prediction, val_set)
    test_map, test_mrr = evaluator(prediction, test_set)

    end = time.time()

    print("MAP and MRR on validation set:", val_map, val_mrr)
    print("MAP and MRR on test set:", test_map, test_mrr)
    print(f"Total time for evaluation:{end - start}")
    print("The end")


if __name__ == "__main__":
    spark = SparkSession.builder.appName('checkpoint').getOrCreate()
    main(spark)
    spark.stop()
