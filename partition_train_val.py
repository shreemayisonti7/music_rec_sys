#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time

from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql.functions import percent_rank


def main(spark, userID):
    """
    Parameters
    spark : SparkSession object
    userID : string, userID to find files in HDFS
    """

    # Perform the same on large files later
    # start_time = time.time() train_small_interactions = spark.read.parquet(
    # f'hdfs:/user/bm106_nyu_edu/1004-project-2023/interactions_train_small.parquet')
    #
    # # print("number of records in train_small_interactions:")
    # # print(train_small_interactions.count())
    #
    # # print("Printing stats")
    # # print("group by user_id count")
    # # train_small_interactions.groupBy('user_id').count()
    # # print("group by user_id and recording_msid count")
    # # train_small_interactions.groupBy('user_id', 'recording_msid').count()
    #
    # # print('Printing train_small_interactions inferred schema')
    # # train_small_interactions.printSchema()
    # # train_small_interactions.show(10)
    # # print(type(train_small_interactions))
    #
    # window_partition_by_users = Window.partitionBy('user_id').orderBy('timestamp') percent_ranked =
    # train_small_interactions.select('*', percent_rank().over(window_partition_by_users).alias('percent_rank'))
    #
    # train_set = percent_ranked.filter(percent_ranked.percent_rank <= 0.8)
    # train_set.write.parquet(f'hdfs:/user/ss16270_nyu_edu/train_small.parquet', mode="overwrite")
    # , partitionBy='user_id')
    # val_set = percent_ranked.filter(percent_ranked.percent_rank > 0.8)
    # val_set.write.parquet(f'hdfs:/user/ss16270_nyu_edu/val_small.parquet', mode="overwrite")
    # , partitionBy='user_id')
    #
    # new_train = spark.read.parquet(f'hdfs:/user/ss16270_nyu_edu/train_small.parquet')
    # new_val = spark.read.parquet(f'hdfs:/user/ss16270_nyu_edu/val_small.parquet')
    #
    # end_time = time.time()
    # print("Time taken to split small interactions into train-val and write to parquet is:", (end_time-start_time))
    # # print("Train set would be")
    # # new_train.show(150)
    # # print("Validation set would be")
    # # new_val.show(150)
    #
    # print("number of records in train_set:")
    # print(train_set.count())
    # print("number of records in val_set:")
    # print(val_set.count())
    # print("number of records in new_train:")
    # print(new_train.count())
    # print("number of records in new_val:")
    # print(new_val.count())

    ###################################################################################################################
    # For large dataset
    print("--------------------------Starting splitting of large file-------------------------")

    start_time = time.time()
    train_interactions = spark.read.parquet(f'hdfs:/user/bm106_nyu_edu/1004-project-2023/interactions_train.parquet')
    print("Counting train elements", train_interactions.count())

    print("User null count")
    print(train_interactions.filter(train_interactions.user_id.isNull()).count())

    print("Timestamp null count")
    print(train_interactions.filter(train_interactions.timestamp.isNull()).count())

    print("MSID null count")
    print(train_interactions.filter(train_interactions.recording_msid.isNull()).count())

    end_time = time.time()
    print("Time taken to get the stats:", (end_time - start_time))

    start_time = time.time()
    window_partition_by_users = Window.partitionBy('user_id').orderBy('timestamp')
    percent_ranked = train_interactions.select('*', percent_rank().over(window_partition_by_users).alias(
        'percent_rank'))

    train_set = percent_ranked.filter(percent_ranked.percent_rank <= 0.8)
    train_set.write.parquet(f'hdfs:/user/ss16270_nyu_edu/train_full.parquet', mode="overwrite")
    # , partitionBy='recording_msid')
    val_set = percent_ranked.filter(percent_ranked.percent_rank > 0.8)
    val_set.write.parquet(f'hdfs:/user/ss16270_nyu_edu/val_full.parquet', mode="overwrite")
    # , partitionBy='recording_msid')

    new_train = spark.read.parquet(f'hdfs:/user/ss16270_nyu_edu/train_full.parquet')
    new_val = spark.read.parquet(f'hdfs:/user/ss16270_nyu_edu/val_full.parquet')

    end_time = time.time()
    print("Time taken to split large interactions into train-val and write to parquet is:", (end_time - start_time))

    print("number of records in full train_set:")
    print(train_set.count())
    print("number of records in full val_set:")
    print(val_set.count())
    print("number of records in full new_train:")
    print(new_train.count())
    print("number of records in full new_val:")
    print(new_val.count())

    print("--------------------------The end--------------------------")


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
