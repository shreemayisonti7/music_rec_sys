#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql.functions import percent_rank


def main(spark, userID):
    '''
    Parameters
    spark : SparkSession object
    userID : string, userID to find files in HDFS
    '''

    #Perform the same on large files later
    train_small_interactions = spark.read.parquet(f'hdfs:/user/bm106_nyu_edu/1004-project-2023/interactions_train_small.parquet')
    # train_small_tracks = spark.read.parquet(f'hdfs:/user/bm106_nyu_edu/1004-project-2023/tracks_train_small.parquet')
    # train_small_users = spark.read.parquet(f'hdfs:/user/bm106_nyu_edu/1004-project-2023/users_train_small.parquet')
    print("number of records in train_small_interactions:")
    train_small_interactions.count()

    # print("Printing stats")
    # print("group by user_id count")
    # train_small_interactions.groupBy('user_id').count().show()
    # print("group by user_id and recording_msid count")
    # train_small_interactions.groupBy('user_id', 'recording_msid').count().show()

    print('Printing train_small_interactions inferred schema')
    train_small_interactions.printSchema()
    train_small_interactions.show(10)
    print(type(train_small_interactions))
    window_partition_by_users = Window.partitionBy('user_id').orderBy('timestamp')
    percent_ranked = train_small_interactions.select('*', percent_rank().over(window_partition_by_users).alias('percent_rank'))

    print("Train set would be")
    train_set = percent_ranked.filter(percent_ranked.percent_rank <= 0.8)
    print("number of records in train_set:")
    train_set.count()
    train_set.write.parquet(f'hdfs:/user/ss16270_nyu_edu/train_small.parquet', mode="overwrite", partitionBy='user_id')
    new_train = spark.read.parquet(f'hdfs:/user/ss16270_nyu_edu/train_small.parquet')
    new_train.show(150)
    print("number of records in new_train:")
    new_train.count()

    print("Validation set would be")
    val_set = percent_ranked.filter(percent_ranked.percent_rank > 0.8)
    print("number of records in val_set:")
    val_set.count()
    val_set.write.parquet(f'hdfs:/user/ss16270_nyu_edu/val_small.parquet', mode="overwrite", partitionBy='user_id')
    new_val = spark.read.parquet(f'hdfs:/user/ss16270_nyu_edu/val_small.parquet')
    new_val.show(150)
    print("number of records in new_val:")
    new_val.count()


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('checkpoint').getOrCreate()

    # Get user userID from the command line to access HDFS folder
    userID = os.environ['USER']
    print(userID)

    # Calling main
    main(spark, userID)