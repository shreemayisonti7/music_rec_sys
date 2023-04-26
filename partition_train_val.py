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

    print('Printing train_small_interactions inferred schema')
    train_small_interactions.printSchema()
    train_small_interactions.show(10)
    print(type(train_small_interactions))
    window_partition_by_users = Window.partitionBy('user_id').orderBy('timestamp')
    percent_ranked = train_small_interactions.select('*', percent_rank().over(window_partition_by_users).alias('percent_rank'))
    print("Train set would be")
    percent_ranked.filter(percent_ranked.user_id == 53).filter(percent_ranked.percent_rank <= 0.8).show(50)
    print("Validation set would be")
    percent_ranked.filter(percent_ranked.user_id == 53).filter(percent_ranked.percent_rank > 0.8).show(50)


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('checkpoint').getOrCreate()

    # Get user userID from the command line to access HDFS folder
    userID = os.environ['USER']
    print(userID)

    # Calling main
    main(spark, userID)
