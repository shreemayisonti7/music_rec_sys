#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from pyspark.sql import SparkSession


def main(spark, userID):
    '''
    Parameters
    spark : SparkSession object
    userID : string, userID to find files in HDFS
    '''

    #Perform the same on large files later

    train_small_interactions = spark.read.parquet(f'hdfs:/user/bm106_nyu_edu/1004-project-2023/interactions_train_small.parquet')
    train_small_tracks = spark.read.parquet(f'hdfs:/user/bm106_nyu_edu/1004-project-2023/tracks_train_small.parquet')
    train_small_users = spark.read.parquet(f'hdfs:/user/bm106_nyu_edu/1004-project-2023/users_train_small.parquet')

    print('Printing train_small_interactions inferred schema')
    train_small_interactions.printSchema()
    train_small_interactions.show(10)

    print('Printing train_small_users inferred schema')
    train_small_tracks.printSchema()
    train_small_tracks.show(10)

    print('Printing train_small_users inferred schema')
    train_small_users.printSchema()
    train_small_users.show(10)



# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('checkpoint').getOrCreate()

    # Get user userID from the command line to access HDFS folder
    userID = os.environ['USER']

    # Calling main
    main(spark, userID)
