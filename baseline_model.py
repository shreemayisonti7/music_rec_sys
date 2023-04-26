#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time

from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql.functions import percent_rank


def main(spark, userID):
    '''
    Parameters
    spark : SparkSession object
    userID : string, userID to find files in HDFS
    '''
###################################################################################################################################
#Baseline model

    train_set = spark.read.parquet(f'hdfs:/user/ss16270_nyu_edu/train_full.parquet')
    train_set = train_set.repartition("recording_msid")
    #train_set.sort("recording_msid")
    
    train_set.createOrReplaceTempView('train_set')
    
#     query_1 = spark.sql("SELECT COUNT(user_id) as ranking FROM train_set GROUP BY recording_msid")
#     query_1.show()
    beta_g = 10
    beta_i = 100

    mu = 1/(1+beta_g)

    query_2 = spark.sql("SELECT recording_msid, COUNT(user_id) as cum_rating, COUNT(DISTINCT(user_id)) as num_users FROM train_set GROUP BY recording_msid ORDER BY cum_rating DESC LIMIT 100")
#     print("Ordered table")
#     query_2.show()
    
    query_2.createOrReplaceTempView("query_2")
    
    query_3 = spark.sql("SELECT recording_msid, (cum_rating-10)/(num_users+100) as avg_rating FROM query_2 ORDER BY avg_rating DESC LIMIT 100")
    query_3.show()


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('checkpoint').getOrCreate()

    # Get user userID from the command line to access HDFS folder
    userID = os.environ['USER']
    print(userID)

    # Calling main
    main(spark, userID)