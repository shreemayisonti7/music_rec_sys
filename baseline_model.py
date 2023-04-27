#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from pyspark.sql import SparkSession
import pyspark.sql.functions as F

import time


def main(spark, userID):
    '''
    Parameters
    spark : SparkSession object
    userID : string, userID to find files in HDFS
    '''
    # #################################################################################################################
    #Baseline model
    start = time.time()
    val_set = spark.read.parquet(f'hdfs:/user/ss16270_nyu_edu/val_full.parquet')
    val_set = val_set.repartition("recording_msid")
    # val_set.sort("recording_msid")

    val_set.createOrReplaceTempView('val_set')
    
    mu = [0.99, 0.909, 0.5, 0.09, 0.009, 0.0009]
    beta_i = [0.01, 0.1, 1, 10, 100, 1000]
    # mu = 1/(1+beta_g)
    
    grouped_result = spark.sql(
        'SELECT recording_msid, COUNT(user_id) as cum_rating, COUNT(DISTINCT(user_id)) as num_users FROM val_set '
        'GROUP BY recording_msid ORDER BY cum_rating DESC LIMIT 100')

    grouped_result.createOrReplaceTempView("grouped_result")

    baseline_output = spark.sql(
        f'SELECT recording_msid, (cum_rating - {mu[0]})/(num_users + {beta_i[0]}) as avg_rating FROM grouped_result '
        f'ORDER BY avg_rating DESC LIMIT 100')
    baseline_output.show()
    baseline_output.createOrReplaceTempView('baseline_output')
    
    ######################################################################################################################
    #evaluation
    val_f = val_set.groupby('user_id').agg(F.collect_set('recording_msid')).alias('unique_recordings')
    ground_truth = val_f.select('unique_recordings').rdd.flatMap(lambda x: x).collect()
    print("Ground truth")
    ground_truth.show()
    
    prediction = baseline_output.select('avg_rating').rdd.flatMap(lambda x:x).collect()
    print("Prediction")
    prediction.show()
    
    end=time.time()
    
    print(f"Total time for evaluation:{end-start}")
    
    

# Only enter this block if we're in main
if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName('checkpoint').getOrCreate()

    # Get user userID from the command line to access HDFS folder
    userID = os.environ['USER']
    print(userID)

    # Calling main
    main(spark, userID)
