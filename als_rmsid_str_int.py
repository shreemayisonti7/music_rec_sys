import os
import time

from pyspark.sql import SparkSession
from pyspark.sql.window import Window
import pyspark.sql.functions as F



def main(spark):

    # print("---------------------------Creating rmsid str-int map and saving as parquet------------------------------")
    # start = time.time()
    #
    # train_data = spark.read.parquet(f'hdfs:/user/ss16270_nyu_edu/train_full_joined.parquet')
    #
    # unique_msids = train_data.select('recording_msid').distinct()  #<class 'pyspark.sql.dataframe.DataFrame'>
    # print(unique_msids.count())
    #
    # window_order_by_rmsid = Window.orderBy('recording_msid')
    # rmsid_mapping = unique_msids.select('recording_msid', F.rank().over(window_order_by_rmsid).alias('rmsid_int'))
    # print(rmsid_mapping.count())
    #
    # rmsid_mapping.write.parquet(f'hdfs:/user/ss16270_nyu_edu/rmsid_str_int_map.parquet', mode="overwrite")
    #
    # end = time.time()
    # print(f"Time for creation and saving a map:{end - start}")

#######################################################################################################################
    start = time.time()

    rsmid_map = spark.read.parquet(f'hdfs:/user/ss16270_nyu_edu/rmsid_str_int_map.parquet')

    print("Converting recording_msids to integer for train using the map and generating counts for training ALS")

    train_data = spark.read.parquet(f'hdfs:/user/ss16270_nyu_edu/train_full_joined.parquet')
    train_set = train_data.join(rsmid_map, on='recording_msid', how='left')
    train_set = train_set.select('user_id', 'rmsid_int', 'timestamp')
    train_set = train_set.groupBy('user_id', 'rmsid_int').agg(F.countDistinct("timestamp").alias("ratings"))
    train_set.show()

    # print("Converting recording_msids to integer for val using the map and generating counts for training ALS")
    # val_data = spark.read.parquet(f'hdfs:/user/ss16270_nyu_edu/val_full_joined.parquet')

    end = time.time()
    print(f"Time for execution:{end - start}")


if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName('checkpoint').getOrCreate()

    main(spark)
