import os
import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import rank
from pyspark.sql.window import Window



def main(spark):
    print("---------------------------Converting recording_msids to integer for train---------------------------------")
    start = time.time()

    train_data = spark.read.parquet(f'hdfs:/user/ss16270_nyu_edu/train_full_joined.parquet')
    # val_data = spark.read.parquet(f'hdfs:/user/ss16270_nyu_edu/val_full_joined.parquet')

    unique_msids = train_data.select('recording_msid').distinct()
    unique_msids.show()
    print("Type of unique_msids", type(unique_msids))

    window_order_by_rmsid = Window.orderBy('recording_msid')
    rmsid_mapping = unique_msids.select('recording_msid', rank().over(window_order_by_rmsid).alias(
        'rmsid_int'))
    rmsid_mapping.show()

    end = time.time()
    print(f"Time for execution:{end - start}")





if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName('checkpoint').getOrCreate()

    main(spark)





