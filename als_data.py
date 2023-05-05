import os
import time

from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
import pyspark.sql.functions as F
from pyspark.sql.functions import desc, row_number, monotonically_increasing_id
from pyspark.sql.window import Window
from pyspark.sql.functions import when


def main(spark, userID):
    print("---------------------------Converting recording_msids to integer for train---------------------------------")
    start=time.time()

    train_tracks = spark.read.parquet(f'hdfs:/user/bm106_nyu_edu/1004-project-2023/tracks_train.parquet')
    test_tracks = spark.read.parquet(f'hdfs:/user/bm106_nyu_edu/1004-project-2023/tracks_test.parquet')

    new_train = train_tracks.select(when(col('recording_mbid').isNotNull(), col('recording_mbid')
                                                             ).otherwise(col('recording_msid')).alias('recording_msid')).distinct()

    new_test = test_tracks.select(when(col('recording_mbid').isNotNull(), col('recording_mbid')
                                                             ).otherwise(col('recording_msid')).alias('recording_msid')).distinct()

    final_df = new_train.union(new_test)
    final_data = final_df.distinct()

    rec_fin = final_data.groupBy("recording_msid").agg(F.countDistinct("recording_msid").alias("rec_frequency"))
    rec_data = rec_fin.withColumn('recording_index',
                                  row_number().over(Window.partitionBy("recording_msid").orderBy("rec_frequency")) - 1)
    print("Index data")
    rec_data.show()

    rec_data.write.parquet(f'hdfs:/user/ss16270_nyu_edu/rec_index_als.parquet', mode="overwrite")

    end = time.time()
    print(f"Time for execution:{end - start}")





if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName('checkpoint').getOrCreate()

    # Get user userID from the command line to access HDFS folder
    userID = os.environ['USER']
    print(userID)

    # Calling main
    main(spark, userID)





