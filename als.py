import os
import time

from pyspark.sql import SparkSession
# from pyspark.sql import Window
# from pyspark.sql.functions import percent_rank
#
# from pyspark.ml.evaluation import RegressionEvaluator
# from pyspark.ml.recommendation import ALS
# from pyspark.sql import Row
from pyspark.ml.feature import StringIndexer
import pyspark.sql.functions as F
from pyspark.sql.functions import desc, row_number, monotonically_increasing_id
from pyspark.sql.window import Window


def main(spark, userID):
    print("---------------------------Converting recording_msids to integer for train---------------------------------")
    # start = time.time()
    # train_data = spark.read.parquet(f'hdfs:/user/ss16270_nyu_edu/train_full_joined.parquet')
    # # val_data = spark.read.parquet(f'hdfs:/user/ss16270_nyu_edu/val_full_joined.parquet')
    # #
    # # test_data = spark.read.parquet(f'hdfs:/user/ss16270_nyu_edu/test_full_joined.parquet')
    #
    # recording_data = train_data.select("recording_msid")
    # rec_fin = recording_data.groupBy("recording_msid").agg(F.countDistinct("recording_msid").alias("rec_frequency"))
    #
    # rec_data = rec_fin.withColumn('recording_index',
    #                                row_number().over(Window.partitionBy("recording_msid").orderBy("rec_frequency")) - 1)
    # print("Index data")
    # rec_data.show()
    #
    # train_new = train_data.join(rec_data,on="recording_msid",how="left")
    # print("Joined data")
    # train_f = train_new.select("user_id","recording_msid","recording_index")
    # train_f.show()
    #
    # train_f.write.parquet(f'hdfs:/user/ss16270_nyu_edu/train_full_als.parquet', mode="overwrite")
    # end = time.time()
    # print(f"Time for execution:{end-start}")

    print("---------------------------Converting recording_msids to integer for train---------------------------------")
    # train_data = spark.read.parquet(f'hdfs:/user/ss16270_nyu_edu/train_full_als.parquet')
    # train_data.show()
    start = time.time()
    #train_data = spark.read.parquet(f'hdfs:/user/ss16270_nyu_edu/train_full_joined.parquet')
    val_data = spark.read.parquet(f'hdfs:/user/ss16270_nyu_edu/val_full_joined.parquet')
    #
    # test_data = spark.read.parquet(f'hdfs:/user/ss16270_nyu_edu/test_full_joined.parquet')

    recording_data = val_data.select("recording_msid")
    rec_fin = recording_data.groupBy("recording_msid").agg(F.countDistinct("recording_msid").alias("rec_frequency"))

    rec_data = rec_fin.withColumn('recording_index',
                                  row_number().over(Window.orderBy("rec_frequency")) - 1)
    print("Index data")
    rec_data.show()

    val_new = val_data.join(rec_data, on="recording_msid", how="left")
    print("Joined data")
    val_f = val_new.select("user_id", "recording_msid", "recording_index")
    val_f.show()

    val_f.write.parquet(f'hdfs:/user/ss16270_nyu_edu/val_full_als.parquet', mode="overwrite")

    print("---------------------------Converting recording_msids to integer for train---------------------------------")
    test_data = spark.read.parquet(f'hdfs:/user/ss16270_nyu_edu/test_full_joined.parquet')

    recording_data_t = test_data.select("recording_msid")
    rec_fin_t = recording_data_t.groupBy("recording_msid").agg(F.countDistinct("recording_msid").alias("rec_frequency"))

    rec_data_t = rec_fin_t.withColumn('recording_index',
                                  row_number().over(Window.orderBy("rec_frequency")) - 1)
    print("Index data")
    rec_data_t.show()

    test_new = test_data.join(rec_data_t, on="recording_msid", how="left")
    print("Joined data")
    test_f = test_new.select("user_id", "recording_msid", "recording_index")
    test_f.show()

    test_f.write.parquet(f'hdfs:/user/ss16270_nyu_edu/test_full_als.parquet', mode="overwrite")

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





