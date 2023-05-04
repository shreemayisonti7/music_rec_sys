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


def main(spark, userID):
    print("--------------------------------Converting interactions ids to integer-------------------------------------")
    start = time.time()
    train_data = spark.read.parquet(f'hdfs:/user/ss16270_nyu_edu/train_full_joined.parquet')
    # val_data = spark.read.parquet(f'hdfs:/user/ss16270_nyu_edu/val_full_joined.parquet')
    #
    # test_data = spark.read.parquet(f'hdfs:/user/ss16270_nyu_edu/test_full_joined.parquet')

    recording_data = train_data.select("recording_msid").distinct()
    print("Unique recording count")
    recording_data.count()

    recording_indexer = StringIndexer(inputCol="recording_msid", outputCol="recordingIndex")
    # Fits a model to the input dataset with optional parameters.
    rec_new = recording_indexer.fit(recording_data).transform(recording_data)
    print("Recording index")
    rec_new.show()

    train_new = train_data.join(rec_new,on="recording_msid",how="left")
    print("Joined data")
    train_new.show()

    train_new.write.parquet(f'hdfs:/user/ss16270_nyu_edu/train_full_als.parquet', mode="overwrite")
    end = time.time()
    print(f"Time for execution:{end-start}")






if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName('checkpoint').getOrCreate()

    # Get user userID from the command line to access HDFS folder
    userID = os.environ['USER']
    print(userID)

    # Calling main
    main(spark, userID)





