import os
import time

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import desc, row_number, monotonically_increasing_id
from pyspark.sql.window import Window
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.ml.tuning import ParamGridBuilder,CrossValidator
from pyspark.ml import Pipeline

def main(spark, userID):
    print("---------------------------Converting recording_msids to integer for train---------------------------------")

    train_data = spark.read.parquet(f'hdfs:/user/ss16270_nyu_edu/train_full_als.parquet')
    val_data = spark.read.parquet(f'hdfs:/user/ss16270_nyu_edu/val_full_als.parquet')

    train_data_f = train_data.select("user_id","recording_index")
    grouped_data = train_data_f.groupBy("user_id","recording_index").agg(F.count("recording_index").
                                                                         alias("rec_frequency"))

    val_data_f = val_data.select("user_id","recording_index")
    grouped_data_v = val_data_f.groupBy("user_id", "recording_index").agg(F.count("recording_index").
                                                                          alias("rec_frequency"))

    # reg_param = [0.001,0.01,0.1,1,10,100,1000]
    # rank_list = np.linspace(1,10,endpoint=True)
    # alpha = np.linspace(1,10,endpoint=True)

    #als = ALS(maxIter=10, regParam=0.1,alpha = 40, userCol="user_id", itemCol="recording_index", ratingCol="rec_frequency",
    #          coldStartStrategy="drop",implicitPrefs=True)

    # pipeline = Pipeline(stages=[als])
    # paramGrid = ParamGridBuilder() \
    #     .addGrid(als.regParam, [0.001,0.01,0.1,1,10,100,1000]) \
    #     .addGrid(als.rank, [1,2,3,4,5,6,7,8,9,10]) \
    #     .addGrid(als.alpha,[1,2,3,4,5,6,7,8,9,10]) \
    #     .build()
    #
    # crossval = CrossValidator(estimator=pipeline,
    #                           estimatorParamMaps=paramGrid,
    #                           evaluator=BinaryClassificationEvaluator(),
    #                           numFolds=2)  # use 3+ folds in practice
    #
    # # Run cross-validation, and choose the best set of parameters.
    # cvModel = crossval.fit(training)

    reg_param = [0.01,0.1]
    rank_list = [1,5,10]
    alpha_list = [10,25,50]

    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rec_frequency",
                                    predictionCol="prediction")

    rmse_min = float('inf')
    model_params = {'reg':[], 'alpha':[], 'rank':[], 'rmse':[]}


    for i in range(len(reg_param)):
        for j in range(len(rank_list)):
            for k in range(len(alpha_list)):
                als = ALS(maxIter=10, regParam=reg_param[i], alpha=alpha_list[k], rank=rank_list[j], userCol="user_id",
                          itemCol="recording_index",
                          ratingCol="rec_frequency",
                          coldStartStrategy="drop", implicitPrefs=True)

                model = als.fit(grouped_data)
                predictions = model.transform(grouped_data_v)
                rmse = evaluator.evaluate(predictions)
                model_params['reg'].append(reg_param[i])
                model_params['alpha'].append(alpha_list[k])
                model_params['rank'].append(rank_list[j])
                model_params['rmse'].append(rmse)

                if rmse<rmse_min:
                    model.write().overwrite().save(f'hdfs:/user/ss16270_nyu_edu/als_model')

    min_rmse = np.argmin(model_params['rmse'])
    print(f"Reg:{model_params['reg_param'][min_rmse]}, alpha:{model_params['alpha'][min_rmse]}, rank:{model_params['rank']}, rmse:{model_params['rmse'][min_rmse]}")

    #print("Root-mean-square error = " + str(rmse))

    start = time.time()
    # userRecs = model.recommendForAllUsers(100)
    # userRecs.show()
    #
    # userRecs.write.parquet(f'hdfs:/user/ss16270_nyu_edu/als_pred.parquet', mode="overwrite")
    end=time.time()

    print(f"Total time for execution:{end-start}")





if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName('checkpoint').getOrCreate()

    # Get user userID from the command line to access HDFS folder
    userID = os.environ['USER']
    print(userID)

    # Calling main
    main(spark, userID)





