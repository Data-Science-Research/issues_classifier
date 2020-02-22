from pyspark.ml import PipelineModel
import os
import joblib
from pyspark.sql import SparkSession

os.environ["PYSPARK_PYTHON"]="/usr/bin/python3.7"

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

pipelineLoad = PipelineModel.load('./classifier_Word2Vec.pipeline')

model = joblib.load('./classifier_MLP.pkl')

while True:

  sec = input('Let us wait for user input. Let me know how many seconds to sleep now.\n')


  test = spark.createDataFrame([(1543413, sec)],
                             ["issue_id_messages", "grouped_body_concat_clear"])

  test = pipelineLoad.transform(test)

  test.show()

  X = test.select('features').rdd.map(lambda row : row[0]).collect()

  pred_MLP_predict = model.predict(X)
  
  print(pred_MLP_predict)