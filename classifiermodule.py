from pyspark.ml import PipelineModel
import os
import joblib
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import pandas as pd
import numpy as np
import sys

os.environ["PYSPARK_PYTHON"]="/usr/bin/python3.7"
spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
pipelineLoad = PipelineModel.load('./classifier_Word2Vec.pipeline')
model = joblib.load('./classifier_MLP.pkl')
modelParagraphVector = joblib.load('./classifier_Paragraph_Vector.pkl')
mensagens_intervencao_identificada_all_issues = spark.read.parquet('./mensagens_intervencao_identificada_all_issues')

def returnprediction(name):	
	test = spark.createDataFrame([(1543413, name)],
	                             ["issue_id_messages", "grouped_body_concat_clear"])	
	test = pipelineLoad.transform(test)
	X = test.select('features').rdd.map(lambda row : row[0]).collect()
	#pred_MLP_predict = model.predict(X)	
	pred_MLP_predict = pd.DataFrame({'prediction':model.predict_proba(X)[0], 'group': model.classes_}).sort_values(by='prediction', ascending=False).values
	return pred_MLP_predict


def returnsynonums(name):	
	synonums = np.array(pipelineLoad.stages[2].findSynonyms(name, 3).collect())[0:,0]
	return synonums

def returnsimilarparagraph(name):
	test = spark.createDataFrame([(1, name)], ["issue_id_messages", "grouped_body_concat_clear"])
	print(pipelineLoad.stages[1].transform(pipelineLoad.stages[0].transform(test)).collect()[0][3], file=sys.stderr) 
	wordsTransform = pipelineLoad.stages[1].transform(pipelineLoad.stages[0].transform(test)).collect()[0][3]
	infer_vector = modelParagraphVector.infer_vector(wordsTransform)
	resultssimilar = modelParagraphVector.docvecs.most_similar([infer_vector], topn = 3)
	return resultssimilar

def returnissues(name):
	mensagens_intervencao_identificada_depois_intervencao = mensagens_intervencao_identificada_all_issues.filter((col('issue_id_messages') == 1895328))
	print(mensagens_intervencao_identificada_depois_intervencao, file=sys.stderr) 
	return mensagens_intervencao_identificada_depois_intervencao.collect()[0].body
