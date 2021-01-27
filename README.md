# COVID19-Cases---Random-Forest-
using Spark
********************************************************
Name: Joy Obisesan                          **  
                                                      **
Assignment 2- Big Data Integration                    **

*********************************************************
*********************************************************
	COVID19 Cases  - Random Forest                     **
*********************************************************
*********************************************************

-- Download datasets and upload into HDFS

hadoop fs -copyFromLocal /home/cloudera/Downloads/COVID19.csv /BigData/.

--- start spark Shell
spark-shell --master yarn --jars commons-csv-1.5.jar,spark-csv_2.10-1.5.0.jar


--- import libraries 

import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.types.{IntegerType, DoubleType}

-- Loading the CSV file into Hdfs with the inferschema set to true

val covid = sqlContext.read.format("com.databricks.spark.csv").option("header", "true") .option("inferSchema", "true")  .load("hdfs://localhost:8020/BigData/COVID19.csv")
-- From the data, l selected outcome and cases in the hospital 
val dataset = covid.select(col("Outcome"), col("Currently Hospitalized"),col("_id").cast(DoubleType),col("Currently in ICU"),col("Currently Intubated"),col("Classification"))

val New_data = dataset.na.drop()

--train and test the data 
val Array(trainingData, testData) = New_data.randomSplit(Array(0.8, 0.2), 754)

val Outcomeindexer = new StringIndexer().setInputCol("Outcome").setOutputCol("Outcome_Indexed")

val Hospitalizedindexer = new StringIndexer().setInputCol("Currently Hospitalized").setOutputCol("Currently Hospitalized_Indexed")
val ICUindexer = new StringIndexer().setInputCol("Currently in ICU").setOutputCol("Currently in ICU_Indexed")
val Intubatedindexer = new StringIndexer().setInputCol("Currently Intubated").setOutputCol("Currently Intubated_Indexed")

val CALindexer = new StringIndexer().setInputCol("Classification").setOutputCol("Classification_Indexed")

---assemble the newly created columns

val assembler = new VectorAssembler().setInputCols(Array("Currently Hospitalized_Indexed", "Currently Intubated_Indexed", "Currently in ICU_Indexed", "Classification_Indexed", "_id")).setOutputCol("assembled-features")
 
 ---create a new random forest 
val rf = new RandomForestClassifier().setFeaturesCol("assembled-features").setLabelCol("Outcome_Indexed").setSeed(1234)

val pipeline = new Pipeline().setStages(Array(Outcomeindexer,indexer,ICUindexer,Intubatedtindexer,CALindexer,assembler, rf))

---evaluate the model

val evaluator = new MulticlassClassificationEvaluator().setLabelCol("Outcome_Indexed").setPredictionCol("prediction").setMetricName("precision")

val paramGrid = new ParamGridBuilder().addGrid(rf.maxDepth, Array(3, 5)).addGrid(rf.impurity, Array("entropy","gini")).build()

---Cross validate the model
val cross_validator = new CrossValidator().setEstimator(pipeline).setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setNumFolds(3)

---Train the model on training data
val cvModel = cross_validator.fit(trainingData)

val predictions = cvModel.transform(testData)

val accuracy = evaluator.evaluate(predictions)

println("accuracy on test data = " + accuracy)
