from pyspark.sql import SparkSession
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder \
    .appName("ISL_Logistic_Regression") \
    .getOrCreate()

df = spark.read.csv("ISL_landmark.csv", header=True, inferSchema=True)

df = df.dropna()

feature_columns = [c for c in df.columns if c != "target"]

train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="raw_features"
)

scaler = StandardScaler(
    inputCol="raw_features",
    outputCol="scaled_features",
    withMean=True,
    withStd=True
)

lr = LogisticRegression(
    featuresCol="scaled_features",
    labelCol="target",
    maxIter=1000
)

pipeline = Pipeline(stages=[assembler, scaler, lr])

model = pipeline.fit(train_data)

predictions = model.transform(test_data)

evaluator = MulticlassClassificationEvaluator(
    labelCol="target",
    predictionCol="prediction",
    metricName="accuracy"
)

accuracy = evaluator.evaluate(predictions)

print("Logistic Regression Accuracy:", accuracy)

predictions.groupBy("target", "prediction").count().orderBy(
    "target", "prediction"
).show()

predictions.select(
    "target", "prediction", "probability"
).show(20, truncate=False)

model.write().overwrite().save("isl_logreg_model")

print("Model saved successfully!")

spark.stop()
