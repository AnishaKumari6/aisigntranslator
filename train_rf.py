
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark import StorageLevel

spark = SparkSession.builder \
    .appName("ISL_Random_Forest") \
    .master("local[4]") \
    .config("spark.driver.memory", "5g") \
    .config("spark.executor.memory", "5g") \
    .config("spark.driver.maxResultSize", "1g") \
    .config("spark.sql.shuffle.partitions", "4") \
    .config("spark.default.parallelism", "4") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

df = spark.read.csv(
    "ISL_landmark.csv",
    header=True,
    inferSchema=True
).dropna()

feature_columns = [
    c for c in df.columns if c != "target"
]

print("Dataset rows:", df.count())
print("Number of features:", len(feature_columns))

train_data, test_data = df.randomSplit(
    [0.8, 0.2],
    seed=42
)

assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="features"
)

rf = RandomForestClassifier(
    labelCol="target",
    featuresCol="features",
    numTrees=50,
    maxDepth=10,
    maxBins=32,
    maxMemoryInMB=512,
    seed=42,
    subsamplingRate=0.8
)

pipeline = Pipeline(stages=[
    assembler,
    rf
])

print("Training Random Forest...")

model = pipeline.fit(train_data)

predictions = model.transform(test_data)

evaluator = MulticlassClassificationEvaluator(
    labelCol="target",
    predictionCol="prediction",
    metricName="accuracy"
)

print(
    "Random Forest Accuracy:",
    evaluator.evaluate(predictions)
)

for metric in [
    "weightedPrecision",
    "weightedRecall",
    "f1"
]:
    evaluator.setMetricName(metric)
    print(metric, ":", evaluator.evaluate(predictions))

predictions.groupBy(
    "target", "prediction"
).count().show()

model.write().overwrite().save(
    "rf_pyspark_model"
)

print("Random Forest model saved successfully!")

spark.stop()
