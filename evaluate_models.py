
print("\n\nRan using Docker!")
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder \
    .appName("ISL_Model_Evaluation") \
    .master("local[2]") \
    .config("spark.driver.memory", "3g") \
    .config("spark.sql.shuffle.partitions", "2") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

try:
    df = spark.read.csv(
        "ISL_landmark.csv",
        header=True,
        inferSchema=True
    )

    df = df.dropna()

    train_data, test_data = df.randomSplit(
        [0.8, 0.2],
        seed=42
    )

    print("Total records:", df.count())
    print("Training records:", train_data.count())
    print("Testing records:", test_data.count())

    logreg_model = PipelineModel.load("isl_logreg_model")
    rf_model = PipelineModel.load("rf_pyspark_model")

    evaluator = MulticlassClassificationEvaluator(
        labelCol="target",
        predictionCol="prediction"
    )

    models = [
        ("Logistic Regression", logreg_model),
        ("Random Forest", rf_model)
    ]

    for name, model in models:

        predictions = model.transform(test_data)

        accuracy = evaluator.setMetricName("accuracy").evaluate(predictions)
        precision = evaluator.setMetricName("weightedPrecision").evaluate(predictions)
        recall = evaluator.setMetricName("weightedRecall").evaluate(predictions)
        f1 = evaluator.setMetricName("f1").evaluate(predictions)

        print("\n" + "=" * 45)
        print(name.upper())
        print("=" * 45)

        print(f"Accuracy:          {accuracy * 100:.2f}%")
        print(f"Weighted Precision:{precision * 100:.2f}%")
        print(f"Weighted Recall:   {recall * 100:.2f}%")
        print(f"F1 Score:          {f1 * 100:.2f}%")

        print("\nConfusion Matrix:")

        matrix = predictions.groupBy("target") \
            .pivot("prediction", list(range(26))) \
            .count() \
            .na.fill(0) \
            .orderBy("target")

        matrix.show(26, False)

    print("\nModel evaluation completed!")

finally:
    spark.stop()
