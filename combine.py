
import cv2
import mediapipe as mp
import numpy as np
import string
import csv

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

LOGREG_PATH = "isl_logreg_model"
RF_PATH = "rf_pyspark_model"
DATASET_PATH = "ISL_landmark.csv"

spark = SparkSession.builder \
    .appName("ISL_LogReg_RF_Live") \
    .master("local[2]") \
    .config("spark.driver.memory", "3g") \
    .config("spark.sql.shuffle.partitions", "2") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("Loading Logistic Regression model...")
logreg_model = PipelineModel.load(LOGREG_PATH)

print("Loading Random Forest model...")
rf_model = PipelineModel.load(RF_PATH)

with open(DATASET_PATH, "r", newline="") as file:
    feature_columns = next(csv.reader(file))

feature_columns.remove("target")

print("Both models loaded successfully!")
print("Number of features:", len(feature_columns))

label_map = {
    i: ch for i, ch in enumerate(string.ascii_uppercase)
}

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils

POINTS = 21


def extract_features(results):
    left = [(-1.0, -1.0, -1.0)] * POINTS
    right = [(-1.0, -1.0, -1.0)] * POINTS

    uses_two = 0.0

    if results.multi_hand_landmarks:
        for lm, hd in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):
            side = hd.classification[0].label.lower()

            pts = [
                (p.x, p.y, p.z)
                for p in lm.landmark
            ]

            if "right" in side:
                right = pts
            else:
                left = pts

        if (
            any(p != (-1.0, -1.0, -1.0) for p in left)
            and any(p != (-1.0, -1.0, -1.0) for p in right)
        ):
            uses_two = 1.0

    features = [uses_two]

    for point in left:
        features.extend(point)

    for point in right:
        features.extend(point)

    return np.array(features, dtype=np.float32)


def predict(model, input_df):
    result = model.transform(input_df) \
        .select("prediction", "probability") \
        .first()

    prediction = int(result["prediction"])
    probabilities = result["probability"].toArray()

    confidence = float(probabilities[prediction])

    letter = label_map.get(prediction, "?")

    return letter, confidence


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

logreg_prediction = "?"
rf_prediction = "?"

logreg_confidence = 0.0
rf_confidence = 0.0

frame_count = 0

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:

            for lm in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    lm,
                    mp_hands.HAND_CONNECTIONS
                )

            frame_count += 1

            if frame_count % 5 == 0:

                features = extract_features(results)

                if len(features) != len(feature_columns):
                    raise ValueError(
                        f"Feature mismatch: expected {len(feature_columns)}, "
                        f"got {len(features)}"
                    )

                row = {
                    feature_columns[i]: float(features[i])
                    for i in range(len(feature_columns))
                }

                input_df = spark.createDataFrame([row])

                logreg_prediction, logreg_confidence = predict(
                    logreg_model,
                    input_df
                )

                rf_prediction, rf_confidence = predict(
                    rf_model,
                    input_df
                )

        # Prediction display panel
        cv2.rectangle(
            frame,
            (10, 10),
            (700, 280),
            (30, 30, 30),
            -1
        )

        # Logistic Regression heading
        cv2.putText(
            frame,
            "LOGISTIC REGRESSION",
            (30, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # Large Logistic Regression prediction
        cv2.putText(
            frame,
            logreg_prediction,
            (40, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.5,
            (0, 255, 0),
            5
        )

        cv2.putText(
            frame,
            f"Confidence: {logreg_confidence * 100:.2f}%",
            (150, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        # Random Forest heading
        cv2.putText(
            frame,
            "RANDOM FOREST",
            (350, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 200, 255),
            2
        )

        # Large Random Forest prediction
        cv2.putText(
            frame,
            rf_prediction,
            (370, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.5,
            (0, 200, 255),
            5
        )

        cv2.putText(
            frame,
            f"Confidence: {rf_confidence * 100:.2f}%",
            (480, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.imshow("ISL - Logistic Regression vs Random Forest", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    hands.close()
    cv2.destroyAllWindows()
    spark.stop()
