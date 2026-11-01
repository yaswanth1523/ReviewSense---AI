
import joblib
import pandas as pd

# Load the saved pipeline
model = joblib.load("sentiment_pipeline.joblib")

print("Model loaded successfully!")

# Sample restaurant reviews
sample_reviews = [
    "The food was absolutely delicious and the service was excellent.",
    "I don't like this restaurant. The food was terrible.",
    "The ambience was good but the service was very slow.",
    "I would definitely recommend this place."
]

# Convert reviews into a pandas Series
# Our preprocessing function uses Series.apply()
reviews = pd.Series(sample_reviews)

# Generate predictions
predictions = model.predict(reviews)

# Display predictions
for review, prediction in zip(sample_reviews, predictions):

    sentiment = "Positive" if prediction == 1 else "Negative"

    print("\nReview:", review)
    print("Predicted Sentiment:", sentiment)
    print("-" * 70)
