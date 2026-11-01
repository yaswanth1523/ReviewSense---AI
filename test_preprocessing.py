from preprocessing import preprocess_basic_lemma

sample_reviews = [
    "I don't like this restaurant!",
    "The food was extremely delicious.",
    "I can't recommend this place.",
    "The service was very bad, but the food was good."
]

for review in sample_reviews:
    print("Original:", review)
    print("Processed:", preprocess_basic_lemma(review))
    print("-" * 60)