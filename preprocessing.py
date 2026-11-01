
import re
import nltk

from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer


# --------------------------------------------------
# 1. Download required NLTK resources
# --------------------------------------------------

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


# --------------------------------------------------
# 2. Define sentiment-related protected words
# --------------------------------------------------

negation_words = {
    'no', 'not', 'nor', 'never', 'without'
}

negation_contractions = {
    "don't", "doesn't", "didn't",
    "isn't", "aren't", "wasn't", "weren't",
    "haven't", "hasn't", "hadn't",
    "won't", "wouldn't", "couldn't",
    "shouldn't", "mustn't", "mightn't",
    "needn't", "shan't", "can't"
}

intensifiers = {
    'very', 'extremely', 'really',
    'too', 'so', 'quite', 'highly',
    'absolutely', 'completely',
    'barely', 'hardly'
}

contrast_words = {
    'but', 'however', 'although',
    'though', 'yet', 'despite', 'still'
}

modifiers = {
    'more', 'most', 'less', 'least',
    'only', 'just', 'again', 'almost',
    'enough', 'much', 'little',
    'always', 'sometimes'
}


# --------------------------------------------------
# 3. Combine protected words
# --------------------------------------------------

protected_words = (
    negation_words
    | negation_contractions
    | intensifiers
    | contrast_words
    | modifiers
)

# Protect 'cannot' after contraction expansion
protected_words.add('cannot')


# --------------------------------------------------
# 4. Customize NLTK stopwords
# --------------------------------------------------

stop_words = set(stopwords.words('english')) - protected_words


# --------------------------------------------------
# 5. Define contraction dictionary
# --------------------------------------------------

contractions = {
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "won't": "will not",
    "wouldn't": "would not",
    "couldn't": "could not",
    "shouldn't": "should not",
    "mustn't": "must not",
    "mightn't": "might not",
    "needn't": "need not",
    "shan't": "shall not",
    "can't": "cannot"
}


# --------------------------------------------------
# 6. Create contraction matching pattern
# --------------------------------------------------

contraction_pattern = re.compile(
    r'\b(' + '|'.join(
        re.escape(word)
        for word in sorted(contractions, key=len, reverse=True)
    ) + r')\b'
)


# --------------------------------------------------
# 7. Expand contractions
# --------------------------------------------------

def expand_contractions(text):

    text = text.replace('’', "'")

    return contraction_pattern.sub(
        lambda match: contractions[match.group()],
        text
    )


# --------------------------------------------------
# 8. Common text preprocessing
# --------------------------------------------------

def preprocess_common(text):

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation while preserving apostrophes
    text = re.sub(r"[^\w\s'’]", " ", text)

    # Remove numbers
    text = re.sub(r'\d+', ' ', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Expand contractions
    text = expand_contractions(text)

    # Remove customized stopwords
    text = ' '.join(
        word for word in text.split()
        if word not in stop_words
    )

    return text


# --------------------------------------------------
# 9. Initialize WordNet Lemmatizer
# --------------------------------------------------

wordnet_lemmatizer = WordNetLemmatizer()


# --------------------------------------------------
# 10. Apply Basic Lemmatization
# --------------------------------------------------

def apply_lemmatization(text):

    words = text.split()

    lemmatized_words = [
        wordnet_lemmatizer.lemmatize(word)
        for word in words
    ]

    return ' '.join(lemmatized_words)


# --------------------------------------------------
# 11. Final preprocessing function
# --------------------------------------------------

def preprocess_basic_lemma(text):

    text = preprocess_common(text)

    return apply_lemmatization(text)


def transform_basic_lemma(texts):

    return texts.apply(preprocess_basic_lemma)
