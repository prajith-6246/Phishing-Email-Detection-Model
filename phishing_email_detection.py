"""
Phishing Email Detection Model
Educational project using Scikit-learn.

Expected dataset: emails.csv with columns:
    email_text,label
Labels should be Phishing or Safe.
"""

import re
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


DATASET_FILE = "emails.csv"
TEXT_COLUMN = "email_text"
LABEL_COLUMN = "label"

SUSPICIOUS_TERMS = [
    "verify", "urgent", "password", "account suspended", "click here",
    "confirm", "login", "limited time", "payment", "update your account"
]


def clean_text(value):
    """Convert a value to safe, normalized text."""
    if pd.isna(value):
        return ""
    return str(value).lower().strip()


def extract_url_features(texts):
    """
    Extract simple numeric indicators from email text.
    This function returns a 2D list suitable for Scikit-learn.
    """
    features = []

    for text in texts:
        text = clean_text(text)
        url_count = len(re.findall(r"https?://|www\\.", text))
        suspicious_count = sum(text.count(term) for term in SUSPICIOUS_TERMS)
        has_url = 1 if url_count > 0 else 0
        features.append([url_count, suspicious_count, has_url])

    return features


def load_dataset():
    """Load and validate the dataset."""
    try:
        data = pd.read_csv(DATASET_FILE)
    except FileNotFoundError:
        print(f"Dataset file '{DATASET_FILE}' was not found.")
        print("Create it using the format shown in README.md.")
        sys.exit(1)

    if TEXT_COLUMN not in data.columns or LABEL_COLUMN not in data.columns:
        print(f"CSV must contain columns '{TEXT_COLUMN}' and '{LABEL_COLUMN}'.")
        sys.exit(1)

    data = data[[TEXT_COLUMN, LABEL_COLUMN]].dropna(subset=[LABEL_COLUMN])
    data[TEXT_COLUMN] = data[TEXT_COLUMN].apply(clean_text)
    data[LABEL_COLUMN] = data[LABEL_COLUMN].astype(str).str.strip().str.lower()

    # Normalize common label variants.
    label_map = {
        "phishing": "Phishing",
        "spam": "Phishing",
        "1": "Phishing",
        "safe": "Safe",
        "legitimate": "Safe",
        "ham": "Safe",
        "0": "Safe",
    }
    data[LABEL_COLUMN] = data[LABEL_COLUMN].map(label_map)
    data = data.dropna(subset=[LABEL_COLUMN])

    if data[LABEL_COLUMN].nunique() < 2:
        print("Dataset must contain examples from both Phishing and Safe classes.")
        sys.exit(1)

    if len(data) < 10:
        print("Please provide at least 10 labeled email examples.")
        sys.exit(1)

    return data


def build_model():
    """Build a pipeline combining TF-IDF text features and URL indicators."""
    text_features = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2),
    )

    url_features = FunctionTransformer(
        extract_url_features,
        validate=False,
    )

    combined_features = FeatureUnion([
        ("text", text_features),
        ("url", url_features),
    ])

    model = Pipeline([
        ("features", combined_features),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

    return model


def main():
    print("=== Phishing Email Detection Model ===")

    data = load_dataset()
    X = data[TEXT_COLUMN]
    y = data[LABEL_COLUMN]

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
    except ValueError:
        print("Not enough examples in each class for a stratified test split.")
        print("Add more labeled examples to emails.csv.")
        sys.exit(1)

    model = build_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    print(f"\\nAccuracy: {accuracy:.2%}")

    print("\\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        labels=["Phishing", "Safe"],
        zero_division=0,
    ))

    print("Confusion Matrix (rows = actual, columns = predicted)")
    print("Label order: Phishing, Safe")
    print(confusion_matrix(y_test, predictions, labels=["Phishing", "Safe"]))

    print("\\nEnter an email message to classify (or type 'exit' to quit).")
    while True:
        email_text = input("\\nEmail: ").strip()

        if email_text.lower() == "exit":
            break

        if not email_text:
            print("Please enter some email text.")
            continue

        result = model.predict([email_text])[0]
        print(f"Prediction: {result}")


if __name__ == "__main__":
    main()
