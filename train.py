"""Train a TF-IDF + Logistic Regression document classifier.

Labels come from the file name: files containing "invoice" are invoices,
everything else is a bank statement.

Usage:
    python generate_data.py   # create synthetic training data first
    python train.py
"""
import os
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from utils import extract_text

DATA_DIR = "data"

texts, labels = [], []
for name in sorted(os.listdir(DATA_DIR)):
    text = extract_text(os.path.join(DATA_DIR, name))
    if len(text.strip()) < 20:
        continue  # skip empty / unreadable documents
    texts.append(text)
    labels.append("invoice" if "invoice" in name.lower() else "bank")

print(f"Loaded {len(texts)} documents")

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_features=5000,
    # Words only (2+ letters). Numbers are ignored so account numbers,
    # amounts and IDs never end up in the saved model's vocabulary.
    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
)

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

model = LogisticRegression(max_iter=1000)
model.fit(vectorizer.fit_transform(X_train), y_train)

print(classification_report(y_test, model.predict(vectorizer.transform(X_test))))

# Refit on all data for the saved model
model.fit(vectorizer.fit_transform(texts), labels)
with open("model.pkl", "wb") as f:
    pickle.dump((model, vectorizer), f)

print("Saved model.pkl")
