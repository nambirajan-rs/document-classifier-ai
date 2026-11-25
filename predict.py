"""Classify a document from the command line.

Usage:
    python predict.py samples/sample_invoice.txt
"""
import pickle
import sys

from utils import extract_text

if len(sys.argv) != 2:
    sys.exit("Usage: python predict.py <file.pdf|file.txt>")

with open("model.pkl", "rb") as f:
    model, vectorizer = pickle.load(f)

text = extract_text(sys.argv[1])
if not text.strip():
    sys.exit("No text could be extracted (supported: .pdf with a text layer, .txt)")

X = vectorizer.transform([text])
proba = model.predict_proba(X)[0]

print("Document type:", model.predict(X)[0])
for label, p in zip(model.classes_, proba):
    print(f"  {label:<8} {p * 100:.1f}%")
