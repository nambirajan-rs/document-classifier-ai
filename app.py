import os
import pickle
import tempfile

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from utils import extract_text

ALLOWED = {".pdf", ".txt"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

with open("model.pkl", "rb") as f:
    model, vectorizer = pickle.load(f)


@app.route("/", methods=["GET", "POST"])
def home():
    result, error = None, None

    if request.method == "POST":
        file = request.files.get("file")
        name = secure_filename(file.filename) if file else ""
        ext = os.path.splitext(name)[1].lower()

        if not name:
            error = "Please choose a file."
        elif ext not in ALLOWED:
            error = "Only PDF and TXT files are supported."
        else:
            # Save to a temp file, classify, then delete: uploads are never kept.
            fd, path = tempfile.mkstemp(suffix=ext)
            os.close(fd)
            try:
                file.save(path)
                text = extract_text(path)
            finally:
                os.remove(path)

            if not text.strip():
                error = "No text found. Scanned PDFs need OCR first."
            else:
                X = vectorizer.transform([text])
                proba = model.predict_proba(X)[0]
                scores = sorted(zip(model.classes_, proba), key=lambda s: -s[1])
                result = {
                    "filename": name,
                    "label": scores[0][0],
                    "scores": [(label, round(p * 100, 1)) for label, p in scores],
                }

    return render_template("index.html", result=result, error=error)


if __name__ == "__main__":
    app.run(debug=True)
