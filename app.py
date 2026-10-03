"""
app.py
Flask web app: receives text from the browser and returns counts + sentiment.
Run:  python app.py   then open http://127.0.0.1:5000
"""
import re
import joblib
from flask import Flask, render_template, request, jsonify
from train_model import clean_text      # reuse the same preprocessing used in training

app = Flask(__name__)

# Load the saved model and TF-IDF vectorizer (created by train_model.py)
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


def count_text(text):
    """Return word, character and sentence counts."""
    words = len(text.split())
    characters = len(text)
    sentences = len([s for s in re.split(r"[.!?]+", text) if s.strip()])
    return words, characters, sentences


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    text = request.get_json().get("text", "")
    words, characters, sentences = count_text(text)

    sentiment, confidence = "-", 0
    cleaned = clean_text(text)
    if cleaned:                                       # predict only if there is some text
        features = vectorizer.transform([cleaned])    # text -> TF-IDF numbers
        sentiment = model.predict(features)[0]        # Logistic Regression prediction
        confidence = round(model.predict_proba(features).max() * 100)

    return jsonify({"words": words, "characters": characters, "sentences": sentences,
                    "sentiment": sentiment.upper(), "confidence": confidence})


if __name__ == "__main__":
    app.run(debug=True)
