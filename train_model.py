"""
train_model.py
Trains a sentiment classifier:  Text -> Preprocessing -> TF-IDF -> Logistic Regression
Run once:  python train_model.py
"""
import re
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # draw plots without opening a window
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay)


def clean_text(text):
    """Basic preprocessing: lowercase, remove punctuation, remove extra spaces."""
    if not isinstance(text, str):          # handle empty / missing values
        return ""
    text = text.lower()                    # 1. lowercase
    text = text.replace("n't", " not")     # "don't" -> "do not" (keeps the meaning of negation)
    text = re.sub(r"[^a-z0-9\s]", " ", text)   # 2. remove punctuation
    text = re.sub(r"\s+", " ", text).strip()   # 3. remove extra spaces
    return text


def main():
    # ---- 1. Load the local dataset and preprocess ----
    df = pd.read_csv("sentiment_dataset.csv")
    df = df.dropna()                                   # drop rows with missing values
    df["clean"] = df["text"].apply(clean_text)
    df = df[df["clean"] != ""]
    print("Total records:", len(df))

    # ---- 2. Split: 80% training, 20% testing ----
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["sentiment"], test_size=0.2, random_state=42, stratify=df["sentiment"])

    # ---- 3. Feature extraction: TF-IDF (text -> numbers) ----
    # ngram_range=(1, 2) also looks at word pairs such as "not good"
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)  # learn vocabulary from training data only
    X_test_tfidf = vectorizer.transform(X_test)

    # ---- 4. Train Logistic Regression ----
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    # ---- 5. Evaluate ----
    train_acc = accuracy_score(y_train, model.predict(X_train_tfidf))
    y_pred = model.predict(X_test_tfidf)
    print(f"\nTraining accuracy: {train_acc:.2%}")
    print(f"Testing accuracy : {accuracy_score(y_test, y_pred):.2%}")
    print(f"Precision        : {precision_score(y_test, y_pred, average='macro'):.2%}")
    print(f"Recall           : {recall_score(y_test, y_pred, average='macro'):.2%}")
    print(f"F1-Score         : {f1_score(y_test, y_pred, average='macro'):.2%}")

    labels = ["positive", "negative", "neutral"]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print("\nConfusion Matrix (rows = actual, columns = predicted):")
    print(labels)
    print(cm)

    # ---- 6. Confusion matrix picture ----
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(cmap="Blues")
    plt.title("Confusion Matrix (Test Data)")
    plt.savefig("confusion_matrix.png", bbox_inches="tight")
    print("\nSaved confusion_matrix.png")

    # ---- 7. Save model and vectorizer so the app does not need to retrain ----
    joblib.dump(model, "model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")
    print("Saved model.pkl and vectorizer.pkl")


if __name__ == "__main__":
    main()
