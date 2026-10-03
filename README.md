# Real-Time Text Analyzer Using Machine Learning

A small AIML mini-project. The user types text in a web page and the system shows, in real time:
word count, character count, sentence count, **sentiment** (Positive / Negative / Neutral) and **confidence**.

**Pipeline:** Text → Preprocessing → TF-IDF → Logistic Regression → Sentiment

---

## 1. Project Structure

| File | Purpose |
|---|---|
| `sentiment_dataset.csv` | Local dataset (`text`, `sentiment`), 757 rows |
| `train_model.py` | Preprocess, TF-IDF, train Logistic Regression, evaluate, save model |
| `app.py` | Flask server: receives text, returns counts + sentiment |
| `model.pkl` | Saved Logistic Regression model (created by `train_model.py`) |
| `vectorizer.pkl` | Saved TF-IDF vectorizer (created by `train_model.py`) |
| `confusion_matrix.png` | Confusion matrix picture (created by `train_model.py`) |
| `templates/index.html` | Web page |
| `static/style.css` | Page styling |
| `static/script.js` | Sends typed text to Flask (with 400 ms debounce) and updates the page |
| `requirements.txt` | Python libraries needed |

## 2. How to Run

```
pip install -r requirements.txt
python train_model.py
python app.py
```
Open **http://127.0.0.1:5000** in the browser. (`model.pkl` and `vectorizer.pkl` are already included; running `train_model.py` re-creates them.)

---

## 3. Which parameters decide Positive, Negative or Neutral?

The model does **not** use hand-written rules. It learns from the dataset. Each word (and each pair of words) gets a **weight** for each class. The class with the highest total score wins, and that score is converted to a probability (the confidence).

### 3.1 Words and phrases that push the model toward each class

These are the highest-weighted features taken from the trained model.

| Sentiment | Strongest words / phrases (learned weight) | Meaning |
|---|---|---|
| **Positive** | happy (1.53), excellent (1.36), impressive (1.24), very happy (1.18), great (1.16), superb (1.14), awesome (1.13), fantastic (1.07), wonderful (1.01), brilliant (0.96) | Praise and happiness words |
| **Negative** | not (2.09), very upset (1.26), upset (1.26), horrible (1.23), do not (1.20), boring (1.15), awful (1.14), annoying (1.11), is bad (0.83), pathetic (0.83) | Complaint words and negation (`not`, `do not`; "don't" is changed to "do not" in preprocessing) |
| **Neutral** | nothing (1.59), nothing more (1.26), overall (1.22), just another (1.07), is just (1.07), so (1.07), the (1.74), have (1.65) | Plain, matter-of-fact wording with no strong emotion |

> **Note:** Neutral weights such as `the` and `have` are partly an effect of the small, pattern-based dataset (many neutral sentences look like "The product is ..." or "I have used ..."). Real neutral words are `okay`, `average`, `nothing special`. This is a limitation of the dataset, not of the method.

### 3.2 Parameters of the pipeline

| Stage | Parameter / setting | Value used | Why |
|---|---|---|---|
| Preprocessing | Lowercase | Yes | "Great" and "great" become the same word |
| Preprocessing | Expand `n't` → ` not` | Yes | Keeps the negative meaning of "don't", "can't" |
| Preprocessing | Remove punctuation and extra spaces | Yes | Removes noise |
| Preprocessing | Empty / missing text | Skipped | Nothing to predict |
| TF-IDF | `ngram_range` | (1, 2) | Reads single words AND word pairs such as "not good" |
| TF-IDF | Stop-word removal | Not used | Words like "not" must stay, they change the meaning |
| TF-IDF | Vocabulary size | 632 features | Learned from the training data only |
| Logistic Regression | `max_iter` | 1000 | Enough iterations to finish learning |
| Logistic Regression | Classes | positive, negative, neutral | 3-class classification |
| Data split | Train / Test | 80% / 20% (`random_state=42`, stratified) | Each class is equally represented in both sets |

### 3.3 How the final answer is chosen

| Step | What happens |
|---|---|
| 1 | The text is cleaned and converted to TF-IDF numbers |
| 2 | Logistic Regression adds up the weights of the words present, one score per class |
| 3 | The 3 scores are converted into 3 probabilities that add up to 100% |
| 4 | The class with the **highest probability** is the **Sentiment** |
| 5 | That highest probability is shown as the **Confidence** |

Example: "I don't like this" → cleaned to "i do not like this" → words `not`, `do not` push strongly to **Negative** → Negative 91%.

### 3.4 Quick guide: what makes each sentiment more likely

| To get... | Text usually contains | Example input | Result seen in testing |
|---|---|---|---|
| **Positive** | happy, excellent, great, amazing, wonderful, "love", "enjoyed" | "I really enjoyed this movie. It was amazing." | Positive, 82% |
| **Negative** | hate, awful, horrible, boring, bad, "not", "don't" | "I don't like this" | Negative, 91% |
| **Neutral** | okay, average, "nothing special", "works as described", "just another" | "It is okay, nothing special" | Neutral, 83% |
| **No result** | Empty text or only spaces | (empty) | Sentiment `-`, 0% |

A **low confidence** (for example 40–60%) means the sentence mixes words from more than one class, or contains many words the model has never seen.

---

## 4. How the Counts Are Calculated

| Value | Rule |
|---|---|
| Words | Text split on spaces |
| Characters | Total length of the text (including spaces) |
| Sentences | Text split on `.`, `!`, `?`, empty parts ignored |

## 5. Model Evaluation (shown after `python train_model.py`)

| Metric | Meaning | Result on test data |
|---|---|---|
| Accuracy | Correct predictions / all predictions | 100% |
| Precision | Of texts predicted as a class, how many were correct | 100% |
| Recall | Of texts truly in a class, how many were found | 100% |
| F1-Score | Balance of precision and recall | 100% |
| Confusion Matrix | Actual vs predicted table (saved as `confusion_matrix.png`) | Only diagonal values, no mistakes |

> **Why 100%?** The dataset is small and made of repeated sentence patterns, so the test sentences are very similar to the training sentences. This does **not** mean the model is perfect on real-world text (sarcasm, long reviews, mixed opinions). Adding more varied sentences to `sentiment_dataset.csv` and retraining gives a more realistic score.

## 6. AIML Concepts Used

| Concept | Where |
|---|---|
| Supervised Learning | Dataset has text + correct label |
| Classification | Output is a category (3 classes) |
| Feature Engineering | Preprocessing + TF-IDF |
| Logistic Regression | Main algorithm in `train_model.py` |
| Training / Testing | 80% / 20% split |
| Accuracy, Precision, Recall, F1 | Printed after training |
| Confusion Matrix + Visualization | `confusion_matrix.png` |

## 7. Technologies

Python, Flask, HTML, CSS, JavaScript, Pandas, Scikit-learn, Joblib, Matplotlib (for the confusion matrix picture only).

## 8. Limitations

- Small, pattern-based dataset.
- English only.
- Cannot detect sarcasm or mixed opinions.
- Words not seen during training are ignored.
