# Real-Time Text Analyzer Using Machine Learning: Guide

## How to run
```
cd real-time-text-analyzer
pip install -r requirements.txt
python train_model.py
python app.py
```
Open http://127.0.0.1:5000

(`model.pkl`, `vectorizer.pkl` and `confusion_matrix.png` are already included in the zip. Running `train_model.py` recreates them.)

---

## Where each AIML concept is used

| # | Concept | Where / what it means in this project |
|---|---|---|
| 1 | Supervised Learning | The dataset has text AND the correct label (`sentiment`). The model learns from these labelled examples. |
| 2 | Classification | The output is a category (positive / negative / neutral), not a number. |
| 3 | Logistic Regression | `LogisticRegression` in `train_model.py`. It calculates a probability for each class and picks the highest. |
| 4 | Feature Engineering | `clean_text()` and `TfidfVectorizer` turn raw text into useful numeric features. |
| 5 | Training | `model.fit(X_train_tfidf, y_train)` on 80% of the data. |
| 6 | Testing | `model.predict(X_test_tfidf)` on the unseen 20% of the data. |
| 7 | Accuracy | Correct predictions / total predictions (`accuracy_score`). |
| 8 | Precision | Of the texts predicted as a class, how many really were that class (`precision_score`). |
| 9 | Recall | Of the texts that really belong to a class, how many were found (`recall_score`). |
| 10 | F1-Score | Balance of precision and recall (`f1_score`). |
| 11 | Confusion Matrix | Table of actual vs predicted classes (`confusion_matrix`). |
| 12 | Visualization | `confusion_matrix.png` is drawn with matplotlib. |

Training accuracy is also printed next to testing accuracy. If training accuracy is much higher than testing accuracy, that is overfitting.

---

## Result on my run (and an honest note)

757 records (252 positive, 252 negative, 253 neutral). Training 605 / Testing 152.
Accuracy, Precision, Recall and F1 were all 100% on the test set.

**Be ready to explain this in viva:** the dataset is small and was built from repeated sentence patterns, so the test sentences look very similar to the training sentences. That is why the score is perfect. It does not mean the model will be perfect on real-world text, such as long reviews, sarcasm or mixed opinions. Say this yourself before the examiner asks. If your teacher expects a more realistic score, add more varied, hand-written sentences to the CSV and retrain.

---

## Project Report Content

**1. Project Title**
Real-Time Text Analyzer Using Machine Learning

**2. Problem Statement**
Manually reading text to find its tone is slow. The project builds a system that counts words, characters and sentences and predicts the sentiment of text as the user types.

**3. Motivation**
Sentiment analysis is used in reviews, feedback and social media. A small real-time tool is a simple way to apply supervised learning on text.

**4. Objectives**
- Build a sentiment classifier using Logistic Regression and TF-IDF.
- Evaluate it using accuracy, precision, recall, F1-score and a confusion matrix.
- Provide a local web interface that gives results in real time.

**5. Proposed System**
A Flask web app with a saved ML model. The browser sends typed text to Flask, Flask returns the counts, the sentiment and the confidence, and the page updates without refreshing.

**6. Methodology**
Text, then Preprocessing, then TF-IDF, then Logistic Regression, then Sentiment (Positive / Negative / Neutral). The dataset is split 80% training and 20% testing. The trained model is saved with joblib.

**7. Technologies Used**
Python, Flask, HTML, CSS, JavaScript, Pandas, Scikit-learn, Joblib, Matplotlib (for the confusion matrix picture).

**8. Dataset Description**
A local CSV file `sentiment_dataset.csv` with two columns (`text`, `sentiment`). It has 757 short sentences, about 250 for each of the three classes. No internet or API is needed.

**9. Data Preprocessing**
Convert to lowercase, expand "n't" to "not", remove punctuation, remove extra spaces, and drop empty values.

**10. TF-IDF**
TF-IDF gives each word a number. Words that appear often in one sentence but rarely in the whole dataset get a high weight. This converts text into numbers that Logistic Regression can use. Word pairs (such as "not good") are also included.

**11. Logistic Regression**
A supervised classification algorithm. It computes a probability for each class (softmax for 3 classes) and predicts the class with the highest probability. That probability is shown as the confidence.

**12. Model Training**
80% of the data is vectorized with TF-IDF (`fit_transform`) and used to train `LogisticRegression`. The model and vectorizer are saved as `model.pkl` and `vectorizer.pkl`.

**13. Model Testing**
The remaining 20% of the data, which the model has never seen, is transformed with the same vectorizer and passed to the model for prediction.

**14. Evaluation Metrics**
Accuracy, Precision, Recall, F1-Score (macro average over the 3 classes) and a Confusion Matrix.

**15. Results**
The model reached 100% accuracy, precision, recall and F1-score on the test data of this small dataset. The confusion matrix has only diagonal values (no wrong predictions). The app returns a prediction within a fraction of a second.

**16. Advantages**
Simple, fast, works offline, easy to explain, real-time response, and the model is saved so it is not retrained at every start.

**17. Limitations**
- Small, pattern-based dataset, so results may not hold on real-world text.
- Cannot understand sarcasm or mixed opinions.
- Only English text.
- Words not in the vocabulary are ignored.

**18. Future Scope**
Use a larger real dataset, try SVM or Random Forest and compare, use cross-validation and hyperparameter tuning, and add more languages.

**19. Conclusion**
The project shows a complete beginner-level ML pipeline: dataset, preprocessing, TF-IDF, Logistic Regression, evaluation and real-time prediction in a local web app.

---

## 15 Viva Questions and Answers

**1. What is Artificial Intelligence?**
AI is making machines do tasks that normally need human intelligence, such as understanding language or making decisions.

**2. What is Machine Learning?**
Machine Learning is a part of AI where a computer learns patterns from data instead of being programmed with fixed rules.

**3. What is supervised learning?**
Learning from labelled data, where each input has a correct answer. Here, each sentence has a sentiment label.

**4. What is classification?**
Predicting a category for an input. Our model classifies text as positive, negative or neutral.

**5. Why did you use Logistic Regression?**
It is simple, fast, works well for text classification, and gives probabilities, which we show as confidence. It is also part of our syllabus.

**6. What is TF-IDF?**
Term Frequency-Inverse Document Frequency. It gives each word a weight based on how often it appears in a sentence and how rare it is across all sentences.

**7. Why do we convert text into numbers?**
Machine learning models can only do mathematics on numbers, so text must be converted into numeric features first.

**8. What is training data?**
The part of the data (80% here) that the model learns from.

**9. What is testing data?**
The part of the data (20% here) kept aside and not used in training. It checks how well the model works on unseen data.

**10. What is accuracy?**
Accuracy = correct predictions / total predictions.

**11. What is precision?**
Of all the texts predicted as one class (say positive), the fraction that really were that class.

**12. What is recall?**
Of all the texts that really belong to one class, the fraction the model correctly found.

**13. What is F1-score?**
The harmonic mean of precision and recall. It is one number that balances both.

**14. What is a confusion matrix?**
A table that compares actual classes with predicted classes. The diagonal shows correct predictions and the other cells show mistakes.

**15. How does your real-time analyzer work?**
When the user types, JavaScript waits 400 ms (debounce) and sends the text to Flask. Flask cleans the text, applies the saved TF-IDF vectorizer, predicts with the saved Logistic Regression model, and returns counts, sentiment and confidence. The page updates without reloading.

**Likely follow-up:** *"Why is your accuracy 100%?"* Because the dataset is small and its sentences follow similar patterns, so the test set is very similar to the training set. On real-world text, accuracy would be lower.
