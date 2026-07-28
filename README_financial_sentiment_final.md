## Financial PhraseBank Sentiment Analysis: An NLP-based Approach

This project involves natural language processing and sentiment classification of financial news sentences using the Financial PhraseBank dataset. The objective is to classify each sentence into positive, negative, or neutral sentiment, and deploy the final model as an interactive app for real-time predictions.

### Files

* `FINAL_FINANCIAL_SENTIMENT_ANALYSIS_PROJECT_4.ipynb` — Jupyter Notebook containing the entire workflow:

  * Text cleaning and preprocessing
  * Feature vectorization
  * Model training and evaluation

* `financial_phrasebank.csv` — The dataset used for analysis (expected to be in the same directory)
* `sentiment_model.pkl` — Serialized final model (Logistic Regression, class-weight balanced)
* `tfidf_vectorizer.pkl` — Serialized TF-IDF vectorizer, fit on the training set
* `sentiment_numbers.json` — Label mapping used to decode model predictions
* `app.py` — Streamlit app for real-time sentiment prediction on user-input text

---

### Features

* **Text preprocessing**, retaining numeric tokens (relevant for financial context) while cleaning the raw sentences.
* **Train/test split**, with all vectorizers fit on the training set only.
* **Feature vectorization** using Bag of Words (CountVectorizer) and TF-IDF (TfidfVectorizer).
* **Model training** using:

  * Naive Bayes
  * Logistic Regression (standard and class-weight balanced)
* **Class imbalance handling** via `class_weight='balanced'`.
* **Hyperparameter tuning** via GridSearchCV.
* **Model evaluation** using accuracy, macro F1-score, and per-class recall.
* **Deployment** as an interactive **Streamlit** web app: the serialized model, vectorizer, and label mapping are loaded to serve real-time predictions on user-entered financial text.

---

### Results

The notebook compares Naive Bayes and Logistic Regression across both vectorizers, with and without class balancing. The best-performing configuration, Logistic Regression (balanced) with TF-IDF features, achieved 67.9% accuracy, a 0.626 macro F1-score, and 50.6% recall on the minority negative-sentiment class. This model was serialized and deployed via a Streamlit app for interactive use.

### How to Run the App

```
streamlit run app.py
```

---

📌 Author

Sujan Ghosh
