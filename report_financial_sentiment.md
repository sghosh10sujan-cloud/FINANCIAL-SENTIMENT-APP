Project Report : Financial PhraseBank Sentiment Analysis

Problem Statement :
Financial news sentences carry sentiment that can move markets, but manually reading and classifying large volumes of financial text does not scale. This project aims to build a text classification model that automatically labels financial news sentences as positive, negative, or neutral in sentiment.

Project Overview :
A natural language processing pipeline was developed to clean, vectorize, and classify 5,842 financial news sentences from the Financial PhraseBank dataset. The project compared multiple vectorization schemes and classifiers, and specifically addressed class imbalance, since financial sentiment datasets are typically dominated by neutral statements.

Dataset Summary :
The dataset (Financial PhraseBank) contains 5,842 financial news sentences, each labeled with one of three sentiment classes: positive, negative, or neutral. Numeric tokens were deliberately retained during text cleaning, since numbers (e.g., percentage changes, monetary figures) carry meaningful signal in financial text rather than being noise to strip out. The dataset was split into training and test sets before any vectorizer was fit.

Exploratory Data Analysis (EDA Highlights) :

• The dataset showed a class imbalance typical of financial sentiment corpora, with neutral statements forming a large share of sentences and negative statements forming the smallest class.
• Both Bag-of-Words (CountVectorizer) and TF-IDF vectorization were explored to compare a simple frequency-based representation against a weighted one.

Model Development :
Two vectorization schemes (BoW, TF-IDF) were combined with two classifiers (Naive Bayes, Logistic Regression), with vectorizers fit on the training set only and applied to the test set via transform. To address class imbalance, a class-weight-balanced variant of Logistic Regression was also trained, and hyperparameters were tuned via GridSearchCV.

The best-performing configuration — Logistic Regression (class-weight balanced) with TF-IDF features — achieved :

• Accuracy: 67.9%
• Macro F1-score: 0.626
• Recall (negative class): 50.6%

Evaluation Metric :
Macro F1-score and per-class recall were prioritized over plain accuracy, since accuracy alone would be misleading on an imbalanced three-class problem (a model that always predicted "neutral" could still score deceptively well). Recall on the negative class specifically was tracked, since correctly catching negative financial sentiment has the most practical value for risk monitoring.

Challenges :

• Class imbalance across three classes made the minority (negative) class harder to detect than a simple binary imbalance problem — class-weighting was needed rather than accuracy alone as a guide.

• Choosing between BoW and TF-IDF required empirical comparison rather than assumption, since the two can perform differently depending on text length and vocabulary size.

• Financial language is domain-specific (e.g., "loss" and "decline" can appear in neutral factual statements), which limits how much generic sentiment heuristics transfer.

Impact :

This tool can :

• Serve as a lightweight sentiment-tagging layer for financial news feeds

• Help analysts triage large volumes of financial text by sentiment before manual review

• Serve as a template for extending to other domain-specific text classification problems

Tech Stack :

• Python

• Scikit-learn

• Pandas, NumPy

• Matplotlib, Seaborn


Future Work :

• Explore transformer-based embeddings (e.g., FinBERT) as a stronger alternative to BoW/TF-IDF
• Expand training data with additional labeled financial text sources to improve minority-class recall
• Add explainability (e.g., top contributing tokens per prediction) for analyst trust
• Deploy as an API or lightweight dashboard for real-time sentence scoring
