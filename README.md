 
# Amazon Fine Food Reviews — NLP Sentiment Analysis Pipeline

## Overview
This repository contains an end-to-end **Natural Language Processing (NLP)** pipeline for **binary sentiment analysis** on Amazon Fine Food Reviews. It transforms raw review text into machine-readable features via **negation-aware** preprocessing, compares multiple representations (**Bag-of-Words, TF-IDF, Word2Vec**), and benchmarks classic classifiers (**Logistic Regression, Multinomial Naive Bayes, Random Forest**) with **cross-validated** hyperparameter tuning.  
**Best model (LogReg + TF-IDF)** achieves **~0.91 Accuracy** and **~0.91 Macro-F1** on a held-out test split.

## Key Features
- **Clean labeling:** Neutral reviews removed; stars → binary sentiment (positive/negative)
- **Data hygiene:** Duplicate removal and helpfulness consistency checks
- **Negation-aware preprocessing:** preserves cues like “not good”, “didn’t like”
- **Rich representations:** **BoW**, **TF-IDF (uni/bi-grams)**, **Word2Vec** sentence embeddings
- **Model selection:** **GridSearchCV** over Logistic Regression; baselines with MNB and RF
- **Transparent metrics:** Accuracy, Precision, Recall, F1 + qualitative word-clouds

## Dataset
- **Source:** Amazon Fine Food Reviews (`Reviews.csv`)
- **Target construction:**  
  - Drop neutral (`Score == 3`)  
  - Map `Score > 3 → 1 (positive)`, `Score < 3 → 0 (negative)`
- **Sanity checks:**  
  - De-duplicate on `{UserId, ProfileName, Time, Text}`  
  - Remove rows with `HelpfulnessNumerator > HelpfulnessDenominator`

## Methodology
### 1) Preprocessing (negation-aware)
- Remove **URLs/HTML**, **de-contract** (e.g., “don’t” → “do not”)
- Remove digits/non-alphanumerics, lowercase
- **Stopwords:** drop general stopwords but **keep negations** (“not”, “no”, “n’t”)
- Output column: `preprocessed_text`

### 2) Class balance & split
- Balance labels using **RandomUnderSampler**
- **Train/Test split: 80/20**  
- Fit vectorizers on **training** data; transform test data

### 3) Vectorization
- **BoW:** `CountVectorizer`
- **TF-IDF:** `TfidfVectorizer(ngram_range=(1,2))`
- **Word2Vec:** gensim model (e.g., 50-dim) trained on corpus; sentence vectors = mean of word embeddings

### 4) Modeling & tuning
- **Logistic Regression:** `GridSearchCV` (10-fold) over `C`, `penalty='l2'`, `solver='lbfgs'`
- **Multinomial Naive Bayes:** on BoW/TF-IDF
- **Random Forest:** on BoW, TF-IDF, and Word2Vec embeddings

### 5) Evaluation & visualization
- Report **Accuracy, Precision, Recall, F1** (per class + macro)
- **Word clouds** for positive vs negative subsets

## Results (summary)
| Model                    | Features | Accuracy | Macro-F1 |
|-------------------------|----------|----------|----------|
| Logistic Regression     | TF-IDF   | ~0.91    | ~0.91    |
| Multinomial Naive Bayes | BoW      | ~0.866   | ~0.86    |
| Random Forest           | TF-IDF   | ~0.877   | ~0.87    |
| Random Forest           | Word2Vec | ~0.593   | ~0.59    |

**Key takeaway:** **TF-IDF + Logistic Regression** captures salient **n-grams** (including negation patterns) effectively, delivering the strongest performance on sparse text.

## How to Run
### Option A: Google Colab
1. Open the notebook and connect to runtime.  
2. Mount Drive (if using `Reviews.csv` from Drive).  
3. Run all cells in order.

### Option B: Local
```bash
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt                      # scikit-learn, nltk, gensim, bs4, imbalanced-learn
jupyter notebook                                     # open the notebook and Run All
