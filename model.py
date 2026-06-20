# ============================================
# Mental Stress Predictor — Improved Version
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
import pickle
import os
import scipy.sparse as sp

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('vader_lexicon')  

from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)


# ================================================
# STEP 1: Load Data
# ================================================

df = pd.read_csv("Stress.csv")
print("Shape:", df.shape)
print(df.head(3))
print(df.dtypes)
print(df.isnull().sum())


# ================================================
# STEP 2: EDA — Visualizations
# ================================================

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 5)

# Label distribution — bar + pie
label_counts = df['label'].value_counts()
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].bar(
    ['No Stress (0)', 'Stress (1)'],
    label_counts.values,
    color=['#1D9E75', '#E24B4A'],
    edgecolor='white', width=0.5
)
axes[0].set_title('Label Distribution', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Number of posts')
for i, v in enumerate(label_counts.values):
    axes[0].text(i, v + 10, str(v), ha='center', fontweight='bold')

axes[1].pie(
    label_counts.values,
    labels=['No Stress (0)', 'Stress (1)'],
    autopct='%1.1f%%',
    colors=['#1D9E75', '#E24B4A'],
    startangle=90,
    wedgeprops={'edgecolor': 'white', 'linewidth': 2}
)
axes[1].set_title('Label Balance', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('label_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# Stress rate per subreddit
subreddit_stress = df.groupby('subreddit')['label'].mean() * 100
subreddit_stress = subreddit_stress.sort_values(ascending=True)
colors = ['#E24B4A' if x > 55 else '#EF9F27' if x > 45 else '#1D9E75'
          for x in subreddit_stress.values]

plt.figure(figsize=(10, 6))
bars = plt.barh(subreddit_stress.index, subreddit_stress.values, color=colors, edgecolor='white')
for bar, val in zip(bars, subreddit_stress.values):
    plt.text(val + 0.5, bar.get_y() + bar.get_height()/2,
             f'{val:.1f}%', va='center', fontweight='bold', fontsize=10)
plt.axvline(x=50, color='gray', linestyle='--', alpha=0.7, label='50% line')
plt.xlabel('Stress Rate (%)')
plt.title('Stress Rate per Subreddit', fontsize=14, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('subreddit_stress.png', dpi=150, bbox_inches='tight')
plt.close()

# Text length distribution
df['text_length'] = df['text'].apply(len)
print(f"Avg length: {df['text_length'].mean():.0f}")
print(f"Stressed avg: {df[df['label']==1]['text_length'].mean():.0f}")
print(f"Non-stressed avg: {df[df['label']==0]['text_length'].mean():.0f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].hist(df[df['label']==1]['text_length'], bins=40, color='#E24B4A', alpha=0.7, edgecolor='white')
axes[0].axvline(df[df['label']==1]['text_length'].mean(), color='darkred', linestyle='--', linewidth=2, label='Mean')
axes[0].set_title('Stressed Post Length', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Characters')
axes[0].set_ylabel('Count')
axes[0].legend()

axes[1].hist(df[df['label']==0]['text_length'], bins=40, color='#1D9E75', alpha=0.7, edgecolor='white')
axes[1].axvline(df[df['label']==0]['text_length'].mean(), color='darkgreen', linestyle='--', linewidth=2, label='Mean')
axes[1].set_title('Non-Stressed Post Length', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Characters')
axes[1].set_ylabel('Count')
axes[1].legend()
plt.tight_layout()
plt.savefig('text_length.png', dpi=150, bbox_inches='tight')
plt.close()

# Annotation confidence distribution
plt.figure(figsize=(9, 4))
df['confidence'].hist(bins=20, color='#378ADD', alpha=0.8, edgecolor='white')
plt.axvline(x=0.6, color='red', linestyle='--', linewidth=2, label='Threshold (0.6)')
plt.title('Annotation Confidence Distribution', fontsize=13, fontweight='bold')
plt.xlabel('Confidence Score')
plt.ylabel('Number of posts')
plt.legend()
plt.tight_layout()
plt.savefig('confidence.png', dpi=150, bbox_inches='tight')
plt.close()

low_conf = df[df['confidence'] < 0.6]
print(f"Low confidence samples: {len(low_conf)} ({len(low_conf)/len(df)*100:.1f}%)")

# NEW: Sentiment score distribution — visualize happy vs stressed posts
sia = SentimentIntensityAnalyzer()
df['sentiment'] = df['text'].apply(lambda x: sia.polarity_scores(x)['compound'])

plt.figure(figsize=(10, 4))
plt.hist(df[df['label']==1]['sentiment'], bins=40, color='#E24B4A', alpha=0.7,
         edgecolor='white', label='Stressed')
plt.hist(df[df['label']==0]['sentiment'], bins=40, color='#1D9E75', alpha=0.7,
         edgecolor='white', label='Not Stressed')
plt.axvline(x=0, color='black', linestyle='--', alpha=0.6, label='Neutral (0)')
plt.title('Sentiment Score Distribution by Label', fontsize=13, fontweight='bold')
plt.xlabel('VADER Compound Sentiment Score  (-1=Negative, +1=Positive)')
plt.ylabel('Count')
plt.legend()
plt.tight_layout()
plt.savefig('sentiment_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\nAvg sentiment — Stressed:     {df[df['label']==1]['sentiment'].mean():.3f}")
print(f"Avg sentiment — Not Stressed: {df[df['label']==0]['sentiment'].mean():.3f}")


# ================================================
# STEP 3: Preprocessing
# ================================================

df_original = df.copy()
# REMOVED: 'sentence_range', 'social_timestamp' dropped
# KEPT: subreddit for EDA only, not for features in prediction
df = df.drop(columns=['post_id', 'sentence_range', 'social_timestamp'])

stop_words = set(stopwords.words('english'))


def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)     # remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)         # keep letters only
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)                 # remove extra spaces
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]
    return ' '.join(words)


print("Cleaning text...")
df['clean_text'] = df['text'].apply(clean_text)
print("Done!")

# Remove low-confidence samples
df_filtered = df[df['confidence'] >= 0.6].copy()
print(f"Removed {len(df) - len(df_filtered)} low-confidence samples")
print(f"Remaining: {len(df_filtered)}")

df_clean = df_filtered.copy()

# Compute sentiment on original text (before cleaning, so VADER works better)
df_clean['sentiment'] = df_clean['text'].apply(
    lambda x: sia.polarity_scores(x)['compound']
)
print("Sentiment scores computed.")


# ================================================
# STEP 4: Feature Engineering
# ================================================

# TF-IDF: convert text to numeric features
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),   # unigrams + bigrams
    min_df=2,             # ignore very rare words
    sublinear_tf=True     # log-scale term frequency
)
X_tfidf = tfidf.fit_transform(df_clean['clean_text'])
print(f"TF-IDF shape: {X_tfidf.shape}")

# NEW: Sentiment score feature (replaces text_length — more meaningful)
# VADER compound score: -1 (very negative) to +1 (very positive)
# Negative score strongly indicates stress
scaler = MinMaxScaler()
X_sentiment = scaler.fit_transform(df_clean['sentiment'].values.reshape(-1, 1))
X_sentiment_sparse = sp.csr_matrix(X_sentiment)
print(f"Sentiment feature shape: {X_sentiment_sparse.shape}")

# Combine: TF-IDF + Sentiment only
X = sp.hstack([X_tfidf, X_sentiment_sparse])

y = df_clean['label'].values
print(f"Final X shape: {X.shape}")
print(f"y shape: {y.shape}")


# ================================================
# STEP 5: Train-Test Split
# ================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y          # keep class ratio same in both splits
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Train stress rate: {y_train.mean()*100:.1f}%")
print(f"Test stress rate:  {y_test.mean()*100:.1f}%")

# Same split for TF-IDF only (used by Naive Bayes — cannot use negative values)
X_tfidf_train, X_tfidf_test, _, _ = train_test_split(
    X_tfidf, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ================================================
# STEP 6: Model Training
# ================================================

# --- Logistic Regression ---
print("\nTraining Logistic Regression...")
lr_model = LogisticRegression(
    C=1.0,           # regularization strength
    max_iter=1000,   # enough iterations to converge on text data
    solver='saga',   # best solver for large sparse data
    random_state=42
)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
y_prob_lr = lr_model.predict_proba(X_test)[:, 1]
print("Logistic Regression trained.")



# --- Naive Bayes ---
print("\nTraining Naive Bayes...")
nb_model = MultinomialNB(alpha=0.1)   # alpha = Laplace smoothing
nb_model.fit(X_tfidf_train, y_train)
y_pred_nb = nb_model.predict(X_tfidf_test)
y_prob_nb = nb_model.predict_proba(X_tfidf_test)[:, 1]
print("Naive Bayes trained.")
# Save lr model bundle — subreddit_cols REMOVED, scaler now for sentiment
bundle = {
    'model':   lr_model,
    'tfidf':   tfidf,
    'scaler':  scaler,   # now scales sentiment score (not text length)
}

with open('stress_model.pkl', 'wb') as f:
    pickle.dump(bundle, f)
print("LR Model saved!")
bundle_nb = {
    'model': nb_model,
    'tfidf': tfidf,
}
with open('stress_model_nb.pkl', 'wb') as f:
    pickle.dump(bundle_nb, f)
print("NB Model saved!")
# ================================================
# STEP 7: Metrics Calculation
# ================================================

def compute_metrics(y_true, y_pred, y_prob, model_name):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)
    auc  = roc_auc_score(y_true, y_prob)

    print(f"\n{'='*45}")
    print(f"  {model_name} — Metrics")
    print(f"{'='*45}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['No Stress', 'Stress']))

    return {'model': model_name, 'accuracy': acc, 'precision': prec,
            'recall': rec, 'f1': f1, 'roc_auc': auc}


metrics_lr = compute_metrics(y_test, y_pred_lr, y_prob_lr, "Logistic Regression")
metrics_nb = compute_metrics(y_test, y_pred_nb, y_prob_nb, "Naive Bayes")


# ================================================
# STEP 8: Confusion Matrix — Both Models
# ================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
models_info = [
    (y_pred_lr, "Logistic Regression", axes[0]),
    (y_pred_nb, "Naive Bayes",         axes[1])
]

for y_pred, name, ax in models_info:
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', ax=ax,
        xticklabels=['No Stress', 'Stress'],
        yticklabels=['No Stress', 'Stress'],
        linewidths=0.5
    )
    ax.set_title(f'{name}\nConfusion Matrix', fontsize=13, fontweight='bold')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()


# ================================================
# STEP 9: ROC Curve — Both Models
# ================================================

fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_nb, tpr_nb, _ = roc_curve(y_test, y_prob_nb)

plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, color='#378ADD', linewidth=2,
         label=f"Logistic Regression (AUC = {metrics_lr['roc_auc']:.3f})")
plt.plot(fpr_nb, tpr_nb, color='#E24B4A', linewidth=2,
         label=f"Naive Bayes (AUC = {metrics_nb['roc_auc']:.3f})")
plt.plot([0,1], [0,1], 'k--', alpha=0.5, label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — Model Comparison', fontsize=14, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150, bbox_inches='tight')
plt.close()


# ================================================
# STEP 10: Metric Bar Chart — Side-by-Side Comparison
# ================================================

metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
lr_scores = [metrics_lr['accuracy'], metrics_lr['precision'],
             metrics_lr['recall'],   metrics_lr['f1'],
             metrics_lr['roc_auc']]
nb_scores = [metrics_nb['accuracy'], metrics_nb['precision'],
             metrics_nb['recall'],   metrics_nb['f1'],
             metrics_nb['roc_auc']]

x = np.arange(len(metric_names))
width = 0.35

fig, ax = plt.subplots(figsize=(11, 6))
bars_lr = ax.bar(x - width/2, lr_scores, width, label='Logistic Regression',
                 color='#378ADD', edgecolor='white')
bars_nb = ax.bar(x + width/2, nb_scores, width, label='Naive Bayes',
                 color='#E24B4A', edgecolor='white')

for bar in bars_lr:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar in bars_nb:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylim(0, 1.1)
ax.set_ylabel('Score')
ax.set_title('Model Comparison — All Metrics', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metric_names)
ax.legend()
ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.4, label='0.9 reference')
plt.tight_layout()
plt.savefig('metric_comparison.png', dpi=150, bbox_inches='tight')
plt.close()


# ================================================
# STEP 11: Best Metric per Model — Analysis
# ================================================

print("\n" + "="*55)
print("  WHICH METRIC IS MOST EFFECTIVE — ANALYSIS")
print("="*55)

summary = pd.DataFrame([metrics_lr, metrics_nb]).set_index('model')
print("\nSummary Table:")
print(summary.round(4))

print("\nWinner per Metric:")
for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
    if metrics_lr[metric] > metrics_nb[metric]:
        winner = "Logistic Regression"
        diff = metrics_lr[metric] - metrics_nb[metric]
    else:
        winner = "Naive Bayes"
        diff = metrics_nb[metric] - metrics_lr[metric]
    print(f"  {metric:<12} → {winner} (by {diff:.4f})")

print("""
Most Effective Metric for This Problem:
-----------------------------------------
F1 Score is the most effective metric here because:
  1. Dataset is slightly imbalanced (52% stress, 48% no-stress)
  2. F1 balances Precision and Recall equally
  3. Both false positives (wrong stress alert) and
     false negatives (missed stress) are harmful

ROC-AUC is second most important because:
  It measures how well the model separates both classes
  across ALL thresholds, not just 0.5

Accuracy alone is misleading on imbalanced data —
  a model predicting all 1s would get 52% accuracy.

Recommendation:
  Use F1 Score as primary metric.
  Use ROC-AUC as secondary for threshold tuning.
""")

print("="*55)
if metrics_lr['f1'] > metrics_nb['f1']:
    print(f"  BEST MODEL (by F1): Logistic Regression")
    print(f"  LR F1  = {metrics_lr['f1']:.4f}")
    print(f"  NB F1  = {metrics_nb['f1']:.4f}")
else:
    print(f"  BEST MODEL (by F1): Naive Bayes")
    print(f"  NB F1  = {metrics_nb['f1']:.4f}")
    print(f"  LR F1  = {metrics_lr['f1']:.4f}")
print("="*55)


# ================================================
# STEP 12: predict_stress() — called by gui.py
# ================================================
#   - Sentiment score  (detects happy/sad correctly)
# ================================================

_bundle = None


def _load_bundle():
    global _bundle
    if _bundle is None:
        with open('stress_model.pkl', 'rb') as f:
            _bundle = pickle.load(f)


def predict_stress(text: str):
    """
    Predicts stress from raw input text.
    Returns: (label: str, confidence: float)
      label      → 'Stressed' or 'Not Stressed'
      confidence → probability of predicted class (0.0 to 1.0)
    """
    _load_bundle()

    # Step A: clean text for TF-IDF
    cleaned = clean_text(text)

    # Step B: TF-IDF features
    X_text = _bundle['tfidf'].transform([cleaned])

    # Step C: Sentiment score (on original text — better for VADER)
    sentiment_score = sia.polarity_scores(text)['compound']
    X_sent = sp.csr_matrix(
        _bundle['scaler'].transform([[sentiment_score]])
    )

    # Step D: Combine features (TF-IDF + Sentiment only)
    X_in = sp.hstack([X_text, X_sent])

    # Step E: Predict
    probs = _bundle['model'].predict_proba(X_in)[0]

    # Step F: Threshold = 0.50 (FIXED from 0.45)
    threshold = 0.50

    if probs[1] >= threshold:
        return "Stressed", float(probs[1])
    else:
        return "Not Stressed", float(probs[0])
_bundle_nb = None

def _load_bundle_nb():
    global _bundle_nb
    if _bundle_nb is None:
        with open('stress_model_nb.pkl', 'rb') as f:
            _bundle_nb = pickle.load(f)

def predict_stress_nb(text: str):
    _load_bundle_nb()
    
    cleaned = clean_text(text)
    X_text = _bundle_nb['tfidf'].transform([cleaned])
    
    probs = _bundle_nb['model'].predict_proba(X_text)[0]
    
    threshold = 0.50
    if probs[1] >= threshold:
        return "Stressed", float(probs[1])
    else:
        return "Not Stressed", float(probs[0])