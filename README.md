# Mental Stress Predictor

A Machine Learning and Natural Language Processing (NLP) project that predicts whether a piece of text reflects a stressed or non-stressed mental state.

The system uses text preprocessing, sentiment analysis, TF-IDF feature extraction, and machine learning algorithms to provide accurate stress predictions through an interactive desktop application.

---

## Project Overview

Mental stress has become a major concern in modern life. This project aims to automatically detect stress-related patterns in textual content using Machine Learning techniques.

Users can enter any text into the application, and the system predicts whether the text indicates stress along with a confidence score.

---

## Features

- User-friendly desktop interface
- Stress prediction from text input
- Confidence score visualization
- Sentiment-aware classification
- Fast real-time predictions
- Machine Learning powered
- Clean modern UI using Tkinter
- Offline functionality

---

## Technologies Used

### Programming Language
- Python

### Machine Learning
- Logistic Regression
- Naive Bayes

### Natural Language Processing
- NLTK
- VADER Sentiment Analysis
- TF-IDF Vectorization

### GUI Development
- Tkinter

### Data Analysis
- Pandas
- NumPy

### Visualization
- Matplotlib
- Seaborn

---

## Dataset Processing

The dataset undergoes several preprocessing steps:

- Text Cleaning
- Lowercase Conversion
- Stopword Removal
- URL Removal
- Punctuation Removal
- Sentiment Score Extraction
- TF-IDF Feature Generation

---

## Prediction Workflow

User Text

↓

Text Cleaning

↓

TF-IDF Transformation

↓

Sentiment Analysis

↓

Machine Learning Model

↓

Stress Prediction

↓

Confidence Score

---

## Application Preview

### Main Screen
- Enter text
- Click Analyze Text
- View prediction result

### Prediction Result
- Stress Status
- Confidence Percentage

(Add screenshots here)

---

## Installation

### Clone Repository

```bash
git clone https://github.com/noormangi17/Mental-Stress-Predictor.git
```

### Open Project

```bash
cd Mental-Stress-Predictor
```

### Install Requirements

```bash
pip install -r requirements.txt
```

### Run Application

```bash
python gui.py
```

---

## Example

### Input

I feel exhausted and overwhelmed by my workload and responsibilities.

### Output

Prediction: Stressed

Confidence: 94%

---

## Model Evaluation

The project evaluates the following metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score

Models Compared:

- Logistic Regression
- Naive Bayes

The best-performing model is used for final predictions.

---

## Learning Outcomes

This project demonstrates:

- Machine Learning
- Natural Language Processing
- Text Classification
- Feature Engineering
- Sentiment Analysis
- Data Visualization
- GUI Development

---

## Author

Fateh Noor Mangi

BSCS Student
