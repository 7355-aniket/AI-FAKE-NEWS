# 📰 AI Fake News Detection System

An Artificial Intelligence and Machine Learning-based project that analyzes news articles or text and classifies them as **REAL** or **FAKE** using **Natural Language Processing (NLP)** and Machine Learning.

The project uses **TF-IDF (Term Frequency-Inverse Document Frequency)** for text feature extraction and **Logistic Regression** for classification.

> **Note:** This is an educational AI/ML project. A model's prediction is not definitive proof that a news article is true or false and should be verified using reliable sources.

---

## 🎯 Project Objective

The main objective of this project is to develop an AI-based system that can identify potentially fake news from textual content.

The system:

* Takes a news article or headline as input.
* Processes the text using NLP techniques.
* Converts text into numerical features using TF-IDF.
* Uses a Machine Learning classifier to make a prediction.
* Displays the predicted class and model confidence.

---

## 🚀 Features

* 📰 News article classification
* 🤖 Machine Learning-based prediction
* 🧠 Natural Language Processing
* 🔤 TF-IDF text vectorization
* 📊 Logistic Regression classifier
* 📈 Accuracy evaluation
* 📋 Classification report
* 🔲 Confusion matrix
* 🎯 Prediction confidence
* 💻 Google Colab compatible
* 🐍 Python based

---

# 🧠 Technologies Used

### Programming Language

* Python

### Machine Learning

* Scikit-learn
* Logistic Regression
* TF-IDF Vectorization

### Data Processing

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn

### Model Saving

* Joblib

### Development Environment

* Google Colab
* Jupyter Notebook
* VS Code

---

# 🏗️ System Architecture

```text
                 NEWS ARTICLE
                      │
                      ▼
               TEXT PREPROCESSING
                      │
                      ▼
                 TF-IDF
              VECTORIZATION
                      │
                      ▼
             FEATURE EXTRACTION
                      │
                      ▼
            LOGISTIC REGRESSION
                      │
                      ▼
              ┌───────┴───────┐
              │               │
              ▼               ▼
            REAL             FAKE
              │               │
              └───────┬───────┘
                      ▼
              CONFIDENCE SCORE
```

---

# 📂 Project Structure

```text
AI-Fake-News-Detection/
│
├── AI_Fake_News_Detection.ipynb
│
├── dataset/
│   └── news.csv
│
├── models/
│   └── fake_news_model.pkl
│
├── requirements.txt
│
└── README.md
```

---

# 📊 Dataset

The dataset should contain news text and its corresponding label.

Example:

```text
| text                                      | label |
|-------------------------------------------|-------|
| Government announces new education policy | REAL  |
| Scientists publish a new research study  | REAL  |
| Miracle drink cures every disease        | FAKE  |
| Humans can live forever using this trick  | FAKE  |
```

For meaningful model evaluation, use a **large, properly labeled dataset** rather than a tiny demonstration dataset.

Possible datasets for experimentation include publicly available fake-news datasets from research repositories and dataset platforms, subject to their respective licenses and terms.

---

# 🔄 Machine Learning Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Missing Value Handling
   ↓
Train/Test Split
   ↓
TF-IDF Vectorization
   ↓
Logistic Regression
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Prediction
```

---

# 🧹 Data Preprocessing

The project prepares the text before training.

Typical processing includes:

* Removing missing values
* Converting text into numerical representation
* Removing common English stop words
* TF-IDF feature extraction

---

# 🔤 TF-IDF

TF-IDF stands for:

**Term Frequency-Inverse Document Frequency**

It converts textual information into numerical values that can be processed by a Machine Learning algorithm.

Conceptually:

```text
News Text
    ↓
Words / Tokens
    ↓
TF-IDF
    ↓
Numerical Feature Vector
    ↓
Machine Learning Model
```

---

# 🤖 Machine Learning Model

The project uses:

## Logistic Regression

Logistic Regression is a classification algorithm that learns patterns from labeled training data and predicts the class of new text.

In this project:

```text
Input:
News Article

Output:
REAL / FAKE
```

---

# 📈 Model Evaluation

The model can be evaluated using:

### Accuracy

Measures the proportion of predictions that are correct.

```text
Accuracy =
Correct Predictions / Total Predictions
```

### Precision

Measures how many predicted samples of a class were actually that class.

### Recall

Measures how many actual samples of a class were correctly detected.

### F1-Score

Provides a combined measure of precision and recall.

### Confusion Matrix

The confusion matrix shows correct and incorrect classifications.

```text
                  PREDICTED
                FAKE     REAL

ACTUAL FAKE      TP       FN

ACTUAL REAL      FP       TN
```

---

# 🧪 Example Prediction

### Input

```text
Scientists announce a new research project to study climate change.
```

### Example Output

```text
Prediction: REAL
Confidence: 89.42%
```

Another example:

```text
A miracle drink allows every person to become immortal within one day.
```

Example output:

```text
Prediction: FAKE
Confidence: 94.17%
```

> The confidence values above are examples. Actual values depend on the dataset and trained model.

---

# 💻 Running the Project in Google Colab

### Step 1 — Open Google Colab

Create a new notebook.

### Step 2 — Install libraries

```python
!pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

### Step 3 — Import libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

### Step 4 — Load the dataset

```python
df = pd.read_csv("news.csv")

print(df.head())
print(df.shape)
```

### Step 5 — Prepare the data

```python
df = df.dropna(subset=["text", "label"])

X = df["text"]
y = df["label"]
```

### Step 6 — Split the data

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

### Step 7 — Create the model

```python
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            stop_words="english",
            max_features=10000
        )
    ),
    (
        "classifier",
        LogisticRegression(max_iter=1000)
    )
])
```

### Step 8 — Train

```python
model.fit(X_train, y_train)

print("Training completed!")
```

### Step 9 — Evaluate

```python
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

### Step 10 — Confusion Matrix

```python
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["FAKE", "REAL"],
    yticklabels=["FAKE", "REAL"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Fake News Detection Confusion Matrix")

plt.show()
```

### Step 11 — Test new news

```python
news = """
Scientists announce a new research project to study climate change.
"""

prediction = model.predict([news])[0]

confidence = model.predict_proba([news])[0].max()

print("Prediction:", prediction)
print("Confidence:", round(confidence * 100, 2), "%")
```

### Step 12 — Save the model

```python
joblib.dump(
    model,
    "fake_news_model.pkl"
)

print("Model saved successfully!")
```

---

# 📊 Results

The project evaluates the trained model using:

```text
✓ Accuracy
✓ Precision
✓ Recall
✓ F1-Score
✓ Confusion Matrix
```

The exact results depend on the dataset, preprocessing, train/test split, and model configuration.

---

# 🔮 Future Improvements

The project can be extended with:

* Transformer-based NLP models
* BERT-based classification
* DistilBERT
* Advanced text preprocessing
* Source credibility analysis
* Multilingual fake-news detection
* Real-time news analysis
* News verification using external sources
* Flask web application
* REST API
* User prediction history
* Explainable AI
* RAG-based fact verification

---

# ⚠️ Limitations

* Model performance depends strongly on the quality and diversity of the training dataset.
* A model may learn dataset-specific writing patterns rather than factual truth.
* Confidence scores represent model outputs and should not be interpreted as certainty.
* The system does not independently establish whether the claims in an article are factually correct.
* New types of misinformation may not be detected reliably if they differ from the training data.

---

# 🔐 Ethical Considerations

This project is intended for:

* Education
* Research
* AI/ML learning
* Demonstration of NLP classification

The prediction should be treated as an **AI-assisted signal**, not as a final fact-checking decision.

Important news or claims should be verified through reliable and independent sources.

---

# 🎓 Learning Outcomes

After completing this project, you will understand:

* Python for AI/ML
* Data preprocessing
* Natural Language Processing
* TF-IDF
* Text classification
* Logistic Regression
* Train/test splitting
* Model evaluation
* Confusion matrices
* Precision, recall and F1-score
* Saving and loading ML models
* Building an end-to-end AI project

---

# ⭐ Project Highlights

```text
✔ Artificial Intelligence
✔ Machine Learning
✔ Natural Language Processing
✔ TF-IDF
✔ Logistic Regression
✔ Fake News Classification
✔ Text Classification
✔ Model Evaluation
✔ Confusion Matrix
✔ Confidence Score
✔ Python
✔ Google Colab
✔ VS Code
```

---

# 📌 Conclusion

The **AI Fake News Detection System** demonstrates the application of Natural Language Processing and Machine Learning to classify news text.

By combining **TF-IDF feature extraction** with **Logistic Regression**, the system learns patterns from labeled news data and generates predictions for new text.

The project provides a foundation for developing more advanced AI-powered information verification systems using modern NLP models and external evidence sources.
