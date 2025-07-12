# Sentiment Analysis of Bank Customer Feedback

This project presents a comprehensive analysis of customer reviews collected from a banking service. It aims to classify the sentiment behind each review using a combination of rule-based, classical machine learning, and deep learning methods, including transformer-based models.



##  Objective

To identify the sentiment expressed in customer feedback — whether **positive**, **neutral**, or **negative** — and to compare multiple modeling approaches to determine the most suitable one for customer-facing financial data.



##  Dataset Description

- **Source**: Bank reviews collected from real users.
- **Format**: CSV file containing various fields including:
  - `review`: Text content of the feedback
  - `rating`: Numerical score from the user
  - `date`: Date of review submission
  - `city`: Location of the user
- **Target Variable**: Sentiment label derived from `rating` (or optionally generated using VADER/TextBlob)

### Label Mapping Strategy:
```
Rating >= 4  → Positive  
Rating == 3  → Neutral  
Rating <= 2  → Negative
```



##  Project Workflow

### 1️⃣ Data Cleaning & Preprocessing
- Removed irrelevant fields (`bank_image`, etc.)
- Converted all text to lowercase
- Removed missing or malformed entries
- Tokenized or detokenized review text where necessary

### 2️⃣ Exploratory Data Analysis (EDA)
- Analyzed:
  - Review lengths
  - Monthly review trends
  - Word and character counts
  - Reviews grouped by city
- Visualized using `seaborn`, `matplotlib`

### 3️⃣ Sentiment Labeling
- **Rule-based labeling** using:
  - VADER (NLTK)
  - TextBlob
  - spaCy + TextBlob (optional)
- Alternatively, mapped sentiment using user ratings

### 4️⃣ Feature Engineering
- TF-IDF Vectorization (including n-grams)
- Word2Vec and GloVe word embeddings
- Dimensionality reduction (PCA) for visualization

### 5️⃣ Class Imbalance Handling
- Identified skewed distribution in sentiment classes
- Used:
  - `RandomOverSampler` from `imblearn`
  - `SMOTE` for synthetic oversampling
- Visualized class balance before and after resampling

### 6️⃣ Model Training & Evaluation

#### 🔹 Classical ML Models
- Logistic Regression
- Naive Bayes

#### 🔹 Deep Learning Models
- LSTM with word embeddings

#### 🔹 Transformer-Based Models
- **DistilBERT** using HuggingFace `transformers`
- **RoBERTa** (Robustly Optimized BERT Pretraining Approach)

  Trained using:
  - HuggingFace's `Trainer` API
  - Tokenization with `RobertaTokenizer`
  - Cross-entropy loss for multiclass classification

### 7️⃣ Performance Evaluation
- Accuracy and F1-scores per class
- Classification report and confusion matrix
- Comparison of:
  - Model precision/recall trade-offs
  - Sentiment agreement between tools (VADER vs TextBlob)



##  Visualizations

- Monthly trends in feedback submission
- City-wise sentiment distribution
- Confusion matrices for each model
- PCA plots for high-dimensional embeddings
- Bar charts comparing model performance metrics



##  Dependencies

Install required libraries with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk transformers datasets imbalanced-learn
```

Also download NLTK resources:
```python
import nltk
nltk.download('vader_lexicon')
```



##  Future Improvements

- Deploy the best model via REST API using Flask/FastAPI
- Add interpretability using LIME/SHAP
- Extend to multilingual datasets
- Incorporate sarcasm detection and contextual sentiment shifts



##  Author

**Madhu Kumari**  
Natural Language Processing Enthusiast | Applied Machine Learning
