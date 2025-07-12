# Bank Customer Feedback Sentiment Analyzer

This Streamlit app analyzes customer feedback submitted to banks and performs:

-  **Sentiment analysis** (Positive / Neutral / Negative)
-  **Department classification** (Loan, App, Customer Service, etc.)
-  **City-wise & Department-wise sentiment breakdowns**



## 🔍 Features

-  Upload a `.csv` file with feedback and city names
-  Uses RoBERTa model (`cardiffnlp/twitter-roberta-base-sentiment`) for sentiment prediction
-  Classifies each feedback into a **banking department** based on keywords
-  Shows summary charts:
  - Sentiment by Department
  - Sentiment by City
  - City-wise Department Sentiment Matrix



##  Input Format

Your CSV should have the following columns:

| City     | Feedback                                                |
|----------|----------------------------------------------------------|
| Mumbai   | The loan process was very smooth and transparent.       |
| Delhi    | App keeps crashing while transferring funds.            |
| Bangalore| Customer service was quick and resolved my issue.       |

Column names must be exactly: `City`, `Feedback`



## 🧠 Models Used

- **Sentiment Analysis**: [`cardiffnlp/twitter-roberta-base-sentiment`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment)
- **Department Classification**: Rule-based on keywords (e.g., "loan", "app", "support")

---
