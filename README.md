
# WhatsApp Chat Analysis Project

## Overview
The **WhatsApp Chat Analysis project** applies **Natural Language Processing (NLP)** and **Machine Learning (ML)** techniques to analyze chat data. It extracts meaningful insights such as top contributors, sentiment trends, spam detection, common topics, and conversation summaries. This project is ideal for identifying patterns in communication, visualizing trends, and gaining deeper insights into chat behaviors.

---

## Project Workflow

### 1. Data Collection & Preprocessing
- **Export Chat:** WhatsApp provides a `.txt` export of chat history.
- **Preprocessing Steps:**
  - **Remove timestamps, emojis, and special characters.**
  - **Lowercase Conversion:** Standardize all text to lowercase.
  - **Tokenization:** Split sentences into individual words.
  - **Stopword Removal:** Eliminate common words (like “the,” “is”).
  - **Stemming/Lemmatization:** Reduce words to their root forms (e.g., "running" → "run").

**Example Code:**
```python
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

def preprocess_message(message):
    message = re.sub(r'\W+', ' ', message)  # Remove non-alphabetic chars
    tokens = message.lower().split()  
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    return ' '.join([lemmatizer.lemmatize(word) for word in tokens if word not in stop_words])
```

---

### 2. Exploratory Data Analysis (EDA)
- **Analyze:** Identify patterns such as:
  - **Top Contributors:** Who sends the most messages?
  - **Frequent Words:** Visualize common words using **word clouds**.
  - **Message Trends:** Track daily/monthly chat activity.

**Visualization Example:**
```python
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('chat.csv')
user_message_counts = df['sender'].value_counts()

user_message_counts.plot(kind='bar')
plt.title('Top Contributors')
plt.show()
```

---

### 3. Sentiment Analysis (NLP)
- **Sentiment Polarity:** Determine whether messages are positive, negative, or neutral.
- Tools:
  - **TextBlob** for polarity and subjectivity.
  - **VADER** for social media-focused sentiment.

**Example Code:**
```python
from textblob import TextBlob

def get_sentiment(message):
    analysis = TextBlob(message)
    return analysis.sentiment.polarity  # Output: [-1, 1]

df['sentiment'] = df['message'].apply(get_sentiment)
```

---

### 4. Topic Modeling (NLP)
- **Latent Dirichlet Allocation (LDA):** Identify common topics based on word co-occurrence patterns.
- Use LDA to uncover themes like "work," "movies," or "travel."

**Example Code:**
```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

vectorizer = CountVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(df['message'])

lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(X)

for idx, topic in enumerate(lda.components_):
    print(f"Topic {idx}: {', '.join([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]])}")
```

---

### 5. Spam Detection / Message Classification (ML)
- Use **supervised learning models** like Logistic Regression to classify messages as spam or non-spam.
- **TF-IDF Vectorization** is used to convert text into numerical data.

**Example Code:**
```python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['message'])

y = df['label']  # 0 (non-spam), 1 (spam)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))
```

---

### 6. Chat Summarization (NLP)
- **Summarize conversations** using:
  - **Extractive Summarization:** Select key sentences (TextRank).
  - **Abstractive Summarization:** Generate condensed versions with **BART**.

**Example Code:**
```python
from transformers import pipeline

summarizer = pipeline('summarization', model='facebook/bart-large-cnn')

chat_summary = summarizer(" ".join(df['message'].tolist()), max_length=50, min_length=25, do_sample=False)
print(chat_summary[0]['summary_text'])
```

---

### 7. Dashboard and Visualization
- Use **Streamlit** or **Plotly Dash** to build an interactive dashboard for analysis.
- Dashboard Features:
  - Display top contributors, word clouds, and sentiment trends.
  - Allow users to search chats by keywords or topics.

**Example Code:**
```python
import streamlit as st

st.title("WhatsApp Chat Analysis Dashboard")
st.bar_chart(user_message_counts)
st.write(chat_summary[0]['summary_text'])
```

---

## Conclusion
This project combines **NLP** (sentiment analysis, topic modeling, summarization) and **ML** (spam detection) techniques to deliver actionable insights from WhatsApp chat data. With visual dashboards, users can explore chat patterns, track emotional trends, and efficiently summarize conversations. The project can be extended for personal or business applications to monitor communication habits, detect spam, or identify emerging themes.

---

## Requirements
- Python 3.8+
- Libraries: `pandas`, `nltk`, `sklearn`, `textblob`, `transformers`, `streamlit`

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run the script for analysis: `python chat_analysis.py`
3. (Optional) Start the dashboard: `streamlit run dashboard.py`

---

## License
This project is open-source and available under the MIT License.
