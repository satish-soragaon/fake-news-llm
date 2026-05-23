# 📰 Fake News Detection System (Hybrid ML + LLM)

## 📌 Overview

This project is a **Hybrid AI-powered web application** that detects whether a news headline is **REAL or FAKE**.

It combines:
- **Machine Learning (ML)** for pattern detection  
- **Large Language Models (LLM)** for contextual understanding  

This hybrid approach improves prediction accuracy, especially for real-world and unseen news data.

---

## 🚀 Key Features

- 🔍 Real-time fake news detection  
- 🤖 Hybrid model (ML + LLM integration)  
- 📊 Confidence score for predictions  
- 🧠 Context analysis using transformer-based LLM  
- ⚡ Fast prediction using TF-IDF  
- 🌐 Flask-based web application  
- 🛡️ Input validation (gibberish & invalid text detection)

---

## 🧠 Tech Stack

- **Language:** Python  
- **Framework:** Flask  
- **Machine Learning:** Scikit-learn  
- **NLP:** TF-IDF Vectorization  
- **LLM:** Hugging Face Transformers (Zero-shot Classification)  
- **Libraries:** PyTorch, Pandas, NumPy, Pickle, Math, Regex  
- **Frontend:** HTML, CSS (Jinja Templates)

---

## 🏗️ Project Architecture
    User Input
    ↓
    Preprocessing
    ↓
    TF-IDF Vectorization
    ↓
    ML Model Prediction
    ↓
    LLM Context Analysis (Transformers)
    ↓
    Combine Results
    ↓
    Final Output + Confidence Score


---

## 📂 Project Structure
Fake-News-Detection/
│
├── app.py
├── finalized_model.pkl
├── vectorizer.pkl
├── dataset/
├── notebooks/
├── templates/
│ ├── index.html
│ └── prediction.html
├── static/
├── Examiner_Testing_Samples.txt
├── requirements.txt
└── Procfile

---

## ⚙️ How It Works

1. User enters a news headline  
2. Input is validated (checks for empty/gibberish text)  
3. Text is preprocessed (cleaning, stopword removal)  
4. TF-IDF converts text into numerical format  
5. ML model predicts REAL or FAKE  
6. Confidence score is calculated using sigmoid function  
7. LLM analyzes context using zero-shot classification  
8. ML and LLM results are combined  
9. Final prediction + confidence + explanation is displayed  

---

## 🧠 Models Used

### Machine Learning Models:
- Logistic Regression  
- Naive Bayes  
- Passive Aggressive Classifier  
- Random Forest  

Best model is selected based on performance.

---

### LLM Model:
- distilbert-base-uncased-mnli (Hugging Face)  
- Used for zero-shot classification  
- Performs contextual verification  

---

## 📊 Output Example
Prediction: REAL
Confidence: 82%
LLM Analysis: Linguistic structure indicates objective reporting


---

## 📊 Dataset

- Source: Hugging Face (GonzaloA/fake_news)  
- Contains labeled REAL and FAKE news  
- Used for training and evaluation  

---

## ▶️ Installation & Setup

```bash
pip install -r requirements.txt
python app.py

## Open in browser:
http://127.0.0.1:5000/