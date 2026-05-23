# 📰 Fake News Detection System (ML + LLM)

## 📌 Overview

This project is a web-based application that classifies news headlines as **REAL or FAKE** using a combination of machine learning and natural language processing techniques.

The system uses **TF-IDF** for feature extraction and a **Passive Aggressive Classifier** for efficient text classification. Additionally, a transformer-based **LLM** is used to enhance contextual understanding and improve prediction quality.

---

## 🚀 Key Features

- Real-time fake news detection  
- Combination of ML and LLM techniques  
- Confidence score for predictions  
- Flask-based web application  
- Input validation for invalid or meaningless text  

---

## 🧠 Tech Stack

- Python  
- Flask  
- Scikit-learn  
- TF-IDF  
- Passive Aggressive Classifier  
- Hugging Face Transformers  
- HTML, CSS  

---

## ⚙️ How It Works

1. User enters a news headline  
2. Text is preprocessed  
3. TF-IDF converts text into numerical format  
4. Passive Aggressive Classifier predicts REAL or FAKE  
5. LLM performs contextual analysis  
6. Results are combined to generate final output  

---

## 📂 Project Structure

```
app.py
finalized_model.pkl
vectorizer.pkl
templates/
static/
requirements.txt
README.md
```

---

## 📊 Dataset

- Source: Hugging Face (GonzaloA/fake_news)  
- Contains labeled news articles categorized as REAL and FAKE  
- Used for training and evaluating the machine learning model  

**Note:** The dataset file is not included in this repository due to size limitations.

---

## ▶️ Run the Project

```bash
pip install -r requirements.txt
python app.py
```

Open in browser:  
http://127.0.0.1:5000/

---

## 📌 Note

This project demonstrates the integration of machine learning and language models for text classification.
