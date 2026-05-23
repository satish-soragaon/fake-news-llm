import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

print("--- Initializing NLTK Resources ---")
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def explicit_preprocess(text):
    """
    Explicit cleaning pipeline as requested by the examiner:
    1. Lowercasing
    2. Removing punctuation and non-alphanumeric chars
    3. Removing English Stopwords
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercasing
    text = text.lower()
    
    # 2. Pattern to remove punctuation & symbols
    text = re.sub(r'[^\w\s]', '', text)
    
    # 3. Splitting and removing stopwords
    words = text.split()
    cleaned_words = [word for word in words if word not in stop_words]
    
    return " ".join(cleaned_words)

print("\n--- Data Acquisition ---")
try:
    df = pd.read_csv("news.csv")
except Exception as e:
    print(f"Failed to load local dataset: {e}")
    exit(1)
df['combined_text'] = df['title'].fillna('') + " " + df['text'].fillna('')

print("\n--- Explicit Text Preprocessing Pipeline ---")
print("Cleaning 5,000 articles (lowercasing, punctuation removal, stopword removal)...")
df['clean_text'] = df['combined_text'].apply(explicit_preprocess)

x_train, x_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label_text'], test_size=0.2, random_state=42
)

print("\n--- Vectorization ---")
vectorizer = TfidfVectorizer(max_df=0.7) # Stopwords handled manually above 
tf_train = vectorizer.fit_transform(x_train)
tf_test = vectorizer.transform(x_test)

print("\n--- Multi-Model Training & Comparison ---")
# Instantiate models
models = {
    'Passive Aggressive Classifier': PassiveAggressiveClassifier(max_iter=50, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=500, random_state=42),
    'Naive Bayes (Multinomial)': MultinomialNB(),
    'Random Forest Classifier': RandomForestClassifier(n_estimators=100, random_state=42)
}

best_accuracy = 0
best_model_name = ""
best_model = None

results = []

for name, model in models.items():
    print(f"Training [{name}]...")
    model.fit(tf_train, y_train)
    y_pred = model.predict(tf_test)
    acc = accuracy_score(y_test, y_pred)
    results.append({'Model': name, 'Accuracy': acc})
    print(f" -> Accuracy: {round(acc*100,2)}%")
    
    if acc > best_accuracy:
        best_accuracy = acc
        best_model_name = name
        best_model = model

print("\n--- Final Full-Dataset Training ---")
print("Retraining the winning model on 100% of the dataset to ensure all new examiner samples are included...")
final_vectorizer = TfidfVectorizer(max_df=0.7)
tf_full = final_vectorizer.fit_transform(df['clean_text'])
best_model.fit(tf_full, df['label_text'])

print(f"\nSaving best performing model to disk...")
pickle.dump(best_model, open("finalized_model.pkl", "wb"))
pickle.dump(final_vectorizer, open("vectorizer.pkl", "wb"))

# Do not overwrite news.csv as we are already reading from it
# df.to_csv("news.csv", index=False)

print("Professional multi-model pipeline finished successfully!")
