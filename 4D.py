#Exploratory Data Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

fake = pd.read_csv(r"C:\Users\ASUS\Documents\Fake.csv")
true = pd.read_csv(r"C:\Users\ASUS\Documents\True.csv")

fake['label'] = 0  # 0 for Fake
true['label'] = 1  # 1 for Real

data = pd.concat([fake, true], ignore_index=True)

print("Dataset Shape:", data.shape)
print("\nColumns:", data.columns)
print("\nData Types:\n", data.dtypes)

print("\nMissing values:\n", data.isnull().sum())

print(data.sample(5))

sns.countplot(x='label', data=data)
plt.xticks([0, 1], ['Fake', 'Real'])  # Proper labels
plt.title('Count of Real vs Fake News')
plt.xlabel('News Type')
plt.ylabel('Count')
plt.show()

#Text Cleaning and Pre-Processing

import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')  
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Tokenize
    words = word_tokenize(text)
    # Remove punctuation and stopwords
    cleaned = [word for word in words if word not in stop_words and word not in string.punctuation]
    return " ".join(cleaned)
data['cleaned_text'] = data['text'].apply(preprocess_text)
print(data[['text', 'cleaned_text']].sample(3))

#TF-IDF Vectorization, Model building
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
X = data['cleaned_text']  # Features
y = data['label']         # Labels
tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42
)
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)


print("🔹 Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print("🔹 Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))

print("\nClassification Report (Logistic Regression):\n")
print(classification_report(y_test, y_pred_lr))

print("\nClassification Report (Naive Bayes):\n")
print(classification_report(y_test, y_pred_nb))

#confusion matrix

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import joblib

cm_lr = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

cm_nb = confusion_matrix(y_test, y_pred_nb)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Oranges')
plt.title("Confusion Matrix - Naive Bayes")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

joblib.dump(lr_model, 'logistic_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
joblib.dump(nb_model, 'naive_bayes_model.pkl')
