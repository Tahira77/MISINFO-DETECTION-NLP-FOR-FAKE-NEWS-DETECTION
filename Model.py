import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, f1_score
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')

# Load dataset
df = pd.read_csv('C:/Users/Thahira/Desktop/Tahira_project/Final year project/Sourcecode/Dataset/train.csv')

df.head()
df.shape
df.info()

# Handle missing data
df.isnull().sum()
df = df.fillna(' ')

# Combine 'author' and 'title' for content
df['content'] = df['author'] + " " + df['title']

# Initialize PorterStemmer
ps = PorterStemmer()

# Preprocessing function
def stemming(content):
    # Remove non-alphabetic characters, convert to lowercase, and split
    stemmed_content = re.sub('[^a-zA-Z]', " ", content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    
    # Remove stopwords
    stemmed_content = [ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = " ".join(stemmed_content)
    
    return stemmed_content

# Apply stemming
df['content'] = df['content'].apply(stemming)

# Define features and target variable
X = df['content']
y = df['label']

# Text vectorization using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X)

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Logistic Regression Model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_train_pred = log_reg.predict(X_train)
log_reg_test_pred = log_reg.predict(X_test)

# 2. Random Forest Model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_train_pred = rf_model.predict(X_train)
rf_test_pred = rf_model.predict(X_test)

# 3. Decision Tree Model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_train_pred = dt_model.predict(X_train)
dt_test_pred = dt_model.predict(X_test)

# 4. LSTM Model (For Text Classification)
# Tokenization and padding - Use raw text before TF-IDF vectorization
tokenizer = Tokenizer(num_words=5000, lower=True)
tokenizer.fit_on_texts(df['content'].astype(str))  # Use raw text here
X_sequences = tokenizer.texts_to_sequences(df['content'].astype(str))
X_sequences = pad_sequences(X_sequences, maxlen=100)

X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_sequences, y, test_size=0.2, random_state=42)

# LSTM model
lstm_model = Sequential()
lstm_model.add(Embedding(input_dim=5000, output_dim=128, input_length=100))
lstm_model.add(LSTM(units=128, dropout=0.2, recurrent_dropout=0.2))
lstm_model.add(Dense(1, activation='sigmoid'))

lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
lstm_model.fit(X_train_lstm, y_train_lstm, epochs=5, batch_size=64, validation_data=(X_test_lstm, y_test_lstm), verbose=2)

lstm_test_pred = lstm_model.predict(X_test_lstm)
lstm_test_pred = (lstm_test_pred > 0.5).astype(int)  # If it's a binary classification problem

lstm_train_pred = lstm_model.predict(X_train_lstm)
lstm_train_pred = (lstm_model.predict(X_train_lstm) > 0.5).astype("int32")


# Model Evaluation
def evaluate_model(model_name, y_train_pred, y_test_pred):
    print(f"{model_name} - Train Accuracy: ", accuracy_score(y_train, y_train_pred))
    print(f"{model_name} - Test Accuracy: ", accuracy_score(y_test, y_test_pred))
    print(f"{model_name} - Classification Report: \n", classification_report(y_test, y_test_pred))
    print(f"{model_name} - Confusion Matrix: \n", confusion_matrix(y_test, y_test_pred))
    
    # ROC and AUC
    fpr, tpr, _ = roc_curve(y_test, y_test_pred)
    auc = roc_auc_score(y_test, y_test_pred)
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.2f})")
    
# Evaluate all models
evaluate_model("Logistic Regression", log_reg_train_pred, log_reg_test_pred)
evaluate_model("Random Forest", rf_train_pred, rf_test_pred)
evaluate_model("Decision Tree", dt_train_pred, dt_test_pred)
evaluate_model("LSTM", lstm_train_pred, lstm_test_pred)

# Plot ROC curve for all models
plt.plot([0, 1], [0, 1], linestyle='--', label='Random')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.show()

# Save the models using pickle
pickle.dump(log_reg, open('log_reg_model.pkl', 'wb'))
pickle.dump(rf_model, open('rf_model.pkl', 'wb'))
pickle.dump(dt_model, open('dt_model.pkl', 'wb'))
pickle.dump(vectorizer, open('tfidf_vectorizer.pkl', 'wb'))
pickle.dump(tokenizer, open('tokenizer.pkl', 'wb'))

# For LSTM, you can save the Keras model as follows:
lstm_model.save('lstm_model.h5')
