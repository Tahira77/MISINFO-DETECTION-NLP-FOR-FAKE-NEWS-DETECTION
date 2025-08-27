# MISINFO-DETECTION-NLP-FOR-FAKE-NEWS-DETECTION
Detecting misinformation using Natural Language Processing and Machine Learning

## 📌 Project Overview
The rise of online misinformation has made it crucial to distinguish between real and fake news.  
This project leverages *Natural Language Processing (NLP)* and *Machine Learning* techniques to build a system that classifies news articles as *Real* or *Fake*.  

---

## 🎯 Objectives
- Preprocess textual news data using NLP techniques.  
- Train ML models to classify fake and real news.  
- Compare model performances to identify the most accurate approach.  
- Provide a user-friendly interface for real-time prediction.  

---

## 🛠️ Methodology
1. *Data Preprocessing*  
   - Tokenization, stop-word removal, stemming/lemmatization  
   - Text representation using *TF-IDF Vectorization*  

2. *Model Development*  
   - Logistic Regression  
   - Decision Tree Classifier  
   - K-Nearest Neighbors (KNN)  

3. *Evaluation Metrics*  
   - Accuracy  
   - Precision  
   - Recall  
   - F1-Score  

4. *Deployment*  
   - Integrated with *Streamlit* for an interactive prediction interface.  

---

## 📊 Results
- Logistic Regression: High accuracy with efficient classification.  
- Decision Tree: Moderate accuracy, useful for interpretability.  
- KNN: Lower accuracy compared to other models.  

✅ *Logistic Regression outperformed other models* in detecting fake news.  

---

## 🚀 Tech Stack
- *Programming Language:* Python  
- *Libraries:* Pandas, NumPy, Scikit-learn, NLTK, Matplotlib, Streamlit  
- *Techniques:* NLP (TF-IDF, tokenization), ML classification models  

---

## 📂 Project Structure
Fake-News-Detection/ │ ├── data/                 # Dataset files (not included due to size/license) ├── notebooks/            # Jupyter notebooks (EDA & model training) ├── app.py                # Streamlit web app for prediction ├── requirements.txt      # Project dependencies ├── README.md             # Project documentation

## ▶️ How to Run
1. Clone the repository  
   ```bash
   git clone https://github.com/your-username/Fake-News-Detection.git
   cd Fake-News-Detection

2. Install dependencies

pip install -r requirements.txt


3. Run the app

streamlit run app.py
