from flask import Flask, render_template, request
import pickle
import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# --- Ensure NLTK data ---
from nltk.data import find
for resource in ['stopwords', 'wordnet', 'omw-1.4', 'punkt', 'punkt_tab']:
    try:
        find(f'tokenizers/{resource}')
    except:
        nltk.download(resource)

# --- App setup ---
app = Flask(__name__)

# Load model + vectorizer
model = load_model('model/model.keras')
with open('model/tfidf.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# --- Text cleaning function ---
def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# --- Prediction function ---
def predict_job(job_text):
    cleaned_text = clean_text(job_text)
    text_tfidf = tfidf_vectorizer.transform([cleaned_text])
    prediction = model.predict(text_tfidf)
    return bool(prediction > 0.5)

# --- Flask routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        title = request.form.get('title', '')
        company = request.form.get('company', '')
        location = request.form.get('location', '')
        description = request.form.get('description', '')
        requirements = request.form.get('requirements', '')
        benefits = request.form.get('benefits', '')

        job_text = ' '.join([title, company, location, description, requirements, benefits])

        if job_text:
            is_fake = predict_job(job_text)
            if is_fake:
                result = "This job posting is likely **FAKE**."
            else:
                result = "This job posting appears **REAL**."
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
