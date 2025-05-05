from flask import Flask, request, jsonify, render_template
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords

# Ensure NLTK stopwords are downloaded
nltk.download('stopwords')

# Initialize Flask app
app = Flask(__name__)

# Load vectorizer and model
with open("fake_news_model.pkl", "rb") as f:
    vectorizer, model = pickle.load(f)

# Preprocessing function
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# Home route
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "")
        if not text:
            return jsonify({"error": "No input text provided."}), 400

        cleaned = clean_text(text)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]
        label = "REAL" if prediction == 1 else "FAKE"
        return jsonify({"prediction": label})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run app
if __name__ == "__main__":
    app.run(debug=True)