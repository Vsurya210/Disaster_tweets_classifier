from flask import Flask, render_template, request
import joblib
import re

app = Flask(__name__)

model = joblib.load("logistic_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text.strip()

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    
    if request.method == "POST":
        tweet = request.form["tweet"]
        cleaned = clean_text(tweet)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0][1]
        
        prediction = "🚨 Disaster Tweet" if pred == 1 else "✅ Not a Disaster Tweet"
        confidence = round(prob * 100, 2)
    
    return render_template("index.html", prediction=prediction, confidence=confidence)

if __name__ == "__main__":
    app.run(debug=True)
