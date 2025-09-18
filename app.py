from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# Load model and vectorizer
MODEL_PATH = os.path.join(os.path.dirname(__file__), "logistic_model.pkl")
VEC_PATH = os.path.join(os.path.dirname(__file__), "tfidf_vectorizer.pkl")
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VEC_PATH)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        tweet = request.form["tweet"]
        # preprocess same as training
        # vectorize
        vect = vectorizer.transform([tweet])
        pred = model.predict(vect)[0]
        return render_template("result.html", prediction=int(pred), tweet=tweet)
    return render_template("index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
