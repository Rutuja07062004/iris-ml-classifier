from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        features = [float(x) for x in request.form.values()]
        prediction = model.predict([features])[0]
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
