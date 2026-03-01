from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        cgpa = float(request.form["cgpa"])
        projects = float(request.form["projects"])
        certifications = float(request.form["certifications"])
        internship_exp = float(request.form["internship_exp"])
        communication = float(request.form["communication"])

        features = np.array([[cgpa, projects, certifications, internship_exp, communication]])
        prediction = model.predict(features)

        if prediction[0] == 1:
            result = "Candidate is likely to be SELECTED ✅"
        else:
            result = "Candidate is NOT likely to be selected ❌"

        return render_template("index.html", prediction_text=result)

    except:
        return render_template("index.html", prediction_text="Error in input. Please enter valid numbers.")

if __name__ == "__main__":
    app.run(debug=True)