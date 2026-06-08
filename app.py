import os
import joblib
import numpy as np
from flask import Flask, render_template, request, flash, redirect, url_for

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-this-in-production")

# Load models once at startup
try:  
    models = {
        "Logistic Regression": joblib.load("Logistic_Regression.pkl"),
        "Random Forest": joblib.load("Random_Forest.pkl"),
        "XGBoost": joblib.load("XGBoost.pkl")
    }
except Exception as e:
    raise RuntimeError(f"Could not load models: {e}")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Convert incoming form data values directly to floats
        # Note: Ensure the form fields arrive in the exact order your models expect
        features = [float(val) for val in request.form.values() if val.replace('.', '', 1).isdigit()]
        
        if not features:
            flash("No valid numeric features found in the submission.", "error")
            return render_template("index.html", form_data=request.form)

        # Format the data for scikit-learn (expects a 2D array)
        input_data = [features]
        results = {}

        selected_model_name = request.form.get("model")

        if selected_model_name and selected_model_name in models:
            # User selected one specific model
            model = models[selected_model_name]
            pred = model.predict(input_data)[0]
            prob = model.predict_proba(input_data)[0][1]
            
            results[selected_model_name] = {
                "label": int(pred),
                "probability": round(prob * 100, 2)
            }
        else:
            # Generate predictions for all models to compare them
            for name, model in models.items():
                pred = model.predict(input_data)[0]
                prob = model.predict_proba(input_data)[0][1]

                results[name] = {
                    "label": int(pred),
                    "probability": round(prob * 100, 2)
                }

        return render_template(
            "index.html",
            results=results,
            form_data=request.form
        )

    except Exception as e:
        flash(f"Prediction error: {str(e)}", "error")
        return render_template(
            "index.html",
            form_data=request.form
        )


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug)