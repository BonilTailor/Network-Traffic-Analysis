from flask import Flask, render_template, request, redirect, url_for
import json
import joblib
from utils import make_predictions  # Make sure this matches your utils.py

app = Flask(__name__)

# Load pre-trained models
rf_model_unsw = joblib.load("models/rf_model_unsw.joblib")
xgb_model_unsw = joblib.load("models/xgb_model_unsw.joblib")
iso_forest_model_unsw = joblib.load("models/iso_forest_model_unsw.joblib")
# autoencoder_model_unsw = joblib.load("models/autoencoder_model_unsw.h5")

rf_model_nsl = joblib.load("models/rf_model_nsl.joblib")
xgb_model_nsl = joblib.load("models/xgb_model_nsl.joblib")
iso_forest_model_nsl = joblib.load("models/iso_forest_model_nsl.joblib")
# autoencoder_model_nsl = joblib.load("models/autoencoder_model_nsl.h5")

rf_model_cicids = joblib.load("models/rf_model_cicids.joblib")
xgb_model_cicids = joblib.load("models/xgb_model_cicids.joblib")
iso_forest_model_cicids = joblib.load("models/iso_forest_model_cicids.joblib")
# autoencoder_model_cicids = joblib.load("models/autoencoder_model_cicids.h5")


# Homepage - Dataset and Model selection
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        dataset = request.form['dataset']
        model_choice = request.form['model']
        return redirect(url_for('predict', dataset=dataset, model_choice=model_choice))
    return render_template('index.html')

# Prediction Page
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    dataset = request.args.get('dataset')
    model_choice = request.args.get('model_choice')

    if request.method == 'POST':
        # Get JSON data from user input
        json_data = request.form['json_data']
        data = json.loads(json_data)

        # Use the make_predictions function from utils.py
        if dataset == 'UNSW':
            if model_choice == 'RandomForest':
                prediction = make_predictions(data, rf_model_unsw, dataset)
            elif model_choice == 'XGBoost':
                prediction = make_predictions(data, xgb_model_unsw, dataset)
            elif model_choice == 'IsolationForest':
                prediction = make_predictions(data, iso_forest_model_unsw, dataset)
            # elif model_choice == 'Autoencoder':
            #     prediction = make_predictions(data, autoencoder_model_unsw)
            else:
                prediction = "Invalid selection"
        elif dataset == 'NSL-KDD':
            if model_choice == 'RandomForest':
                prediction = make_predictions(data, rf_model_nsl, dataset)
            elif model_choice == 'XGBoost':
                prediction = make_predictions(data, xgb_model_nsl, dataset)
            elif model_choice == 'IsolationForest':
                prediction = make_predictions(data, iso_forest_model_nsl, dataset)
            # elif model_choice == 'Autoencoder':
            #     prediction = make_predictions(data, autoencoder_model_nsl)
            else:
                prediction = "Invalid selection"
        elif dataset == 'CICIDS':
            if model_choice == 'RandomForest':
                prediction = make_predictions(data, rf_model_cicids, dataset)
            elif model_choice == 'XGBoost':
                prediction = make_predictions(data, xgb_model_cicids, dataset)
            elif model_choice == 'IsolationForest':
                prediction = make_predictions(data, iso_forest_model_cicids, dataset)
            # elif model_choice == 'Autoencoder':
            #     prediction = make_predictions(data, autoencoder_model_cicids)
            else:
                prediction = "Invalid selection"
        else:
            prediction = "Invalid selection"

        return render_template('result.html', dataset=dataset, model_choice=model_choice, prediction=prediction)

    return render_template('predict.html', dataset=dataset, model_choice=model_choice)

if __name__ == '__main__':
    app.run(debug=True)
