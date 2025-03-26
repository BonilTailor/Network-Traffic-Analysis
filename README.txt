NETWORK TRAFFIC ANALYSIS

Overview

This project implements a web-based application for detecting anomalies in datasets using advanced machine learning models, including Random Forest, XGBoost, Isolation Forest, and Autoencoder. Users can upload JSON datasets, select models, and view prediction results through an intuitive interface.

Features

Home Page: Select models and datasets for analysis.
Upload Page: Input datasets in JSON format for processing.
Results Page: View dataset name, selected model, and prediction results.

Requirements

Python 3.8+
Flask
Machine Learning Libraries: scikit-learn, XGBoost, TensorFlow, pandas, numpy
 
Install dependencies:

pip install -r requirements.txt  

Run the Flask application:

python app.py 
 
Open the application in a browser:

http://127.0.0.1:5000

File Structure

app.py: Main Flask application.
templates/: HTML files for the UI.
static/: CSS/JS assets.
models/: Pre-trained machine learning models.
data/: Sample datasets for testing.