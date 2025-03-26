import pandas as pd
import numpy as np
from joblib import load
import tensorflow as tf
from keras import backend as K
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Define the mean squared error function
@tf.keras.utils.register_keras_serializable()
def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)

# Load trained models for each dataset
def load_model_with_custom_objects(model_path):
    return tf.keras.models.load_model(model_path, custom_objects={'mean_squared_error': mean_squared_error})

# Load models
rf_model_unsw = load('models/rf_model_unsw.joblib')
xgb_model_unsw = load('models/xgb_model_unsw.joblib')
iso_forest_model_unsw = load('models/iso_forest_model_unsw.joblib')
# autoencoder_model_unsw = load_model_with_custom_objects('models/autoencoder_model_unsw.h5')

rf_model_nsl = load('models/rf_model_nsl.joblib')
xgb_model_nsl = load('models/xgb_model_nsl.joblib')
iso_forest_model_nsl = load('models/iso_forest_model_nsl.joblib')
# autoencoder_model_nsl = load_model_with_custom_objects('models/autoencoder_model_nsl.h5')

rf_model_cicids = load('models/rf_model_cicids.joblib')
xgb_model_cicids = load('models/xgb_model_cicids.joblib')
iso_forest_model_cicids = load('models/iso_forest_model_cicids.joblib')
# autoencoder_model_cicids = load_model_with_custom_objects('models/autoencoder_model_cicids.h5')

# Load the fitted preprocessor
preprocessor_nsl = load('models/preprocessor_nsl.joblib')  # Update this for each dataset as necessary
preprocessor_unsw = load('models/preprocessor_unsw.joblib')  # Update this for each dataset as necessary


def preprocess_data(data, dataset):
    """Preprocess incoming data."""
    df = pd.DataFrame([data])
    if dataset == 'UNSW':
        return preprocessor_unsw.transform(df)
    elif dataset == 'NSL-KDD':
        return preprocessor_nsl.transform(df)
    


def make_predictions(data, model, dataset):
    """Make predictions with the provided model and return results."""
    X_processed = preprocess_data(data, dataset)

    # Check the model type and make predictions accordingly
    if 'RandomForest' in model.__class__.__name__:
        prediction = model.predict(X_processed) 
    elif 'XGBClassifier' in model.__class__.__name__:
        prediction = model.predict(X_processed)
    elif 'IsolationForest' in model.__class__.__name__:
        prediction = model.predict(X_processed)
    else:
        prediction = [0]

    return int(prediction[0])
