import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Paths for artifacts
ART_DIR = "./artifacts_phase3/"
SCALER_PATH = ART_DIR + "structured_scaler.pkl"
ENCODER_PATH = ART_DIR + "structured_encoder.pkl"
XGB_PATH = ART_DIR + "structured_xgb.model"

# Load artifacts
scaler = joblib.load(SCALER_PATH)
encoder = joblib.load(ENCODER_PATH)
bst = xgb.Booster()
bst.load_model(XGB_PATH)

def preprocess_structured(df):
    # Numeric: scale age
    df_numeric = scaler.transform(df[["age"]])

    # Categorical: gender, drug, effects
    cat_features = df[["gender", "drug", "effects"]].astype(str)
    df_cat = encoder.transform(cat_features)

    X = np.hstack([df_numeric, df_cat])
    return X

def predict_structured_file(file):
    df = pd.read_csv(file)
    X = preprocess_structured(df)
    dmatrix = xgb.DMatrix(X)
    probs = bst.predict(dmatrix)
    labels = (probs >= 0.5).astype(int)

    # Return results
    results = []
    for i, row in df.iterrows():
        results.append({
            "row": i+1,
            "age": row.get("age"),
            "gender": row.get("gender"),
            "drug": row.get("drug"),
            "effects": row.get("effects"),
            "probability": float(probs[i]),
            "label": int(labels[i])
        })
    return results
