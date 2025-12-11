import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# Paths
ART_DIR = r"Z:\COLLEGE NOTES\3RD YEAR\SEM 5\CSC1366-Health Care Data Analytics\PROJECT\WORKSPACE\phase-2\webapp_flask\artifacts_phase3"
scaler_path = ART_DIR + "structured_scaler.pkl"
encoder_path = ART_DIR + "structured_encoder.pkl"
xgb_path = ART_DIR + "structured_xgb.model"
path_csv = r"Z:\COLLEGE NOTES\3RD YEAR\SEM 5\CSC1366-Health Care Data Analytics\PROJECT\WORKSPACE\Dataset\structured_data.csv"
# Load your CSV
df = pd.read_csv(path_csv)  # replace with your file

# Numeric + categorical
X_num = df[["age"]].values
X_cat = df[["gender", "drug", "effects"]].astype(str)

scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(X_num)

encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

X_cat_encoded = encoder.fit_transform(X_cat)

X = np.hstack([X_num_scaled, X_cat_encoded])
y = df["label"].values

# Train XGBoost
dtrain = xgb.DMatrix(X, label=y)
bst = xgb.train({"objective":"binary:logistic","eval_metric":"auc"}, dtrain, num_boost_round=100)

# Save artifacts
import os
os.makedirs(ART_DIR, exist_ok=True)
joblib.dump(scaler, scaler_path)
joblib.dump(encoder, encoder_path)
bst.save_model(xgb_path)

print("Structured artifacts saved.")
