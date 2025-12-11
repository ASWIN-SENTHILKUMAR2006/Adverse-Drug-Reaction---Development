# phase2_embedding_ner_pipeline.py
import os, re, json, joblib, math
import numpy as np
import pandas as pd

from gensim.models import Word2Vec, FastText
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score, roc_auc_score,
    confusion_matrix, f1_score
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import spacy
nlp = spacy.load("en_core_sci_sm")

# ---------- Config ----------
#DATA_PATH = r"Z:\COLLEGE NOTES\3RD YEAR\SEM 5\CSC1366-Health Care Data Analytics\PROJECT\WORKSPACE\Dataset\ade_classification.csv"
DATA_PATH = r"Z:\COLLEGE NOTES\3RD YEAR\SEM 5\CSC1366-Health Care Data Analytics\PROJECT\WORKSPACE\Dataset\ade_augmented.csv"
ART_DIR  = r"Z:\COLLEGE NOTES\3RD YEAR\SEM 5\CSC1366-Health Care Data Analytics\PROJECT\WORKSPACE\phase-2\webapp_flask\artifacts"
os.makedirs(ART_DIR, exist_ok=True)

W2V_PATH      = os.path.join(ART_DIR, "w2v_model.model")
FT_PATH       = os.path.join(ART_DIR, "fasttext_model.model")
TFIDF_PATH    = os.path.join(ART_DIR, "tfidf.pkl")
SVD_PATH      = os.path.join(ART_DIR, "svd_300.pkl")
SCALER_PATH   = os.path.join(ART_DIR, "scaler.pkl")
XGB_PATH      = os.path.join(ART_DIR, "xgb.model")
THRESH_PATH   = os.path.join(ART_DIR, "best_threshold.json")
SYMLEX_PATH   = os.path.join(ART_DIR, "symptom_lexicon.json")

USE_FASTTEXT  = True           # set False if you don't want FastText
SVD_DIM       = 300            # TF-IDF -> SVD dims
SEED          = 42

# ---------- Load data ----------
df = pd.read_csv(DATA_PATH)
df = df[['text', 'label']].dropna().reset_index(drop=True)
df['label'] = df['label'].astype(int)

# ---------- Cleaning ----------
def clean_text_simple(s: str):
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

df['clean_text'] = df['text'].apply(clean_text_simple)
sentences = [t.split() for t in df['clean_text'].tolist()]

# ---------- Symptom lexicon (tiny starter you can expand later) ----------
symptom_terms = [
    "nausea","vomiting","rash","hives","urticaria","angioedema","anaphylaxis",
    "dizziness","headache","sedation","confusion","bleeding","hemorrhage",
    "diarrhea","constipation","cough","dyspnea","wheezing","palpitations",
    "arrhythmia","qt","nephrotoxicity","hepatotoxicity","myopathy","tremor",
    "seizure","fever","chills","photosensitivity","hyperglycemia","hypoglycemia",
]
with open(SYMLEX_PATH, "w") as f:
    json.dump(symptom_terms, f)

def symptom_counts(text: str):
    t = clean_text_simple(text).split()
    cnt = sum(1 for w in t if w in symptom_terms)
    uniq = len(set(w for w in t if w in symptom_terms))
    return cnt, uniq

# ---------- Train Word2Vec ----------
w2v = Word2Vec(
    sentences=sentences, vector_size=200, window=5,
    min_count=2, workers=4, epochs=15, seed=SEED
)
w2v.save(W2V_PATH)
print("Word2Vec trained and saved ->", W2V_PATH)

# ---------- Train FastText (optional) ----------
if USE_FASTTEXT:
    ft = FastText(
        sentences=sentences, vector_size=200, window=5,
        min_count=2, workers=4, epochs=15, seed=SEED
    )
    ft.save(FT_PATH)
    print("FastText trained and saved ->", FT_PATH)

# ---------- TF-IDF + SVD ----------
tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
X_tfidf = tfidf.fit_transform(df['clean_text'])
joblib.dump(tfidf, TFIDF_PATH)
print("TF-IDF saved ->", TFIDF_PATH, "; shape:", X_tfidf.shape)

svd = TruncatedSVD(n_components=SVD_DIM, random_state=SEED)
X_svd = svd.fit_transform(X_tfidf)
joblib.dump(svd, SVD_PATH)
print("SVD saved ->", SVD_PATH, "; shape:", X_svd.shape)

# ---------- IDF map for weighted sentence embeddings ----------
idf_map = {w: tfidf.idf_[i] for w, i in tfidf.vocabulary_.items()}
idf_default = float(np.median(list(idf_map.values()))) if idf_map else 1.0

def idf_weight(token: str):
    return 1.0 / (idf_map.get(token, idf_default))

def sentence_vec(tokens, kv, dim):
    # Smooth Inverse Frequency (SIF-like) weighting
    weights = []
    vecs = []
    for t in tokens:
        if t in kv:
            w = idf_weight(t)
            weights.append(w)
            vecs.append(kv[t])
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    vecs = np.array(vecs)
    weights = np.array(weights).reshape(-1,1)
    v = (vecs * weights).sum(axis=0) / (weights.sum() + 1e-8)
    return v

def sent_embed_pack(text):
    toks = clean_text_simple(text).split()
    v_w2v = sentence_vec(toks, w2v.wv, w2v.vector_size)
    if USE_FASTTEXT:
        v_ft = sentence_vec(toks, ft.wv, ft.vector_size)
        return np.concatenate([v_w2v, v_ft])
    return v_w2v

# ---------- NER features ----------
def ner_features(raw: str):
    doc = nlp(raw)
    total = len(doc.ents)
    by_label = {}
    for e in doc.ents:
        by_label[e.label_] = by_label.get(e.label_, 0) + 1
    # normalize: pick top common labels from this model
    labels = ["CHEMICAL", "DISEASE", "GENE_OR_GENE_PRODUCT", "CELL_LINE", "ORGANISM"]
    feats = [total] + [by_label.get(l, 0) for l in labels]
    return np.array(feats, dtype=np.float32)

# ---------- Build feature matrix ----------
emb_list, ner_list, sym_list = [], [], []
for raw in df['text'].tolist():
    emb_list.append(sent_embed_pack(raw))
    ner_list.append(ner_features(raw))
    sym_list.append(symptom_counts(raw))

X_emb = np.vstack(emb_list)                           # (N, 200 or 400)
X_ner = np.vstack(ner_list)                           # (N, 1+len(labels))
X_sym = np.array(sym_list, dtype=np.float32)          # (N, 2)
X = np.hstack([X_emb, X_svd, X_ner, X_sym])

y = df['label'].values

# ---------- Scale dense features ----------
scaler = StandardScaler()
X = scaler.fit_transform(X)
joblib.dump(scaler, SCALER_PATH)
print("Scaler saved ->", SCALER_PATH, "; X shape:", X.shape)

# ---------- Split (train/valid/test) ----------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=SEED
)
X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED
)

# ---------- Class imbalance handling ----------
pos = (y_train == 1).sum()
neg = (y_train == 0).sum()
scale_pos_weight = neg / max(pos, 1)
print(f"scale_pos_weight ≈ {scale_pos_weight:.3f} (neg={neg}, pos={pos})")

# ---------- Train XGBoost ----------
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)
dtest  = xgb.DMatrix(X_test,  label=y_test)

params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 6,
    "eta": 0.08,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "min_child_weight": 1,
    "reg_lambda": 1.0,
    "scale_pos_weight": scale_pos_weight,
    "seed": SEED,
    "verbosity": 1
}

bst = xgb.train(
    params,
    dtrain,
    num_boost_round=1200,
    evals=[(dtrain,"train"), (dvalid,"valid")],
    early_stopping_rounds=60,
    verbose_eval=50
)

bst.save_model(XGB_PATH)
print("XGBoost saved ->", XGB_PATH)

# ---------- Threshold tuning (maximize F1 on valid) ----------
p_valid = bst.predict(dvalid, iteration_range=(0, bst.best_iteration+1))
best_t, best_f1 = 0.5, 0.0
for t in np.linspace(0.2, 0.8, 61):
    f1 = f1_score(y_valid, (p_valid >= t).astype(int))
    if f1 > best_f1:
        best_f1, best_t = f1, float(t)

with open(THRESH_PATH, "w") as f:
    json.dump({"best_threshold": best_t}, f)
print(f"Best threshold (valid) = {best_t:.3f} ; best F1 = {best_f1:.4f}")

# ---------- Final evaluation on test ----------
p_test = bst.predict(dtest, iteration_range=(0, bst.best_iteration+1))
y_pred = (p_test >= best_t).astype(int)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, p_test))
print("Classification report:\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
