import os
import re
import json
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify
from gensim.models import Word2Vec, FastText
import xgboost as xgb
import spacy
from structured_predict import predict_structured_file

# ---------- Paths ----------
ART_DIR = os.path.abspath(".\\artifacts")
W2V_PATH    = os.path.join(ART_DIR, "w2v_model.model")
FT_PATH     = os.path.join(ART_DIR, "fasttext_model.model")
TFIDF_PATH  = os.path.join(ART_DIR, "tfidf.pkl")
SVD_PATH    = os.path.join(ART_DIR, "svd_300.pkl")
SCALER_PATH = os.path.join(ART_DIR, "scaler.pkl")
XGB_PATH    = os.path.join(ART_DIR, "xgb.model")
THRESH_PATH = os.path.join(ART_DIR, "best_threshold.json")
SYMLEX_PATH = os.path.join(ART_DIR, "symptom_lexicon.json")

# ---------- Load models ----------
nlp = spacy.load("en_core_sci_sm")

w2v = Word2Vec.load(W2V_PATH)
USE_FASTTEXT = os.path.exists(FT_PATH)
ft = FastText.load(FT_PATH) if USE_FASTTEXT else None

tfidf = joblib.load(TFIDF_PATH)
svd = joblib.load(SVD_PATH)
scaler = joblib.load(SCALER_PATH)

with open(THRESH_PATH) as f:
    best_threshold = json.load(f)["best_threshold"]

with open(SYMLEX_PATH) as f:
    symptom_terms = set(json.load(f))

bst = xgb.Booster()
bst.load_model(XGB_PATH)

# ---------- Helpers ----------
def clean_text_simple(s):
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

idf_map = {w: tfidf.idf_[i] for w, i in tfidf.vocabulary_.items()}
idf_default = float(np.median(list(idf_map.values()))) if idf_map else 1.0

def idf_weight(t):
    return 1.0 / (idf_map.get(t, idf_default))

def sentence_vec(tokens, kv, dim):
    vecs, weights = [], []
    for t in tokens:
        try:
            vec = kv[t]
            vecs.append(vec)
            weights.append(idf_weight(t))
        except KeyError:
            continue
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    vecs = np.array(vecs)
    weights = np.array(weights).reshape(-1, 1)
    return (vecs * weights).sum(axis=0) / (weights.sum() + 1e-8)

def sent_embed_pack(raw):
    toks = clean_text_simple(raw).split()
    v_w2v = sentence_vec(toks, w2v.wv, w2v.vector_size)
    if USE_FASTTEXT:
        v_ft = sentence_vec(toks, ft.wv, ft.vector_size)
        return np.concatenate([v_w2v, v_ft])
    return v_w2v

def ner_features(raw):
    doc = nlp(raw)
    total = len(doc.ents)
    labels = ["CHEMICAL", "DISEASE", "GENE_OR_GENE_PRODUCT", "CELL_LINE", "ORGANISM"]
    counts = {l: 0 for l in labels}
    ents = []
    for e in doc.ents:
        ents.append({"text": e.text, "label": e.label_})
        if e.label_ in counts:
            counts[e.label_] += 1
    feat_vec = np.array([total] + [counts[l] for l in labels], dtype=np.float32)
    return feat_vec, ents

def symptom_counts(raw):
    toks = clean_text_simple(raw).split()
    hits = [w for w in toks if w in symptom_terms]
    cnt = len(hits)
    uniq = len(set(hits))
    return np.array([cnt, uniq], dtype=np.float32), sorted(list(set(hits)))

def features_from_text(raw):
    v_emb = sent_embed_pack(raw).reshape(1, -1)
    v_tfidf = tfidf.transform([clean_text_simple(raw)])
    v_svd = svd.transform(v_tfidf)
    v_ner, ents = ner_features(raw)
    v_sym, sym_hits = symptom_counts(raw)
    v_ner = v_ner.reshape(1, -1)
    v_sym = v_sym.reshape(1, -1)
    X = np.hstack([v_emb, v_svd, v_ner, v_sym])
    Xs = scaler.transform(X)
    return Xs, ents, sym_hits

def predict_from_text(raw_text: str):
    Xs, ents, sym_hits = features_from_text(raw_text)
    p = float(bst.predict(xgb.DMatrix(Xs))[0])
    label = int(p >= best_threshold)
    return {
        "probability": p,
        "label": label,
        "threshold": best_threshold,
        "entities": ents,
        "symptom_hits": sym_hits
    }

# ---------- Flask ----------
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    result_file = None
    text_input = ""
    if request.method == "POST":
        text_input = request.form.get("text", "").strip()
        if text_input:
            result = predict_from_text(text_input)
            result["label_text"] = "ADR likely" if result["label"] == 1 else "ADR unlikely"

        file = request.files.get("file")
        if file:
            result_file = predict_structured_file(file)

    return render_template("index.html",
                           result=result,
                           result_file=result_file,
                           text_input=text_input)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Missing 'text'"}), 400
    return jsonify(predict_from_text(text))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
