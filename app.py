"""
Flask web app for gender classification using KNN, Decision Tree, and Naive Bayes.
Uses HOG features — must match train_models.py preprocessing exactly.
"""
import os, pickle, json, io, base64
import numpy as np
from PIL import Image
from skimage.feature import hog
from flask import Flask, render_template, request, jsonify
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

# Load config saved during training so preprocessing always matches
_cfg_path = os.path.join(MODELS_DIR, 'config.json')
if os.path.exists(_cfg_path):
    with open(_cfg_path) as _f:
        _cfg = json.load(_f)
    IMG_SIZE = tuple(_cfg.get('img_size', [128, 128]))
else:
    IMG_SIZE = (128, 128)

# ── Load models ───────────────────────────────────────────────────────────────
def load_models():
    models = {}
    for key, fname in [('KNN', 'knn.pkl'),
                       ('Decision Tree', 'decision_tree.pkl'),
                       ('Naive Bayes', 'naive_bayes.pkl')]:
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                models[key] = pickle.load(f)
    return models

def load_results():
    path = os.path.join(MODELS_DIR, 'results.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

MODELS  = load_models()
RESULTS = load_results()
CLASSES = ['Female', 'Male']

# ── Image preprocessing — MUST match train_models.py exactly ─────────────────
def extract_hog(img_array):
    gray = np.mean(img_array, axis=2).astype(np.uint8)
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        feature_vector=True
    )
    return features

def preprocess(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert('RGB').resize(IMG_SIZE)
    arr = np.array(img)
    feat = extract_hog(arr)
    return feat.reshape(1, -1)

# ── Confusion matrix as base64 ────────────────────────────────────────────────
def cm_to_base64(cm_list, model_name):
    cm = np.array(cm_list)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'{model_name} – Confusion Matrix')
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    trained = bool(MODELS)
    return render_template('index.html', trained=trained, results=RESULTS)

@app.route('/predict', methods=['POST'])
def predict():
    if not MODELS:
        return jsonify({'error': 'Models not trained yet. Run train_models.py first.'}), 400

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded.'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename.'}), 400

    try:
        data = file.read()
        X = preprocess(data)
    except Exception as e:
        return jsonify({'error': f'Could not process image: {e}'}), 400

    predictions = {}
    for name, clf in MODELS.items():
        pred = clf.predict(X)[0]
        try:
            proba = clf.predict_proba(X)[0]
            conf  = round(float(max(proba)) * 100, 1)
        except Exception:
            conf = None
        predictions[name] = {
            'label':      CLASSES[pred],
            'confidence': conf,
            'accuracy':   RESULTS.get(name, {}).get('accuracy', 'N/A'),
        }

    # Best model by accuracy
    best = max(predictions, key=lambda k: predictions[k]['accuracy'] if isinstance(predictions[k]['accuracy'], float) else 0)

    # Confusion matrices as base64
    cms = {}
    for name, res in RESULTS.items():
        cms[name] = cm_to_base64(res['cm'], name)

    return jsonify({
        'predictions': predictions,
        'best_model':  best,
        'cms':         cms,
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
