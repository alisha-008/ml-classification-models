"""
Train KNN, Decision Tree, and Naive Bayes on gender classification dataset.
Uses HOG features + PCA + GridSearchCV hyperparameter tuning for best accuracy.
1000 images per class.
"""
import os, pickle, json, random
import numpy as np
from PIL import Image
from skimage.feature import hog
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_PATH = r"c:\Users\Ayesha Akmal\Downloads\archive (2)\Training"
IMG_SIZE     = (128, 128)
N_PER_CLASS  = 1000
MODELS_DIR   = os.path.join(os.path.dirname(__file__), "models")
STATIC_DIR   = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# ── HOG feature extraction ────────────────────────────────────────────────────
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

def load_images(folder, label, n):
    files = [f for f in os.listdir(folder)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.seed(42)
    random.shuffle(files)
    files = files[:n]
    X, y = [], []
    for i, fname in enumerate(files):
        try:
            img = Image.open(os.path.join(folder, fname)).convert('RGB').resize(IMG_SIZE)
            arr = np.array(img)
            feat = extract_hog(arr)
            X.append(feat)
            y.append(label)
        except Exception as e:
            print(f"  Skipping {fname}: {e}")
        if (i + 1) % 200 == 0:
            print(f"    Loaded {i+1}/{len(files)} from {os.path.basename(folder)}")
    return X, y

def save_confusion_matrix(cm, model_name, classes):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax,
                annot_kws={"size": 14})
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(f'{model_name}\nConfusion Matrix', fontsize=13)
    plt.tight_layout()
    path = os.path.join(STATIC_DIR, f'cm_{model_name.lower().replace(" ", "_")}.png')
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path

# ── Load data ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("  Loading & extracting HOG features …")
print("=" * 60)
Xf, yf = load_images(os.path.join(DATASET_PATH, 'female'), 0, N_PER_CLASS)
Xm, ym = load_images(os.path.join(DATASET_PATH, 'male'),   1, N_PER_CLASS)
print(f"\n  Female samples : {len(Xf)}")
print(f"  Male   samples : {len(Xm)}")
print(f"  Feature length : {len(Xf[0])}")

X = np.array(Xf + Xm)
y = np.array(yf + ym)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\n  Train: {len(X_train)}  |  Test: {len(X_test)}")
classes = ['Female', 'Male']

# ── Base pipelines ────────────────────────────────────────────────────────────
base_pipelines = {
    'KNN': Pipeline([
        ('scaler', StandardScaler()),
        ('pca',    PCA(n_components=0.95, random_state=42)),
        ('clf',    KNeighborsClassifier(n_jobs=-1)),
    ]),
    'Decision Tree': Pipeline([
        ('scaler', StandardScaler()),
        ('pca',    PCA(n_components=0.95, random_state=42)),
        ('clf',    DecisionTreeClassifier(random_state=42)),
    ]),
    'Naive Bayes': Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    GaussianNB()),
    ]),
}

# ── Hyperparameter grids ──────────────────────────────────────────────────────
param_grids = {
    'KNN': {
        'pca__n_components': [0.90, 0.95],
        'clf__n_neighbors':  [3, 5, 7, 9, 11],
        'clf__metric':       ['euclidean', 'manhattan'],
        'clf__weights':      ['uniform', 'distance'],
    },
    'Decision Tree': {
        'pca__n_components':      [0.90, 0.95],
        'clf__max_depth':         [10, 20, 30, None],
        'clf__min_samples_split': [2, 5, 10],
        'clf__min_samples_leaf':  [1, 2, 4],
        'clf__criterion':         ['gini', 'entropy'],
    },
    'Naive Bayes': {
        'clf__var_smoothing': [1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6],
    },
}

# ── Tune, train & evaluate ────────────────────────────────────────────────────
results = {}
print("\n" + "=" * 60)
for name, pipeline in base_pipelines.items():
    print(f"\n  Tuning {name} with GridSearchCV (cv=5) …")
    grid = GridSearchCV(
        pipeline,
        param_grids[name],
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)

    best_pipeline = grid.best_estimator_
    print(f"  Best params : {grid.best_params_}")
    print(f"  CV accuracy : {grid.best_score_*100:.2f}%")

    y_pred = best_pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm  = confusion_matrix(y_test, y_pred)
    save_confusion_matrix(cm, name, classes)

    results[name] = {
        'accuracy':    round(float(acc) * 100, 2),
        'best_params': grid.best_params_,
        'cv_accuracy': round(float(grid.best_score_) * 100, 2),
        'cm':          cm.tolist(),
    }
    print(f"  ✓ {name} test accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=classes))

    pkl_name = name.lower().replace(' ', '_') + '.pkl'
    with open(os.path.join(MODELS_DIR, pkl_name), 'wb') as f:
        pickle.dump(best_pipeline, f)
    print(f"  Saved → models/{pkl_name}")

# ── Save results & config ─────────────────────────────────────────────────────
config = {
    'img_size':     list(IMG_SIZE),
    'feature_type': 'hog',
}
with open(os.path.join(MODELS_DIR, 'results.json'), 'w') as f:
    json.dump(results, f, indent=2)
with open(os.path.join(MODELS_DIR, 'config.json'), 'w') as f:
    json.dump(config, f, indent=2)

print("\n" + "=" * 60)
print("  DONE — Summary")
print("=" * 60)
for k, v in results.items():
    print(f"  {k:20s}: {v['accuracy']:.2f}%  (CV: {v['cv_accuracy']:.2f}%)")
best = max(results, key=lambda k: results[k]['accuracy'])
print(f"\n  Best model: {best} ({results[best]['accuracy']:.2f}%)")
print(f"  Models saved to: {MODELS_DIR}")
