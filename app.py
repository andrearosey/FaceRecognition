import pandas as pd
import streamlit as st
import numpy as np
import cv2
import csv
import re
import shutil
import time
from collections import deque
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

st.set_page_config(layout="wide", page_title="Library Face Recognition", page_icon="📚")

# ----------------------------------------------------------------------
# Global theme / styling
# ----------------------------------------------------------------------
def inject_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

        html, body, [class*="css"]  { font-family: 'Poppins', sans-serif; }

        :root {
            --bg-app: #0e0e13;
            --bg-card: #1b1b24;
            --bg-card-hover: #22222e;
            --border-soft: rgba(255,255,255,0.07);
            --text-primary: #f2f2f5;
            --text-secondary: #9a9aa8;
        }

        /* App background + main containers -- everything dark, no separate
           theme/config file needed, it all lives in this one CSS block */
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stBottomBlockContainer"],
        section[data-testid="stSidebar"] {
            background-color: var(--bg-app) !important;
            color: var(--text-primary) !important;
        }
        [data-testid="stHeader"], [data-testid="stToolbar"] { background: transparent !important; }

        h1, h2, h3, h4, h5, h6, p, span, label, li,
        .stMarkdown, .stCaption, .stText, .stJson {
            color: var(--text-primary);
        }

        /* Gradient hero banner behind the title */
        .hero-banner {
            background: linear-gradient(120deg, #3a4f7a 0%, #4d3866 50%, #2f6b4f 100%);
            padding: 1.6rem 2rem;
            border-radius: 16px;
            color: white;
            margin-bottom: 1.2rem;
            box-shadow: 0 8px 28px rgba(0, 0, 0, 0.45);
            border: 1px solid rgba(255,255,255,0.06);
        }
        .hero-banner h1 { margin: 0; font-weight: 700; font-size: 1.9rem; color: white; }
        .hero-banner p { margin: 0.3rem 0 0 0; opacity: 0.85; font-size: 0.95rem; color: white; }

        /* KPI cards */
        .kpi-card {
            background: var(--bg-card);
            border-radius: 14px;
            padding: 1rem 1.1rem;
            box-shadow: 0 4px 18px rgba(0,0,0,0.35);
            border: 1px solid var(--border-soft);
            border-left: 5px solid var(--accent, #6A8FE0);
            transition: transform 0.15s ease, box-shadow 0.15s ease, background 0.15s ease;
            height: 108px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            overflow: hidden;
        }
        .kpi-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 26px rgba(0,0,0,0.5);
            background: var(--bg-card-hover);
        }
        .kpi-card .kpi-label { font-size: 0.8rem; color: var(--text-secondary); font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }
        .kpi-card .kpi-value { font-size: 1.7rem; font-weight: 700; color: var(--text-primary); margin-top: 0.2rem; }
        .kpi-card .kpi-sub { font-size: 0.78rem; color: var(--text-secondary); margin-top: 0.15rem; }

        /* Buttons */
        div.stButton > button, div.stDownloadButton > button {
            border-radius: 10px;
            font-weight: 600;
            transition: transform 0.1s ease, box-shadow 0.1s ease;
            background-color: var(--bg-card);
            color: var(--text-primary);
            border: 1px solid var(--border-soft);
        }
        div.stButton > button:hover, div.stDownloadButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 16px rgba(0,0,0,0.4);
            border-color: rgba(255,255,255,0.18);
        }
        div.stButton > button[kind="primary"] {
            background: linear-gradient(120deg, #4C72B0, #6A4C93);
            color: white;
            border: none;
        }

        /* Tabs */
        button[data-baseweb="tab"] { font-weight: 600; color: var(--text-secondary); }
        button[data-baseweb="tab"][aria-selected="true"] { color: var(--text-primary); }
        [data-baseweb="tab-highlight"] { background-color: #6A4C93; }
        [data-baseweb="tab-border"] { background-color: var(--border-soft); }
        [data-testid="stTabs"] { color: var(--text-primary); }

        /* Radio controls: keep horizontal selections on one line */
        div[data-testid="stRadio"] label, div[data-testid="stRadio"] p {
            color: var(--text-primary) !important;
            white-space: nowrap !important;
        }
        div[data-testid="stRadio"] > div {
            gap: 0.4rem;
            flex-wrap: nowrap !important;
            overflow-x: auto;
        }

        /* Checkbox */
        div[data-testid="stCheckbox"] label p { color: var(--text-primary) !important; }

        /* Selectbox / dropdowns / text & number inputs */
        div[data-baseweb="select"] > div,
        div[data-baseweb="popover"] ul,
        .stTextInput > div > div input,
        .stNumberInput input,
        .stTextArea textarea {
            background-color: var(--bg-card) !important;
            color: var(--text-primary) !important;
            border-color: var(--border-soft) !important;
        }
        div[data-baseweb="popover"] li { color: var(--text-primary) !important; }

        /* Slider */
        div[data-testid="stSlider"] label { color: var(--text-primary) !important; }

        /* Progress bar */
        div[data-testid="stProgress"] > div > div { background-color: var(--bg-card); }
        div[data-testid="stProgress"] > div > div > div { background-image: linear-gradient(90deg, #4C72B0, #6A4C93); }

        /* Metric widget */
        [data-testid="stMetric"] { background: var(--bg-card); border-radius: 10px; padding: 0.6rem 0.9rem; border: 1px solid var(--border-soft); }
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"], [data-testid="stMetricDelta"] { color: var(--text-primary) !important; }

        /* Forms and expanders */
        div[data-testid="stForm"], details {
            background-color: var(--bg-card);
            border-radius: 12px;
            border: 1px solid var(--border-soft);
        }
        summary { color: var(--text-primary) !important; }

        /* Dataframe polish */
        .stDataFrame, [data-testid="stDataFrame"] {
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--border-soft);
        }

        /* Alerts / info / success / warning / error / toast boxes */
        div[data-testid="stAlert"], div[data-testid="stToast"] {
            border-radius: 10px;
            background-color: var(--bg-card);
            color: var(--text-primary);
        }

        /* Divider */
        hr { border-color: var(--border-soft); }
    </style>
    """, unsafe_allow_html=True)


def kpi_card(label, value, sub="", accent="#4C72B0"):
    st.markdown(
        f"""
        <div class="kpi-card" style="--accent: {accent}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def disable_selectbox_typing():
    """Makes every st.selectbox behave like a plain click/tap dropdown --
    the search input still exists (BaseWeb's combobox), but is set
    read-only via JS so letters can't be typed into it. Arrow keys,
    Enter and mouse clicks still work for picking an option. Re-applied
    on every DOM mutation since Streamlit re-renders selectboxes on rerun."""
    st.components.v1.html(
        """
        <script>
        function disableTyping() {
            const doc = window.parent.document;
            const inputs = doc.querySelectorAll('div[data-baseweb="select"] input');
            inputs.forEach(function(input) {
                input.readOnly = true;
            });
        }
        disableTyping();
        const target = window.parent.document.body;
        if (target && !window.__selectboxTypingObserverAttached) {
            const observer = new MutationObserver(disableTyping);
            observer.observe(target, { childList: true, subtree: true });
            window.__selectboxTypingObserverAttached = true;
        }
        </script>
        """,
        height=0,
    )


inject_custom_css()
disable_selectbox_typing()

# ----------------------------------------------------------------------
# Paths / constants
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
INFO_PATH = BASE_DIR / "data.csv"
LOG_PATH = BASE_DIR / "entrance_log.csv"
CONSENT_LOG_PATH = BASE_DIR / "consent_log.csv"  # audit trail of biometric-data consent
STORED_FACES_DIR = BASE_DIR / "stored-faces"
REGISTERED_MEMBER_IDS_PATH = BASE_DIR / "registered_member_ids.csv"  # IDs added through this app

# --- CNN algorithm paths ---
CNN_DIR = MODEL_DIR / "CNN"
CNN_MODEL_PATH = CNN_DIR / "face_cnn_model.keras"
LABEL_CLASSES_PATH = CNN_DIR / "label_classes.npy"

# --- FaceNet-Style Embedding + ArcFace algorithm paths ---
FACENET_DIR = BASE_DIR / "models2"
FACENET_MODEL_PATH = FACENET_DIR / "face_embedding_model.keras"
FACENET_PROTOTYPE_IDS_PATH = FACENET_DIR / "prototype_ids.npy"
FACENET_PROTOTYPE_VECS_PATH = FACENET_DIR / "prototype_vecs.npy"
FACENET_IMG_SIZE = 100  

# --- SVM algorithm paths ---
SVM_DIR = MODEL_DIR / "SVM + HOG"
SVM_PIPELINE_PATH = SVM_DIR / "face_svm_hog_pipeline.pkl"
SVM_LABELS_PATH = SVM_DIR / "classes_hog.npy"

FACE_INPUT_SIZE = (100, 100)  

# --- YuNet face detector ---
YUNET_DIR = MODEL_DIR / "yunet"
YUNET_MODEL_PATH = YUNET_DIR / "face_detection_yunet_2023mar.onnx"
YUNET_MODEL_URL = (
    "https://raw.githubusercontent.com/opencv/opencv_zoo/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
YUNET_SCORE_THRESHOLD = 0.7   # min confidence to keep a detection
YUNET_NMS_THRESHOLD = 0.3     # non-max suppression IoU threshold
YUNET_TOP_K = 5000            # max detections kept before NMS
YUNET_MIN_FACE_SIZE = 60      # matches the training notebooks' Haar minSize=(60, 60)

# --- Mask detector (pretrained, runs before any recognition algorithm) ---
MASK_DIR = MODEL_DIR / "mask"
MASK_MODEL_PATH = MASK_DIR / "mask_detector.h5"  # .h5 extension matters -- see load_mask_algorithm()
MASK_MODEL_URL = "https://raw.githubusercontent.com/chandrikadeb7/Face-Mask-Detection/master/mask_detector.model"
MASK_IMG_SIZE = 224  # input size expected by this pretrained model

# --- Thresholds: loaded from the confidence_threshold.npy files each training notebook saves ---
CNN_THRESHOLD_PATH = CNN_DIR / "confidence_threshold.npy"
FACENET_CONFIDENCE_THRESHOLD_PATH = FACENET_DIR / "confidence_threshold.npy"
FACENET_MARGIN_THRESHOLD_PATH = FACENET_DIR / "margin_threshold.npy"
SVM_THRESHOLD_PATH = SVM_DIR / "confidence_threshold_hog.npy"


def load_threshold(path, default):
    try:
        return float(np.load(path))
    except (FileNotFoundError, OSError, ValueError):
        st.warning(
            f"No saved threshold at {path.name} -- using fallback default {default}. "
            "Re-run that notebook's training + threshold-selection cells to compute a tuned value."
        )
        return default


CNN_CONF_THRESHOLD = load_threshold(CNN_THRESHOLD_PATH, 0.85)
FACENET_CONFIDENCE_THRESHOLD = load_threshold(FACENET_CONFIDENCE_THRESHOLD_PATH, 0.82)
FACENET_MARGIN_THRESHOLD = load_threshold(FACENET_MARGIN_THRESHOLD_PATH, 0.08)
SVM_CONF_THRESHOLD = load_threshold(SVM_THRESHOLD_PATH, 0.65)

CONFIRM_HOLD_SECONDS = 1.5   # confirmation period before granting access
COOLDOWN_MINUTES = 5         # don't re-log the same person again within this window

# Live recognition stabilization: absorbs brief low-confidence frames caused by
# small movements, facial expressions, lighting changes, or Haar crop jitter.
STABLE_WINDOW = 12
STABLE_MIN_VOTES = 6
IDENTITY_HOLD_SECONDS = 2.0
NO_FACE_RESET_SECONDS = 3.0  # forgiving of brief face-detection loss while moving

# --- Registration settings ---
CAPTURE_TARGET = 70              # photos captured per new member
CAPTURE_INTERVAL_SECONDS = 0.15  # min gap between accepted captures, for pose/lighting variety


# ----------------------------------------------------------------------
# Shared utilities
# ----------------------------------------------------------------------
@st.cache_resource
def load_detector():
    YUNET_DIR.mkdir(parents=True, exist_ok=True)

    if not YUNET_MODEL_PATH.exists():
        import urllib.request
        urllib.request.urlretrieve(YUNET_MODEL_URL, YUNET_MODEL_PATH)

    return cv2.FaceDetectorYN.create(
        str(YUNET_MODEL_PATH),
        "",
        (320, 320),
        YUNET_SCORE_THRESHOLD,
        YUNET_NMS_THRESHOLD,
        YUNET_TOP_K,
    )

@st.cache_data
def load_member_info():

    if not INFO_PATH.exists():
        return {}

    df = pd.read_csv(INFO_PATH)
    df.columns = df.columns.str.strip()

    info = {}

    for _, row in df.iterrows():
        pid = str(row["ID"]).strip()
        info[pid] = {}

        for col in df.columns:
            if col == "ID":
                continue

            if pd.isna(row[col]):
                continue

            info[pid][col] = str(row[col])

    return info

try:
    detector = load_detector()
except Exception:
    st.error(
        "Couldn't load the YuNet face detector (no internet access to "
        f"download it on first run?). Manually place the model at "
        f"{YUNET_MODEL_PATH} and reload the app."
    )
    st.stop()


def detect_faces_yunet(frame):
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)

    boxes = []
    if faces is not None:
        for f in faces:
            x, y, fw, fh = f[:4]
            x, y, fw, fh = int(round(x)), int(round(y)), int(round(fw)), int(round(fh))
            if fw >= YUNET_MIN_FACE_SIZE and fh >= YUNET_MIN_FACE_SIZE:
                boxes.append((max(x, 0), max(y, 0), fw, fh))
    return boxes


member_info = load_member_info()
# case-insensitive fallback lookup, in case data.csv IDs differ only in casing/whitespace
_member_info_upper = {k.upper(): v for k, v in member_info.items()}


@st.cache_resource
def load_mask_algorithm():
    MASK_DIR.mkdir(parents=True, exist_ok=True)

    if not MASK_MODEL_PATH.exists():
        import urllib.request
        try:
            urllib.request.urlretrieve(MASK_MODEL_URL, MASK_MODEL_PATH)
        except Exception:
            return None

    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
    from tensorflow.keras.models import Model

    base = MobileNetV2(weights=None, include_top=False,
                        input_tensor=Input(shape=(MASK_IMG_SIZE, MASK_IMG_SIZE, 3)))
    head = base.output
    head = AveragePooling2D(pool_size=(7, 7))(head)
    head = Flatten(name="flatten")(head)
    head = Dense(128, activation="relu")(head)
    head = Dropout(0.5)(head)
    head = Dense(2, activation="softmax")(head)
    mask_model = Model(inputs=base.input, outputs=head)
    mask_model.load_weights(MASK_MODEL_PATH)

    return {"mask_model": mask_model}


def is_wearing_mask(face_bgr, mask_artifacts):
    import tensorflow as tf

    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (MASK_IMG_SIZE, MASK_IMG_SIZE))
    face_array = tf.keras.preprocessing.image.img_to_array(face_resized)
    face_array = tf.keras.applications.mobilenet_v2.preprocess_input(face_array)
    face_array = np.expand_dims(face_array, axis=0)

    mask_prob, without_mask_prob = mask_artifacts["mask_model"].predict(face_array, verbose=0)[0]
    return mask_prob > without_mask_prob


mask_artifacts = load_mask_algorithm()


def get_info(person_id):
    """Look up a person's info row, tolerant of case/whitespace mismatches."""
    if person_id is None:
        return {}
    key = str(person_id).strip()
    if key in member_info:
        return member_info[key]
    return _member_info_upper.get(key.upper(), {})


def display_name(person_id):
    info = get_info(person_id)
    # fall back to the raw ID rather than "Unknown" -- a matched face should
    # never be labelled "Unknown" just because data.csv has no Name column
    # for them.
    return info.get("Name", person_id)


def is_active_member(person_id):
    """True if this ID still has a row in data.csv (i.e. is still an
    enrolled, active member). Used to gate recognition results -- CNN and
    SVM+HOG bake identities into their trained weights at training time, and
    FaceNet-Style Embedding + ArcFace prototypes persist in
    prototype_ids.npy/prototype_vecs.npy independently of data.csv, so none
    of the recognition algorithms automatically "forget" someone who was
    removed. This check is what makes a deleted member actually get denied."""
    if person_id is None:
        return False
    key = str(person_id).strip()
    return key in member_info or key.upper() in _member_info_upper


def preprocess_face_cnn(gray_crop):
    resized = cv2.resize(gray_crop, FACE_INPUT_SIZE)
    x = resized.astype("float32") / 255.0
    return x.reshape(1, FACE_INPUT_SIZE[0], FACE_INPUT_SIZE[1], 1)

@st.cache_resource
def get_hog_clahe():
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def preprocess_face_svm(gray_crop):
    clahe = get_hog_clahe()
    face = cv2.resize(gray_crop, (100, 100))
    face = clahe.apply(face)
    face = face.astype(np.float32) / 255.0
    return face


def extract_hog_features(face_norm):
    from skimage.feature import hog
    return hog(
        face_norm,
        orientations=12,
        pixels_per_cell=(6, 6),
        cells_per_block=(3, 3),
        block_norm="L2-Hys",
        transform_sqrt=True,
        visualize=False,
    )

# ----------------------------------------------------------------------
# Logging (algorithm-agnostic)
# ----------------------------------------------------------------------
def expected_log_header():
    headers = ["Timestamp", "Algorithm", "Confidence", "ID"]
    if member_info:
        # union of all columns across every person, in case some rows have
        # extra fields others don't (e.g. only some members have "Department")
        seen = []
        for row in member_info.values():
            for key in row:
                if key not in seen:
                    seen.append(key)
        headers.extend(seen)
    return headers


def ensure_log_file():
    expected = expected_log_header()

    if not LOG_PATH.exists():
        with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(expected)
        return

    # migrate an old log file whose header predates data.csv / has fewer
    # columns than we now know about, instead of silently dropping fields
    with open(LOG_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    existing_header = rows[0] if rows else []

    if existing_header != expected:
        backup_path = LOG_PATH.with_name(
            f"{LOG_PATH.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        LOG_PATH.rename(backup_path)
        with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(expected)


def append_log(algorithm, person_id, score):
    ensure_log_file()

    info = get_info(person_id)
    header = expected_log_header()
    info_columns = header[4:]  # everything after Timestamp/Algorithm/ID/Confidence

    row = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), algorithm, f"{score:.3f}", person_id]
    for col in info_columns:
        row.append(info.get(col, ""))

    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def read_log_tail(n=15):
    ensure_log_file()
    with open(LOG_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    return header, body[-n:][::-1]


# ========================================================================
# ALGORITHM: CNN
# ========================================================================
@st.cache_resource
def load_cnn_algorithm():
    missing = [p.name for p in [CNN_MODEL_PATH, LABEL_CLASSES_PATH] if not p.exists()]
    if missing:
        return None
    import tensorflow as tf
    cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
    label_classes = np.load(LABEL_CLASSES_PATH, allow_pickle=True).astype(str)
    return {"cnn_model": cnn_model, "label_classes": label_classes}


def recognize_cnn(gray_crop, color_crop_rgb, artifacts):
    x = preprocess_face_cnn(gray_crop)

    probs = artifacts["cnn_model"].predict(x, verbose=0)[0]
    best_idx = int(np.argmax(probs))
    cnn_conf = float(probs[best_idx])
    candidate = artifacts["label_classes"][best_idx]

    matched = cnn_conf >= CNN_CONF_THRESHOLD

    return {
        "matched": matched,
        "person_id": candidate if matched else None,
        "score": cnn_conf,
    }


# ========================================================================
# ALGORITHM: FaceNet-Style Embedding + ArcFace
# ========================================================================
@st.cache_resource
def load_facenet_algorithm():
    missing = [p.name for p in [FACENET_MODEL_PATH, FACENET_PROTOTYPE_IDS_PATH, FACENET_PROTOTYPE_VECS_PATH]
               if not p.exists()]
    if missing:
        return None
    import tensorflow as tf
    embedding_model = tf.keras.models.load_model(FACENET_MODEL_PATH)
    prototype_ids = np.load(FACENET_PROTOTYPE_IDS_PATH, allow_pickle=True).astype(str)
    prototype_vecs = np.load(FACENET_PROTOTYPE_VECS_PATH)
    return {
        "embedding_model": embedding_model,
        "prototype_ids": prototype_ids,
        "prototype_vecs": prototype_vecs,
    }


def facenet_embed(color_crop_rgb, embedding_model):
    """color_crop_rgb: any-size RGB numpy array. Returns a single L2-normalized
    embedding vector, using the same preprocessing as the training notebook."""
    face = cv2.resize(color_crop_rgb, (FACENET_IMG_SIZE, FACENET_IMG_SIZE))
    face = face.astype(np.float32) / 255.0
    face = np.expand_dims(face, axis=0)  # (1, IMG_SIZE, IMG_SIZE, 3)
    return embedding_model.predict(face, verbose=0)[0]


def recognize_facenet(gray_crop, color_crop_rgb, artifacts):
    embedding = facenet_embed(color_crop_rgb, artifacts["embedding_model"])

    similarities = artifacts["prototype_vecs"] @ embedding
    order = np.argsort(similarities)[::-1]
    best_idx = order[0]
    best_sim = float(similarities[best_idx])

    if len(order) > 1:
        margin = best_sim - float(similarities[order[1]])
    else:
        margin = 1.0

    matched = best_sim >= FACENET_CONFIDENCE_THRESHOLD and margin >= FACENET_MARGIN_THRESHOLD

    return {
        "matched": matched,
        "person_id": artifacts["prototype_ids"][best_idx] if matched else None,
        "score": best_sim,
    }


# ========================================================================
# ALGORITHM: SVM — HOG + StandardScaler + PCA + RBF SVM
# ========================================================================

@st.cache_resource
def load_svm_algorithm():
    required = [SVM_PIPELINE_PATH, SVM_LABELS_PATH]
    missing = [p.name for p in required if not p.exists()]
    if missing:
        return None

    import joblib

    pipeline = joblib.load(SVM_PIPELINE_PATH)
    classes = np.load(SVM_LABELS_PATH, allow_pickle=True).astype(str)

    return {
        "pipeline": pipeline,
        "classes": classes,
    }


def recognize_svm(gray_crop, color_crop_rgb, artifacts):
    # The notebook's HOG pipeline expects the grayscale face after: resize -> CLAHE -> normalization
    face = preprocess_face_svm(gray_crop)

    hog_features = extract_hog_features(face).reshape(1, -1)

    # The pipeline scales and PCA-reduces the raw HOG features internally, then runs the SVC
    probs = artifacts["pipeline"].predict_proba(hog_features)[0]
    best_pos = int(np.argmax(probs))
    confidence = float(probs[best_pos])

    encoded_class = int(artifacts["pipeline"].classes_[best_pos])
    if 0 <= encoded_class < len(artifacts["classes"]):
        candidate = artifacts["classes"][encoded_class]
    else:
        candidate = str(encoded_class)

    matched = confidence >= SVM_CONF_THRESHOLD

    return {
        "matched": matched,
        "person_id": candidate if matched else None,
        "score": confidence,
    }


# ========================================================================
# Algorithm registry — add new algorithms here
# ========================================================================
ALGORITHMS = {
    "CNN": {
        "loader": load_cnn_algorithm,
        "recognize": recognize_cnn,
        "setup_hint": "Expected in models/cnn/: face_cnn_model.keras, label_classes.npy",
    },
    "FaceNet-Style Embedding + ArcFace": {
        "loader": load_facenet_algorithm,
        "recognize": recognize_facenet,
        "setup_hint": "Expected in FaceNet-Style Embedding + ArcFace/: face_embedding_model.keras, "
                      "prototype_ids.npy, prototype_vecs.npy",
    },
    "SVM + HOG": {
        "loader": load_svm_algorithm,
        "recognize": recognize_svm,
        "setup_hint": "Expected in models/SVM + HOG/: face_svm_hog_pipeline.pkl, classes_hog.npy",
    },
}

MODEL_EVALUATION = pd.DataFrame(
    {
        "accuracy": [0.87, 0.84, 0.74],
        "precision": [0.89, 0.84, 0.75],
        "recall": [0.88, 0.83, 0.74],
        "f1": [0.88, 0.83, 0.73],
    },
    index=["CNN", "FaceNet-Style Embedding + ArcFace", "SVM + HOG"],
)

MODEL_CHART_COLORS = {"CNN": "#4C72B0", "FaceNet-Style Embedding + ArcFace": "#55A868", "SVM + HOG": "#B0B0B0"}
METRIC_LABELS = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1"}

# --- Per-model charts, exported as PNGs directly from each training
MODEL_CHARTS_DIR = MODEL_DIR / "charts_images"

# (filename, caption) per model, in the order each notebook produces them.
MODEL_CHARTS = {
    "CNN": [
        ("training_curves.png", "Accuracy / loss over training epochs"),
        ("confusion_matrix.png", "Confusion matrix (test set)"),
        ("roc_pr_threshold.png", "ROC & precision-recall curves used to pick the confidence threshold (validation set)"),
    ],
    "FaceNet-Style Embedding + ArcFace": [
        ("training_curves.png", "ArcFace-style embedding accuracy / loss over training epochs"),
        ("confusion_matrix.png", "Confusion matrix (test set, nearest-prototype matching)"),
        ("roc_pr_confidence_threshold.png", "ROC & precision-recall curves used to pick the confidence threshold (validation set)"),
        ("margin_threshold_tuning.png", "F1 vs. candidate margin threshold, with confidence threshold fixed (validation set)"),
    ],
    "SVM + HOG": [
        ("confusion_matrix.png", "HOG + SVM confusion matrix (test set)"),
        ("roc_pr_threshold.png", "ROC & precision-recall curves used to pick the confidence threshold (out-of-fold training predictions)"),
    ],
}

def render_model_comparison_page():
    st.title("📊 Model Comparison")

    results = MODEL_EVALUATION.sort_values(by="accuracy", ascending=False, na_position="last").round(3)

    # KPI row -- best model per metric, at a glance
    kpi_cols = st.columns(4)
    for col, metric in zip(kpi_cols, METRIC_LABELS.keys()):
        best_model = results[metric].idxmax()
        best_val = results[metric].max()
        with col:
            kpi_card(
                METRIC_LABELS[metric],
                f"{best_val:.2f}",
                f"best: {best_model}",
                accent=MODEL_CHART_COLORS.get(best_model, "#4C72B0"),
            )

    view_tab, table_tab = st.tabs(["📈 Interactive charts", "🔢 Raw table"])

    with table_tab:
        st.dataframe(results.fillna("—"), use_container_width=True)

    with view_tab:
        metrics = list(METRIC_LABELS.keys())

        fig = go.Figure()
        for metric in metrics:
            fig.add_trace(go.Bar(
                name=METRIC_LABELS[metric],
                x=list(results.index),
                y=[results.loc[model_name, metric] for model_name in results.index],
                text=[f"{results.loc[model_name, metric]:.2f}" for model_name in results.index],
                textposition="outside",
                hovertemplate="%{x}: %{y:.2f}<extra>" + METRIC_LABELS[metric] + "</extra>",
            ))
        fig.update_layout(
            barmode="group",
            xaxis_title="Model",
            yaxis=dict(range=[0, 1], title="Score"),
            legend_title_text="Metric",
            margin=dict(t=30, b=10),
            height=440,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#1b1b24",
            font=dict(color="#f2f2f5"),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.title("📈 Per-model evaluation charts")

    tabs = st.tabs(list(MODEL_CHARTS.keys()))
    for tab, model_name in zip(tabs, MODEL_CHARTS.keys()):
        with tab:
            charts = MODEL_CHARTS[model_name]
            chart_folder = {
                "CNN": "cnn",
                "FaceNet-Style Embedding + ArcFace": "FaceNet-Style Embedding + ArcFace",
                "SVM + HOG": "SVM + HOG",
            }[model_name]
            model_dir = MODEL_CHARTS_DIR / chart_folder
            any_shown = False

            for filename, caption in charts:
                img_path = model_dir / filename
                if img_path.exists():
                    display_width = 1100 if (
                        "roc_pr_" in filename
                        or filename == "training_curves.png"
                    ) else 650
                    st.image(
                        str(img_path),
                        caption=caption,
                        width=display_width,
                    )
                    any_shown = True
                else:
                    st.warning(
                        f"Missing chart: {img_path} -- re-export it from "
                        f"{model_name}'s training notebook and place it there."
                    )

            if not any_shown:
                st.info(f"No charts found yet for {model_name} in {model_dir}.")

# ========================================================================
# Registration — capture only; embedding/prototype work
# ========================================================================
def get_data_csv_columns():
    """Column order to use for data.csv -- read the existing file's
    header if present, otherwise fall back to a sensible default."""
    if INFO_PATH.exists():
        return list(pd.read_csv(INFO_PATH, nrows=0).columns.str.strip())
    return ["ID", "Name"]

# Registration fields that should render as a dropdown instead of free text.
# GENDER_OPTIONS is fixed; FACULTY_OPTIONS_FALLBACK is used only if data.csv
# has no Faculty values yet.
GENDER_OPTIONS = ["Male", "Female"]
FACULTY_OPTIONS_FALLBACK = ["FAFB", "FCCI", "FOAS", "FOBE", "FOCS", "FOET"]

# Letters, spaces, hyphens, apostrophes, and periods only (e.g. "Mary-Jane O'Neil").
NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z .'-]*$")


def get_dropdown_options(column):
    """Distinct existing values for a data.csv column, used to populate a
    registration dropdown. Falls back to a sensible default list if the
    column/file doesn't exist yet or has no values."""
    fallback = GENDER_OPTIONS if column == "Gender" else FACULTY_OPTIONS_FALLBACK

    if not INFO_PATH.exists():
        return fallback

    df = pd.read_csv(INFO_PATH)
    df.columns = df.columns.str.strip()
    if column not in df.columns:
        return fallback

    values = sorted({v.strip() for v in df[column].dropna().astype(str) if v.strip()})
    return values or fallback


def get_newly_registered_member_ids():
    """Return IDs that were registered through this app.

    Existing members already present in data.csv are intentionally not included,
    so the Manage Members page can only delete members added through the
    registration workflow.
    """
    if not REGISTERED_MEMBER_IDS_PATH.exists():
        return set()

    try:
        df = pd.read_csv(REGISTERED_MEMBER_IDS_PATH)
        if "ID" not in df.columns:
            return set()
        return {
            str(value).strip()
            for value in df["ID"].dropna().astype(str)
            if str(value).strip()
        }
    except Exception:
        return set()


def mark_as_newly_registered(person_id):
    """Persistently mark a member as registered through this app."""
    registered_ids = get_newly_registered_member_ids()
    key = str(person_id).strip()
    if key:
        registered_ids.add(key)
    pd.DataFrame({"ID": sorted(registered_ids)}).to_csv(
        REGISTERED_MEMBER_IDS_PATH, index=False
    )


def unmark_newly_registered(person_id):
    """Remove an ID from the app-registration list after deletion."""
    key = str(person_id).strip()
    registered_ids = {
        pid for pid in get_newly_registered_member_ids()
        if pid != key
    }
    pd.DataFrame({"ID": sorted(registered_ids)}).to_csv(
        REGISTERED_MEMBER_IDS_PATH, index=False
    )


def append_member_row(field_values):
    """Append one new member's details to data.csv, matching its existing columns."""
    columns = get_data_csv_columns()

    if INFO_PATH.exists():
        df = pd.read_csv(INFO_PATH)
        df.columns = df.columns.str.strip()
    else:
        df = pd.DataFrame(columns=columns)

    row = {col: field_values.get(col, "") for col in columns}
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(INFO_PATH, index=False)


def record_consent(person_id):
    """Appends a timestamped consent record to consent_log.csv.

    Kept as a separate audit trail (rather than a column in data.csv) so it
    survives independently of the member schema and gives a clear, append-only
    record that this person explicitly checked the consent box before any
    facial images were captured -- required since facial images are
    sensitive biometric data.
    """
    is_new_file = not CONSENT_LOG_PATH.exists()
    with open(CONSENT_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new_file:
            writer.writerow(["Timestamp", "ID", "ConsentGiven"])
        writer.writerow([datetime.now().isoformat(timespec="seconds"), person_id, True])


def save_face_photo(person_id, face_bgr, index):
    """Saves one captured face crop to disk, mirroring the
    stored-faces/<ID>/ layout the training notebook reads from."""
    person_dir = STORED_FACES_DIR / str(person_id)
    person_dir.mkdir(parents=True, exist_ok=True)
    filepath = person_dir / f"{person_id}_{index:03d}.jpg"
    cv2.imwrite(str(filepath), face_bgr)
    return filepath


def generate_new_id(role):
    """Auto-assigns the next member ID for the given role, based on data.csv --
    Student -> last S0xx + 1, Staff -> last E0xx + 1 (matches the training
    notebook's ID convention). Falls back to <prefix>001 if no members of
    that role exist yet."""
    prefix = "E" if role == "Staff" else "S"
    width = 3

    if not INFO_PATH.exists():
        return f"{prefix}{1:0{width}d}"

    df = pd.read_csv(INFO_PATH)
    df.columns = df.columns.str.strip()
    if "Role" not in df.columns or "ID" not in df.columns:
        return f"{prefix}{1:0{width}d}"

    ids = df.loc[df["Role"].astype(str).str.strip() == role, "ID"].astype(str).str.strip()
    ids = ids[ids.str.upper().str.startswith(prefix)]

    if ids.empty:
        return f"{prefix}{1:0{width}d}"

    width = max(width, int(ids.str.len().max()) - len(prefix))
    nums = ids.str[len(prefix):]
    nums = nums[nums.str.isdigit()].astype(int)
    next_num = int(nums.max()) + 1 if not nums.empty else 1
    return f"{prefix}{next_num:0{width}d}"


def enroll_facenet_prototype(person_id, photos_dir):
    """Embeds this person's captured photos with the existing, frozen FaceNet-Style Embedding + ArcFace
    embedding_model and appends/updates their prototype vector -- the step
    that used to be a separate Colab cell, now done right here after capture."""
    facenet_artifacts = load_facenet_algorithm()
    if facenet_artifacts is None:
        st.warning(
            "FaceNet-Style Embedding + ArcFace isn't configured yet (missing files in models2/), so no "
            "prototype was built for this person. Photos and details are "
            "still saved -- add the FaceNet-Style Embedding + ArcFace files and re-enroll later."
        )
        return

    embedding_model = facenet_artifacts["embedding_model"]
    prototype_ids = facenet_artifacts["prototype_ids"]
    prototype_vecs = facenet_artifacts["prototype_vecs"]

    embeddings = []
    for filepath in sorted(Path(photos_dir).iterdir()):
        image = cv2.imread(str(filepath))
        if image is None:
            continue
        face_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        embeddings.append(facenet_embed(face_rgb, embedding_model))

    if not embeddings:
        st.warning(f"No usable photos found for {person_id} -- prototype not built.")
        return

    new_proto = np.mean(embeddings, axis=0)
    new_proto = new_proto / np.linalg.norm(new_proto)

    if person_id in prototype_ids:
        idx = int(np.where(prototype_ids == person_id)[0][0])
        prototype_vecs[idx] = new_proto
    else:
        prototype_ids = np.append(prototype_ids, person_id)
        prototype_vecs = np.vstack([prototype_vecs, new_proto])

    FACENET_DIR.mkdir(parents=True, exist_ok=True)
    np.save(FACENET_PROTOTYPE_IDS_PATH, prototype_ids)
    np.save(FACENET_PROTOTYPE_VECS_PATH, prototype_vecs)

    # so the very next recognition run (no page reload needed) picks this up
    load_facenet_algorithm.clear()


def toggle_registration_capture():
    st.session_state.reg_capturing = not st.session_state.reg_capturing


def start_new_registration():
    """Resets the registration form so another member can be enrolled.
    Clears the widget-bound field keys too, so the form comes back blank
    instead of pre-filled with the previous member's details."""
    st.session_state.reg_finished_id = None
    st.session_state.reg_field_values = {}
    for key in list(st.session_state.keys()):
        if key.startswith("reg_field_"):
            del st.session_state[key]


def render_registration_page():
    st.title("🆕 Register New Member")
    st.caption(
        f"Pick a role, and fill in the details, then capture {CAPTURE_TARGET} face "
        "photos."
    )

    for key, default in [
        ("reg_capturing", False),
        ("reg_captured_count", 0),
        ("reg_photo_paths", []),
        ("reg_field_values", {}),
        ("reg_finished_id", None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    existing_columns = get_data_csv_columns()
    if "ID" not in existing_columns:
        existing_columns = ["ID"] + existing_columns
    # ID is auto-generated from Role after submit -- never a manual field
    entry_columns = [col for col in existing_columns if col != "ID"]
    if "Role" not in entry_columns:
        entry_columns = ["Role"] + entry_columns

    if st.session_state.reg_finished_id:
        st.markdown(
            f"""
            <div class="kpi-card" style="--accent: #55A868; margin-bottom: 1rem;">
                <div class="kpi-label">Enrollment complete</div>
                <div class="kpi-value">🎉 {st.session_state.reg_finished_id}</div>
                <div class="kpi-sub">Ready for entrance recognition</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.button(
            "Register Another Member",
            on_click=start_new_registration,
            type="primary",
            use_container_width=True,
        )
        return

    if not st.session_state.reg_capturing:
        with st.form("register_form"):
            st.subheader("Member details")
            field_values = {}
            for col in entry_columns:
                if col == "Role":
                    field_values[col] = st.selectbox(
                        "Role", ["Student", "Staff"],
                        index=None, placeholder="Select Role", key="reg_field_Role",
                    )
                elif col in ("Gender", "Faculty"):
                    field_values[col] = st.selectbox(
                        col, get_dropdown_options(col),
                        index=None, placeholder=f"Select {col}", key=f"reg_field_{col}",
                    )
                else:
                    field_values[col] = st.text_input(col, key=f"reg_field_{col}")

            st.divider()
            st.subheader("Privacy & consent")
            st.markdown(
                "Your facial images are sensitive biometric data. By enrolling, "
                f"the system will capture and store {CAPTURE_TARGET} photos of your face, "
                "which will be used **only** to train and run this entrance-recognition "
                "system for university members. Photos and derived recognition data are "
                "kept only as long as you remain enrolled, and are not shared outside this "
                "system."
            )
            consent_given = st.checkbox(
                "I have read the above and give my informed consent to the collection "
                "and processing of my facial images for entrance-recognition purposes.",
                key="reg_consent_checkbox",
            )
            submitted = st.form_submit_button("Start capture", type="primary")

        if submitted:
            if not consent_given:
                st.error(
                    "You must check the consent box before any facial images can be "
                    "captured."
                )
                return

            missing = [
                col for col in entry_columns
                if col in ("Role", "Gender", "Faculty") and not field_values.get(col)
            ]
            if missing:
                st.error(f"Please select a value for: {', '.join(missing)}.")
                return

            if "Name" in entry_columns:
                name_value = (field_values.get("Name") or "").strip()
                field_values["Name"] = name_value
                if not name_value:
                    st.error("Please fill in the name field.")
                    return
                if not NAME_PATTERN.match(name_value):
                    st.error(
                        "Name can only contain letters, spaces, hyphens, apostrophes, "
                        "and periods (e.g. \"Mary-Jane O'Neil\")."
                    )
                    return

            person_id = generate_new_id(field_values["Role"])
            field_values["ID"] = person_id

            # Consent is checked and logged before setting reg_capturing = True,
            # i.e. before the webcam loop below ever writes a face photo to disk.
            record_consent(person_id)

            st.session_state.reg_field_values = field_values
            st.session_state.reg_captured_count = 0
            st.session_state.reg_photo_paths = []
            st.session_state.reg_capturing = True
            st.rerun()
        return

    # --- capturing in progress ---
    person_id = st.session_state.reg_field_values["ID"].strip()
    st.info(
        f"Capturing photos for **{person_id}**. Look at the camera and slowly turn "
        "your head for variety. Make sure no mask is on."
    )

    st.button(
        "⏹️ Cancel registration",
        on_click=toggle_registration_capture,
        use_container_width=True,
    )

    cap_frame_slot = st.empty()
    cap_progress_slot = st.empty()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not access the webcam. Is it being used by another app?")
        st.session_state.reg_capturing = False
    else:
        last_capture_time = 0.0
        try:
            while st.session_state.reg_capturing and st.session_state.reg_captured_count < CAPTURE_TARGET:
                ret, frame = cap.read()
                if not ret:
                    st.error("Lost connection to the webcam.")
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detect_faces_yunet(frame)

                if len(faces) > 0:
                    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                    face_color = frame[y:y + h, x:x + w]
                    masked = mask_artifacts is not None and is_wearing_mask(face_color, mask_artifacts)

                    if masked:
                        color = (0, 165, 255)  # orange
                        label = "Please remove mask"
                    else:
                        now = time.time()
                        if now - last_capture_time >= CAPTURE_INTERVAL_SECONDS:
                            idx = st.session_state.reg_captured_count
                            path = save_face_photo(person_id, face_color, idx)
                            st.session_state.reg_photo_paths.append(path)
                            st.session_state.reg_captured_count += 1
                            last_capture_time = now
                        color = (0, 200, 0)  # green
                        label = f"Captured {st.session_state.reg_captured_count}/{CAPTURE_TARGET}"

                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label, (x, max(y - 10, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cap_frame_slot.image(frame_rgb, channels="RGB", use_container_width=True)

                with cap_progress_slot.container():
                    pct = min(st.session_state.reg_captured_count / CAPTURE_TARGET, 1.0)
                    st.progress(pct)
                    st.markdown(
                        f"<div style='text-align:center; font-size:1.4rem; font-weight:700; color:#4C72B0;'>"
                        f"{st.session_state.reg_captured_count} / {CAPTURE_TARGET} "
                        f"<span style='font-size:0.9rem; color:#888; font-weight:600;'>({pct*100:.0f}%)</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
        finally:
            cap.release()

    if st.session_state.reg_captured_count >= CAPTURE_TARGET:
        st.session_state.reg_capturing = False

        append_member_row(st.session_state.reg_field_values)
        mark_as_newly_registered(person_id)
        load_member_info.clear()

        photos_dir = STORED_FACES_DIR / person_id
        st.success(f"Saved {CAPTURE_TARGET} photos for **{person_id}** and added them to {INFO_PATH.name}.")

        with st.spinner("Building FaceNet-Style Embedding + ArcFace prototype..."):
            enroll_facenet_prototype(person_id, photos_dir)

        st.session_state.reg_photo_paths = []
        st.session_state.reg_captured_count = 0

        # Registration is done -- show the "finished" screen (with a button
        # to register another member) instead of the capture view.
        st.session_state.reg_finished_id = person_id
        st.balloons()
        st.rerun()


# ========================================================================
# Member removal -- undoes registration across every place a person's data
# lives, since none of it is automatically synced with data.csv:
#   1. data.csv               -- their info row
#   2. stored-faces/<ID>/     -- their captured photos
#   3. FaceNet prototype      -- their stored embedding vector
#
# NOTE: CNN and SVM+HOG bake identities into their trained weights at
# training time (label_classes.npy / classes_hog.npy are just fixed output
# classes). Removing someone here can't un-teach those two models -- only
# retraining their notebooks can. That's exactly why entrance recognition
# gates every match against is_active_member(): even if CNN/SVM+HOG still
# "recognizes" a removed person's face, they're denied because they no
# longer have a data.csv row.
# ========================================================================
def remove_member_row(person_id):
    """Deletes this person's row from data.csv."""
    if not INFO_PATH.exists():
        return
    df = pd.read_csv(INFO_PATH)
    df.columns = df.columns.str.strip()
    if "ID" not in df.columns:
        return
    df = df[df["ID"].astype(str).str.strip() != str(person_id).strip()]
    df.to_csv(INFO_PATH, index=False)


def delete_stored_faces(person_id):
    """Deletes stored-faces/<ID>/ and everything in it, if present."""
    person_dir = STORED_FACES_DIR / str(person_id)
    if person_dir.exists():
        shutil.rmtree(person_dir)


def remove_facenet_prototype(person_id):
    """Removes this person's embedding vector from the FaceNet-Style
    Embedding + ArcFace prototype store, so similarity matching can no
    longer find them. Returns True if a prototype was actually removed."""
    missing = [p.name for p in [FACENET_PROTOTYPE_IDS_PATH, FACENET_PROTOTYPE_VECS_PATH] if not p.exists()]
    if missing:
        return False

    prototype_ids = np.load(FACENET_PROTOTYPE_IDS_PATH, allow_pickle=True).astype(str)
    prototype_vecs = np.load(FACENET_PROTOTYPE_VECS_PATH)

    key = str(person_id).strip()
    matches = np.where(prototype_ids == key)[0]
    if len(matches) == 0:
        return False

    keep = np.ones(len(prototype_ids), dtype=bool)
    keep[matches] = False
    prototype_ids = prototype_ids[keep]
    prototype_vecs = prototype_vecs[keep]

    np.save(FACENET_PROTOTYPE_IDS_PATH, prototype_ids)
    np.save(FACENET_PROTOTYPE_VECS_PATH, prototype_vecs)

    # so the very next recognition run (no page reload needed) picks this up
    load_facenet_algorithm.clear()
    return True


def render_manage_members_page():
    st.title("🗑️ Manage Members")
    st.caption("Only members registered through this application can be deleted here.")

    # Only IDs explicitly created through Register New Member are removable.
    newly_registered_ids = get_newly_registered_member_ids()
    options = sorted(
        pid for pid in member_info.keys()
        if str(pid).strip() in newly_registered_ids
    )

    if not options:
        st.info(
            "No newly registered members are available to delete. "
            "Existing members loaded from data.csv are protected and cannot be removed here."
        )
        return

    selected_id = st.selectbox(
        "Select a newly registered member to remove",
        options,
        index=None,
        placeholder="Choose a newly registered member ID",
        format_func=lambda pid: (
            f"{pid} — {display_name(pid)}" if get_info(pid) else pid
        ),
    )

    if selected_id:
        info = get_info(selected_id)
        with st.container():
            st.markdown(f"**ID:** {selected_id}")
            for column, value in info.items():
                value = str(value).strip()
                if value and value.lower() != "nan":
                    st.markdown(f"**{column}:** {value}")

        person_dir = STORED_FACES_DIR / str(selected_id)
        photo_count = len(list(person_dir.glob("*"))) if person_dir.exists() else 0
        st.caption(f"{photo_count} stored photo(s) will be deleted from disk.")

        confirm = st.checkbox(
            f"I understand this permanently deletes {selected_id}'s registered data and photos.",
            key="manage_members_confirm",
        )

        if st.button(
            "Delete newly registered member",
            type="primary",
            disabled=not confirm,
            use_container_width=True,
        ):
            # Extra safety: never delete a pre-existing data.csv member even if
            # someone manipulates the widget/session state.
            if str(selected_id).strip() not in get_newly_registered_member_ids():
                st.error("This member is protected and cannot be deleted from this page.")
                return

            remove_member_row(selected_id)
            delete_stored_faces(selected_id)
            proto_removed = remove_facenet_prototype(selected_id)
            unmark_newly_registered(selected_id)

            load_member_info.clear()

            st.success(
                f"Deleted newly registered member **{selected_id}** and their stored photos."
            )
            if proto_removed:
                st.success("Also removed their FaceNet-Style Embedding + ArcFace prototype.")
            time.sleep(0.8)
            st.rerun()


# ------------------------------------------------------------------
# Group members — shown ONLY on the Background page
# ------------------------------------------------------------------
GROUP_MEMBERS = [
    {"Name": "Andrea Ng Wing Kay", "ID": "2508258", "Model": "SVM + HOG"},
    {"Name": "Liew Yong Xin", "ID": "2508327", "Model": "FaceNet-Style Embedding + ArcFace"},
    {"Name": "Yap Kah Ying", "ID": "2508390", "Model": "CNN"},
]


def render_group_members():
    st.markdown("## 👥 Group Members")
    st.dataframe(
        pd.DataFrame(GROUP_MEMBERS),
        use_container_width=True,
        hide_index=True,
    )
    st.divider()


def render_background_page():
    # ============================================================
    # Group Members (Background page only)
    # ============================================================
    render_group_members()

    # ============================================================
    # Project Overview
    # ============================================================
    st.markdown("## 📌 Project Overview")

    info_col1, info_col2, info_col3, info_col4 = st.columns(4)

    with info_col1:
        kpi_card("Recognition Algorithms", "3", "SVM + HOG · FaceNet · CNN", accent="#4C72B0")

    with info_col2:
        kpi_card("Face Detector", "YuNet", "ONNX, real-time", accent="#6A4C93")

    with info_col3:
        kpi_card("Mask Detector", "MobileNetV2", "gates recognition", accent="#B0B0B0")

    with info_col4:
        kpi_card(
            "Registered Members",
            str(len(member_info)) if member_info else "0",
            "in data.csv",
            accent="#55A868",
        )

    st.divider()

    # ============================================================
    # Background / Problem Statement
    # ============================================================
    left_col, right_col = st.columns([1.2, 0.8])

    with left_col:
        st.markdown("### 🎯 Background")

        st.write(
            "University libraries are a core pillar of academic life for students "
            "and faculty, and they are modernizing rapidly to keep pace with "
            "smart-campus technologies. Traditional access systems that rely "
            "on physical or RFID cards can be inefficient and may introduce "
            "privacy and security vulnerabilities."
        )

        st.write(
            "Integrating AI-driven computer vision provides an alternative "
            "approach by using facial recognition for automated library "
            "access. This project develops a real-time facial recognition "
            "system designed to identify enrolled users and support a more "
            "efficient library entrance process."
        )

        st.markdown("### ⚠️ Problem Statement")

        st.write(
            "Card-based access systems can be vulnerable to credential "
            "sharing, loss, cloning and unauthorized use. They also provide "
            "limited automatic information about the identity and role of "
            "the person entering the facility."
        )

        st.write(
            "The project therefore investigates whether a facial recognition "
            "system can reliably identify registered users under practical "
            "entrance conditions."
        )

    with right_col:
        st.markdown("### 💡 Project Objectives")

        st.markdown(
            "- **Develop a Real-Time Facial Recognition System** — identify "
            "registered users during library entry under changing lighting, "
            "face angles and other practical conditions.\n"
            "- **Implement Role-Based Classification** — use stored member "
            "information to distinguish betweenuser groups such as students and staff.\n"
            "- **Compare Recognition Algorithms** — evaluate CNN, FaceNet-Style Embedding + ArcFace and "
            "SVM + HOG approaches using the same detection pipeline."
        )

        st.markdown("### 📈 Significance / Contribution")

        st.write(
            "The system aims to improve the efficiency of library entrance "
            "operations by reducing reliance on physical cards and providing "
            "automated identity verification. It also demonstrates how "
            "computer vision and machine learning can be integrated into a "
            "smart-campus application."
        )

    st.divider()

    # ============================================================
    # Innovation
    # ============================================================
    st.markdown("### 🚀 Innovation")

    st.markdown(
       "- **Self-Service Biometric Enrollment** — real-time member registration "
       "that auto-assigns role-based IDs and provisions vector databases "
       "instantly with no manual notebook steps.\n"
       "- **Full Lifecycle Record Management** — complete self-service deletion "
       "support, allowing immediate removal of member profiles and embeddings "
       "from the database.\n"
       "- **Multi-Model Hybrid Architecture** — bridges classical ML (SVM + HOG) "
       "and deep learning (FaceNet + ArcFace) to dynamically balance CPU efficiency "
       "against high-accuracy recognition."
    )

    st.divider()

    # ============================================================
    # Methodology Overview
    # ============================================================
    st.markdown("### 🔬 Methodology Overview")

    st.write(
        "The system uses OpenCV YuNet for face detection, followed by face "
        "cropping and preprocessing. A pretrained MobileNetV2-based mask "
        "detector can gate recognition before the selected recognition "
        "algorithm is executed."
    )

    method_col1, method_col2, method_col3 = st.columns(3)

    with method_col1:
        st.markdown("**🧠 CNN**")
        st.write(
            "A multi-class classifier that predicts the enrolled identity "
            "from the processed face image."
        )

    with method_col2:
        st.markdown("**🔗 FaceNet-Style Embedding + ArcFace**")
        st.write(
            "Generates a 128-dimensional normalized embedding and compares "
            "it with per-identity prototype vectors using cosine similarity."
        )

    with method_col3:
        st.markdown("**📐 SVM + HOG**")
        st.write(
            "Extracts HOG features, applies the trained PCA/scaling pipeline, "
            "and classifies the face using an RBF-kernel SVM."
        )

# ----------------------------------------------------------------------
# Main page controls (replaces the sidebar)
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <h1>📚 Library Face Recognition System</h1>
        <p>Real-time entrance recognition · mask gating · consent-first enrollment</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Mode and Recognition Algorithm are displayed side-by-side in one row.
# Mode receives more width because it contains five choices, while the column gap is kept compact.
mode_col, algorithm_col = st.columns([2.2, 1.2], gap="small", vertical_alignment="top")

with mode_col:
    app_mode = st.radio(
        "Mode",
        ["Background", "Entrance recognition", "Register new member", "Manage members", "Models Evaluation"],
        horizontal=True,
        key="app_mode"
    )

with algorithm_col:
    algo_name = st.radio(
        "Recognition algorithm",
        list(ALGORITHMS.keys()),
        horizontal=True,
        key="algo_name"
    )

algo = ALGORITHMS[algo_name]
artifacts = algo["loader"]()

# Status / setup information
if artifacts is None:
    st.warning(
        f"'{algo_name}' isn't configured yet.\n\n{algo['setup_hint']}"
    )

if mask_artifacts is None:
    st.warning(
        "Mask detector couldn't be loaded (no internet access on first run?). "
        "Recognition will proceed without checking for a mask."
    )

# Recognition thresholds and other settings are kept together on the main page.
with st.expander("⚙️ Recognition Thresholds & System Information"):
    threshold_col1, threshold_col2, threshold_col3 = st.columns(3)

    with threshold_col1:
        kpi_card("CNN Threshold", f"{CNN_CONF_THRESHOLD:.2f}", accent="#4C72B0")

    with threshold_col2:
        kpi_card("FaceNet Confidence", f"{FACENET_CONFIDENCE_THRESHOLD:.2f}",
                  f"margin: {FACENET_MARGIN_THRESHOLD:.2f}", accent="#6A4C93")

    with threshold_col3:
        kpi_card("SVM + HOG Threshold", f"{SVM_CONF_THRESHOLD:.2f}", accent="#B0B0B0")

    st.divider()

    log_col, member_col = st.columns(2)

    with log_col:
        st.subheader("📥 Entrance Log")
        if LOG_PATH.exists():
            with open(LOG_PATH, "rb") as f:
                st.download_button(
                    "Download full log (CSV)",
                    f,
                    file_name="entrance_log.csv",
                    use_container_width=True
                )
        else:
            st.caption("No entries logged yet.")

    with member_col:
        st.subheader("👥 Registered Members")
        if not member_info:
            st.caption(
                f"Tip: add a `{INFO_PATH.name}` (must include an ID column, "
                "plus any other columns like Name/Role) next to this app."
            )
        else:
            st.write(f"Loaded **{len(member_info)}** member record(s)")
            with st.expander("View member IDs"):
                st.caption(
                    "If a recognized ID isn't in this list, its details "
                    "will not be shown."
                )
                st.write(sorted(member_info.keys()))

st.divider()

# ----------------------------------------------------------------------
# Registration / models evaluation pages
# ----------------------------------------------------------------------
if app_mode == "Register new member":
    render_registration_page()
    st.stop()

if app_mode == "Manage members":
    render_manage_members_page()
    st.stop()

if app_mode == "Models Evaluation":
    render_model_comparison_page()
    st.stop()

if app_mode == "Background":
    render_background_page()
    st.stop()

# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------
if "camera_running" not in st.session_state:
    st.session_state.camera_running = False


def toggle_camera():
    st.session_state.camera_running = not st.session_state.camera_running


# ----------------------------------------------------------------------
# Main layout
# ----------------------------------------------------------------------
st.caption(f"Using **{algo_name}**. Tap the camera area below to start scanning.")

if artifacts is None:
    st.warning(
        f"**{algo_name}** is selected but not configured yet.\n\n{algo['setup_hint']}\n\n"
        "Pick **CNN** or **FaceNet-Style Embedding + ArcFace** in the sidebar to use a working model."
    )

video_col, status_col = st.columns([2, 1])

with video_col:
    st.button(
        "⏹️ Tap to stop camera" if st.session_state.camera_running else "📷 Tap to start camera",
        on_click=toggle_camera,
        use_container_width=True,
        disabled=(artifacts is None),
        type="primary" if not st.session_state.camera_running else "secondary",
    )
    frame_slot = st.empty()

with status_col:
    alert_slot = st.empty()
    progress_slot = st.empty()
    confidence_slot = st.empty()
    info_slot = st.empty()

log_slot = st.empty()

_log_render_counter = {"n": 0}


def render_log():
    _log_render_counter["n"] += 1
    call_id = _log_render_counter["n"]

    header, rows = read_log_tail(30)
    with log_slot.container():
        st.subheader("Recent entries")
        if rows:
            records = [dict(zip(header, r)) for r in rows]
            log_df = pd.DataFrame(records)

            filter_col1, filter_col2 = st.columns([1, 1])
            with filter_col1:
                algo_options = ["All"] + list(ALGORITHMS.keys())
                algo_filter = st.selectbox(
                    "Filter by algorithm", algo_options, key=f"log_algo_filter_{call_id}"
                )
            with filter_col2:
                role_options = ["All", "Student", "Staff"]
                role_filter = st.selectbox(
                    "Filter by role", role_options, key=f"log_role_filter_{call_id}"
                )

            filtered = log_df
            if algo_filter != "All":
                filtered = filtered[filtered["Algorithm"] == algo_filter]
            if role_filter != "All" and "Role" in filtered.columns:
                filtered = filtered[filtered["Role"] == role_filter]

            st.dataframe(filtered.head(15), use_container_width=True, hide_index=True)
        else:
            st.info("No entries yet.")

def render_status(matched, person_id, score, held_for=0.0):

    # Toast on a fresh access grant (only fires once per confirmation, not every frame)
    if matched and person_id and render_status.__dict__.get("_last_granted") != person_id:
        st.toast(f"✅ Access granted: {display_name(person_id)}", icon="✅")
        render_status.__dict__["_last_granted"] = person_id
    elif not matched:
        render_status.__dict__["_last_granted"] = None

    # Alert
    if matched:
        alert_slot.success("✅ ACCESS GRANTED")
        progress_slot.empty()

    elif held_for > 0:
        progress = min(held_for / CONFIRM_HOLD_SECONDS, 1.0)

        alert_slot.warning("🟡 VERIFYING...")

        with progress_slot.container():
            st.progress(progress)
            st.metric("Verification Progress", f"{progress*100:.0f}%")

    else:
        alert_slot.error("🚫 ACCESS DENIED")
        progress_slot.empty()

    # Confidence (always replace) -- plain label + big percentage, same
    # style as the "Verification Progress" metric above it
    with confidence_slot.container():
        st.metric("Confidence", f"{score*100:.1f}%")

    # Information (always replace)
    info_slot.empty()
    with info_slot.container():

        st.markdown("### Information")

        if matched and person_id:
            info = get_info(person_id)

            st.write(f"**ID:** {person_id}")

            if info:
                for column, value in info.items():
                    if value is None:
                        continue
                    value = str(value).strip()
                    if value == "" or value.lower() == "nan":
                        continue
                    st.write(f"**{column}:** {value}")

        elif held_for > 0:
            st.caption("Hold your face steady...")

        else:
            st.caption("No match.")


render_log()

if st.session_state.camera_running and artifacts is not None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not access the webcam. Is it being used by another app?")
        st.session_state.camera_running = False
    else:
        last_logged = {}       # person_id -> last logged unix timestamp (cooldown)
        hold_id = None         # identity currently being held/verified
        hold_start = None      # when that hold began
        prediction_history = deque(maxlen=STABLE_WINDOW)
        confirmed_id = None
        confirmed_score = 0.0
        last_valid_face_time = time.time()
        last_confirmed_time = 0.0
        try:
            while st.session_state.camera_running:
                ret, frame = cap.read()
                if not ret:
                    st.error("Lost connection to the webcam.")
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detect_faces_yunet(frame)

                if len(faces) > 0:
                    # Only the largest face is tracked — one person at a time.
                    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

                    face_color = frame[y:y + h, x:x + w]
                    masked = mask_artifacts is not None and is_wearing_mask(face_color, mask_artifacts)

                    if masked:
                        # Mask detected — don't attempt recognition, ask the
                        # person to remove it first.
                        hold_id, hold_start = None, None
                        color = (0, 165, 255)  # orange
                        label = "Please remove mask"

                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(frame, label, (x, max(y - 10, 15)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                        alert_slot.warning("😷 Mask detected — please remove your mask to proceed")
                        progress_slot.empty()
                        confidence_slot.metric("Confidence", "—")
                        with info_slot.container():
                            st.markdown("### Information")
                            st.caption("Waiting for mask to be removed...")

                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_slot.image(frame_rgb, channels="RGB", use_container_width=True)
                        continue

                    face_gray = gray[y:y + h, x:x + w]
                    face_color_rgb = cv2.cvtColor(face_color, cv2.COLOR_BGR2RGB)

                    try:
                        result = algo["recognize"](face_gray, face_color_rgb, artifacts)
                    except NotImplementedError:
                        result = {"matched": False, "person_id": None, "score": 0.0}

                    # Membership gate: CNN/SVM+HOG bake identities into their
                    # trained weights at training time, and FaceNet prototypes
                    # persist in prototype_ids.npy/prototype_vecs.npy -- none
                    # of that is synced with data.csv. So a person removed
                    # from data.csv would otherwise still be "recognized" by
                    # the underlying model. This check denies access for
                    # anyone no longer an active member, regardless of what
                    # the algorithm itself returns.
                    if result["matched"] and not is_active_member(result["person_id"]):
                        result = {"matched": False, "person_id": None, "score": result["score"]}

                    raw_id = result["person_id"] if result["matched"] else None
                    raw_score = float(result["score"])
                    now = time.time()
                    last_valid_face_time = now

                    prediction_history.append(raw_id)
                    known_predictions = [pid for pid in prediction_history if pid is not None]

                    stable_id = None
                    if known_predictions:
                        counts = {}
                        for pid in known_predictions:
                            counts[pid] = counts.get(pid, 0) + 1
                        candidate_id, candidate_votes = max(counts.items(), key=lambda item: item[1])
                        if candidate_votes >= STABLE_MIN_VOTES:
                            stable_id = candidate_id

                    if stable_id is not None:
                        if stable_id != hold_id:
                            hold_id = stable_id
                            hold_start = now
                        held_for = now - hold_start

                        if stable_id == confirmed_id:
                            confirmed_score = max(confirmed_score, raw_score)
                            last_confirmed_time = now
                        elif held_for >= CONFIRM_HOLD_SECONDS:
                            confirmed_id = stable_id
                            confirmed_score = raw_score
                            last_confirmed_time = now
                    else:
                        hold_id, hold_start = None, None
                        held_for = 0.0

                    recently_confirmed = (
                        confirmed_id is not None and
                        (now - last_confirmed_time) <= IDENTITY_HOLD_SECONDS
                    )

                    if recently_confirmed:
                        confirmed = True
                        shown_id = confirmed_id
                        shown_score = confirmed_score if stable_id is None else max(confirmed_score, raw_score)
                        color = (0, 200, 0)
                        label = display_name(shown_id)

                        if shown_id not in last_logged or (now - last_logged[shown_id]) > COOLDOWN_MINUTES * 60:
                            last_logged[shown_id] = now
                            append_log(algo_name, shown_id, shown_score)
                            render_log()
                    elif stable_id is not None:
                        confirmed = False
                        shown_id = None
                        shown_score = raw_score
                        color = (0, 165, 255)
                        label = "Verifying..."
                    else:
                        confirmed = False
                        shown_id = None
                        shown_score = raw_score
                        color = (0, 0, 255)
                        label = "Unknown"

                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label, (x, max(y - 10, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                    render_status(confirmed, shown_id, shown_score, held_for)
                else:
                    now = time.time()
                    if now - last_valid_face_time > NO_FACE_RESET_SECONDS:
                        prediction_history.clear()
                        hold_id, hold_start = None, None
                        confirmed_id = None
                        confirmed_score = 0.0
                        last_confirmed_time = 0.0

                    if confirmed_id is not None and (now - last_confirmed_time) <= IDENTITY_HOLD_SECONDS:
                        render_status(
                            matched=True,
                            person_id=confirmed_id,
                            score=confirmed_score,
                            held_for=now - hold_start if hold_start else 0.0,
                        )
                    else:
                        render_status(matched=False, person_id=None, score=0.0, held_for=0.0)


                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_slot.image(frame_rgb, channels="RGB", use_container_width=True)
        finally:
            cap.release()
else:
    frame_slot.info("Camera is off — tap the button above to start.")

st.divider()