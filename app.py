import os
import io
import uuid
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pymongo import MongoClient
import numpy as np
import cv2
import mediapipe as mp
import joblib
import tensorflow as tf
from gtts import gTTS
import random
import hashlib
from threading import Thread
# -------------------------------
# Load environment
# -------------------------------
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "yogaDB")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "poses")
MODEL_DIR = os.getenv("MODEL_DIR", "model")   # path relative to backend/
AUDIO_DIR = os.getenv("AUDIO_DIR", "feedback_audio")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Ensure audio directory exists
Path(AUDIO_DIR).mkdir(parents=True, exist_ok=True)

# -------------------------------
# Init Flask / CORS / MongoDB
# -------------------------------
app = Flask(__name__, static_folder=None)
CORS(app)

if not MONGO_URI:
    raise RuntimeError("MONGO_URI not set in .env")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
poses_collection = db[COLLECTION_NAME]
users_collection = db.get_collection("users")  # users collection

# -------------------------------
# Load model, scaler, encoder
# -------------------------------
# Expect these files in backend/model/
MODEL_PATH = os.path.join(MODEL_DIR, "yoga_pose_model.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}")
if not os.path.exists(ENCODER_PATH):
    raise FileNotFoundError(f"Label encoder not found at {ENCODER_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(ENCODER_PATH)

# -------------------------------
# Mediapipe setup
# -------------------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose_detector = mp_pose.Pose(static_image_mode=True, model_complexity=2,
                             min_detection_confidence=0.5)

# -------------------------------
# Helpers
# -------------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def calc_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ab, bc = a - b, c - b
    denom = (np.linalg.norm(ab) * np.linalg.norm(bc))
    if denom == 0:
        return 0.0
    cosang = np.dot(ab, bc) / denom
    return float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))

def extract_feature_vector_from_image_bgr(image_bgr):
    """Return feature vector of shape (1, N) or None if no landmarks"""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(image_rgb)
    if not results.pose_landmarks:
        return None, None  # no landmarks
    lm = results.pose_landmarks.landmark

    # 33 landmarks * (x,y,z,visibility) = 132
    keypoints = np.array([[p.x, p.y, p.z, p.visibility] for p in lm]).flatten()

    # same two elbow angles we used in training as example
    left_elbow = calc_angle(
        [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
        [lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y],
        [lm[mp_pose.PoseLandmark.LEFT_WRIST.value].x, lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    )
    right_elbow = calc_angle(
        [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y],
        [lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y],
        [lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    )

    feature_vector = np.concatenate([keypoints, np.array([left_elbow, right_elbow])])
    return feature_vector.reshape(1, -1), results.pose_landmarks

def compute_symmetry_metric(landmarks):
    # simple left-right x diffs average (lower -> more symmetric)
    left_shx = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x
    right_shx = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x
    left_hipx = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x
    right_hipx = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x
    left_kneex = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x
    right_kneex = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x
    return float(np.mean([abs(left_shx - right_shx), abs(left_hipx - right_hipx), abs(left_kneex - right_kneex)]))

def generate_feedback_text(landmarks, predicted_pose, selected_pose, confidence):
    """Enhanced feedback generator with pose-specific and randomized messages."""
    hints = []

    # --- Heuristic posture analysis ---
    left_sh = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
               landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    right_sh = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

    sh_mid_x = (left_sh[0] + right_sh[0]) / 2.0
    hip_mid_x = (left_hip[0] + right_hip[0]) / 2.0
    torso_tilt = abs(sh_mid_x - hip_mid_x)

    left_elbow_a = calc_angle(
        [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
        [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y],
        [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    )
    right_elbow_a = calc_angle(
        [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y],
        [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y],
        [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    )

    left_knee_a = calc_angle(
        [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y],
        [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y],
        [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    )

    if torso_tilt > 0.08:
        hints.append("Keep your torso upright and avoid leaning forward or backward.")
    if left_elbow_a < 40 or right_elbow_a < 40:
        hints.append("Try straightening your arms.")
    if left_knee_a < 160:
        hints.append("Bend your front knee slightly more if required by this pose.")

    # --- Pose-specific guidance ---
    pose_hints = {
        "tadasana": [
            "Feet together, spine tall — imagine growing upward like a mountain.",
            "Relax your shoulders and distribute your weight evenly on both feet."
        ],
        "vrikshasana": [
            "Balance on one leg with calm focus — keep your gaze steady.",
            "Press your foot gently into your inner thigh and open your chest."
        ],
        "phalasana": [
            "Keep your body in a straight line — don’t let your hips sag.",
            "Engage your core and push evenly through both palms."
        ],
        "paschimottanasana": [
            "Lengthen from your hips — avoid rounding your back in the forward fold.",
            "Breathe deeply and keep your knees soft but steady."
        ],
        "bhujangasana": [
            "Lift your chest gently while keeping your elbows close to your sides.",
            "Relax your shoulders away from your ears and look slightly upward."
        ],
        "trikonasana": [
            "Open your chest and extend your arms straight in a line.",
            "Keep both legs firm — don’t collapse your torso forward."
        ],
        "adho mukha svanasana": [
            "Push through your palms and lift your hips high.",
            "Keep your back flat and heels pressing gently toward the floor."
        ]
    }

    # --- Feedback selection logic ---
    feedback_options = []

    if predicted_pose != selected_pose:
        feedback_options = [
            f"Your pose doesn’t quite match {selected_pose.capitalize()}. Focus on alignment and balance.",
            f"Try adjusting your form — it differs slightly from {selected_pose.capitalize()}.",
            f"Not quite there. Compare your angles with {selected_pose.capitalize()} reference.",
            f"Pose mismatch! Realign yourself into {selected_pose.capitalize()}."
        ] + pose_hints.get(selected_pose, [])

    elif confidence < 0.8:
        feedback_options = [
            f"Nice effort on {selected_pose.capitalize()}! A few tweaks will improve your alignment.",
            f"Almost there — focus on symmetry and breathing in {selected_pose.capitalize()}.",
            f"Good work! Try holding {selected_pose.capitalize()} a bit steadier.",
            f"Keep practicing {selected_pose.capitalize()} — you’re close to perfect form!"
        ] + pose_hints.get(selected_pose, [])

    else:
        feedback_options = [
            f"Perfect {selected_pose.capitalize()}! Excellent control and balance.",
            f"Beautiful {selected_pose.capitalize()} — your posture looks spot on!",
            f"Fantastic! You’ve mastered {selected_pose.capitalize()} with great stability.",
            f"Excellent job! That’s a textbook {selected_pose.capitalize()} pose!"
        ]

    # Add hints if available
    feedback = random.choice(feedback_options)
    if hints:
        feedback += " Hints: " + " ".join(hints)

    return feedback, hints

def evaluate_joint_correctness(landmarks):
    """Return a list of {x, y, correct} for overlay."""
    correctness = []
    for i, lm in enumerate(landmarks):
        # Dummy threshold: mark joints red if visibility < 0.6
        correctness.append({
            "x": lm.x,
            "y": lm.y,
            "correct": bool(lm.visibility > 0.6)
        })
    return correctness

def compute_score(confidence, symmetry_metric):
    """
    Compute a more realistic score (0–100) from confidence & symmetry.
    Gives higher weight to model confidence, mild penalty for asymmetry.
    """
    # Cap symmetry within 0–0.3 (higher = more asymmetrical)
    sym_norm = min(symmetry_metric / 0.3, 1.0)
    # Convert to penalty
    sym_penalty = 1.0 - (0.5 * sym_norm)
    
    # Confidence weight (60%) + symmetry (40%)
    combined = (confidence * 0.6) + (sym_penalty * 0.4)
    
    score = int(round(combined * 100))
    return max(1, min(100, score))  # never return 0 to avoid confusion


def make_audio_filename_for_text(text: str) -> str:
    # deterministic name so same text reuses same file
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return f"{h}.mp3"

def make_audio_for_text_sync(text):
    """Create an mp3 file synchronously and return its filesystem path and relative url."""
    safe_name = make_audio_filename_for_text(text)
    filepath = os.path.join(AUDIO_DIR, safe_name)
    # If file exists, return immediate
    if os.path.exists(filepath):
        return filepath, f"/audio/{safe_name}"
    try:
        tts = gTTS(text=text, lang="en")
        tts.save(filepath)
        return filepath, f"/audio/{safe_name}"
    except Exception as e:
        print("gTTS failed (sync):", e)
        return None, None

def make_audio_for_text_async(text):
    """Fire-and-forget background audio generation (if not already present)."""
    safe_name = make_audio_filename_for_text(text)
    filepath = os.path.join(AUDIO_DIR, safe_name)
    if os.path.exists(filepath):
        return filepath, f"/audio/{safe_name}"
    # spawn background thread to create it
    def task():
        try:
            tts = gTTS(text=text, lang="en")
            tts.save(filepath)
            print(f"[audio] generated {filepath}")
        except Exception as e:
            print("gTTS failed (async):", e)
    Thread(target=task, daemon=True).start()
    return None, None
# -------------------------------
# Endpoints
# -------------------------------

@app.route("/")
def health():
    return jsonify({"status": "Yoga Pose Backend Running"}), 200

@app.route("/register_user", methods=["POST"])
def register_user():
    payload = request.get_json()
    if not payload or "email" not in payload:
        return jsonify({"error": "email required"}), 400
    email = payload["email"]
    users_collection.update_one({"email": email}, {"$set": payload}, upsert=True)
    return jsonify({"message": "User registered/updated"}), 201

@app.route("/api/poses_list", methods=["GET"])
def get_pose_list():
    # return classes the label_encoder knows
    poses = list(label_encoder.classes_)
    return jsonify({"poses": poses}), 200

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "message": "No data provided"}), 400

    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({"success": False, "message": "Email and password required"}), 400

    user = users_collection.find_one({"email": email})
    if not user:
        return jsonify({"success": False, "message": "User not found"}), 404

    if user.get("password") != password:
        return jsonify({"success": False, "message": "Incorrect password"}), 401

    return jsonify({
        "success": True,
        "message": "Login successful",
        "user": {
            "name": user.get("name"),
            "email": user.get("email"),
            "age": user.get("age"),
        }
    }), 200

@app.route("/api/analyze_pose", methods=["POST"])
def analyze_pose():
    if "image" not in request.files:
        return jsonify({"error": "image file is required"}), 400
    image_file = request.files["image"]
    if image_file.filename == "" or not allowed_file(image_file.filename):
        return jsonify({"error": "Invalid image file"}), 400

    email = request.form.get("email")
    selected_pose = request.form.get("selected_pose")

    file_bytes = np.frombuffer(image_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Could not decode image"}), 400

    features, landmarks = extract_feature_vector_from_image_bgr(img)
    if features is None:
        return jsonify({"error": "No human pose detected"}), 422

    scaled = scaler.transform(features)
    preds = model.predict(scaled, verbose=0)
    idx = int(np.argmax(preds[0]))
    confidence = float(preds[0][idx])
    predicted_pose = str(label_encoder.inverse_transform([idx])[0])

    symmetry = compute_symmetry_metric(landmarks.landmark)
    score = compute_score(confidence, symmetry)

    # Add keypoints for overlay
    keypoints = evaluate_joint_correctness(landmarks.landmark)

    # 🧘 Dynamic, contextual feedback
    if predicted_pose != selected_pose:
        feedback = f"Your pose seems closer to {predicted_pose}. Try adjusting your balance and posture for {selected_pose}."
    elif confidence < 0.75:
        feedback = f"Nice try on {selected_pose}, refine your alignment for a steadier form."
    else:
        feedback = f"Perfect {selected_pose}! You’re maintaining great alignment and control."

    record = {
        "email": email,
        "selected_pose": selected_pose,
        "predicted_pose": predicted_pose,
        "confidence": confidence,
        "score": score,
        "feedback": feedback,
        "timestamp": datetime.utcnow().isoformat()
    }
    poses_collection.insert_one(record)

    # ✅ Send text feedback only
    return jsonify({
        "predicted_pose": predicted_pose,
        "confidence": confidence,
        "score": score,
        "feedback": feedback,
        "keypoints": keypoints
    }), 200

@app.route("/api/poses_history", methods=["GET"])
def poses_history():
    email = request.args.get("email")
    query = {}
    if email:
        query["email"] = email
    docs = list(poses_collection.find(query, {"_id": 0}))
    return jsonify(docs), 200

# Serve audio files
@app.route("/audio/<path:filename>", methods=["GET"])
def serve_audio(filename):
    return send_from_directory(AUDIO_DIR, filename, as_attachment=False)

@app.route("/api/user_progress", methods=["GET"])
def user_progress():
    """Return summary stats for the given user's progress."""
    email = request.args.get("email")
    if not email:
        return jsonify({"error": "email required"}), 400

    user_poses = list(poses_collection.find({"email": email}))
    if not user_poses:
        return jsonify({
            "sessions": 0,
            "avg_accuracy": 0,
            "streak_days": 0
        }), 200

    # total sessions
    sessions = len(user_poses)

    # average confidence (as %)
    avg_accuracy = round(np.mean([p.get("score", 0) for p in user_poses]), 2)

    # Compute streak (continuous days)
    dates = sorted([
        datetime.fromisoformat(p["timestamp"]).date()
        for p in user_poses if "timestamp" in p
    ])
    streak = 1
    max_streak = 1
    for i in range(1, len(dates)):
        if (dates[i] - dates[i - 1]).days == 1:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 1

    return jsonify({
        "sessions": sessions,
        "avg_accuracy": avg_accuracy,
        "streak_days": max_streak
    }), 200

# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  # Render assigns this automatically
    app.run(host="0.0.0.0", port=port, debug=False)

