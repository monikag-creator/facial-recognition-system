"""
Facial Recognition System
=========================
Author: Monika
Description: A complete facial recognition pipeline using OpenCV and LBPH algorithm.
             Supports dataset collection, model training, and real-time recognition.
"""

import cv2
import numpy as np
import os
import json
import time
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
CONFIG = {
    "dataset_path":     "dataset/train",
    "test_path":        "dataset/test",
    "model_path":       "models/lbph_model.yml",
    "labels_path":      "models/labels.json",
    "results_path":     "results",
    "img_size":         (200, 200),          # resize all faces to this
    "confidence_threshold": 80,              # LBPH confidence (lower = more confident)
    "haarcascade_path": cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
}


# ─────────────────────────────────────────────
# STEP 1 — DATASET COLLECTION
# ─────────────────────────────────────────────

def collect_faces_from_webcam(person_name: str, num_samples: int = 50):
    """
    Capture face samples from webcam for a given person.
    Saves grayscale cropped face images to dataset/train/<person_name>/.
    """
    save_dir = os.path.join(CONFIG["dataset_path"], person_name)
    os.makedirs(save_dir, exist_ok=True)

    face_cascade = cv2.CascadeClassifier(CONFIG["haarcascade_path"])
    cap = cv2.VideoCapture(0)
    count = 0

    print(f"\n[INFO] Collecting {num_samples} samples for '{person_name}'. Press 'q' to quit early.")

    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60))

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_roi, CONFIG["img_size"])
            filename = os.path.join(save_dir, f"{person_name}_{count:04d}.jpg")
            cv2.imwrite(filename, face_resized)
            count += 1

            # Draw rectangle on display frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Captured: {count}/{num_samples}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Collecting Faces — Press 'q' to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Saved {count} images for '{person_name}' → {save_dir}")


# ─────────────────────────────────────────────
# STEP 2 — PREPROCESSING
# ─────────────────────────────────────────────

def preprocess_image(img_path: str) -> np.ndarray:
    """
    Load, convert to grayscale, resize, and apply histogram equalization.
    Returns a normalized face array.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    img = cv2.resize(img, CONFIG["img_size"])
    img = cv2.equalizeHist(img)           # improve contrast
    return img


def load_dataset(dataset_path: str):
    """
    Walk through dataset directory. Each sub-folder = one person label.
    Returns:
        faces  — list of preprocessed face arrays
        labels — list of integer label IDs
        label_map — {int_id: person_name}
    """
    faces, labels = [], []
    label_map = {}
    label_id = 0

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    for person_name in sorted(os.listdir(dataset_path)):
        person_dir = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_dir):
            continue

        label_map[label_id] = person_name
        img_count = 0

        for img_file in os.listdir(person_dir):
            if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(person_dir, img_file)
            try:
                face = preprocess_image(img_path)
                faces.append(face)
                labels.append(label_id)
                img_count += 1
            except Exception as e:
                print(f"[WARN] Skipping {img_path}: {e}")

        print(f"  → Loaded {img_count} images for '{person_name}' (label {label_id})")
        label_id += 1

    print(f"[INFO] Dataset loaded: {len(faces)} images, {len(label_map)} classes")
    return faces, labels, label_map


# ─────────────────────────────────────────────
# STEP 3 — MODEL TRAINING
# ─────────────────────────────────────────────

def train_model(faces: list, labels: list, label_map: dict):
    """
    Train an LBPH (Local Binary Pattern Histogram) face recognizer.
    Saves model weights and label mapping to disk.
    """
    os.makedirs("models", exist_ok=True)

    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1,
        neighbors=8,
        grid_x=8,
        grid_y=8
    )

    print("\n[INFO] Training LBPH model...")
    start = time.time()
    recognizer.train(faces, np.array(labels))
    elapsed = time.time() - start

    recognizer.save(CONFIG["model_path"])

    # Save label map as JSON
    with open(CONFIG["labels_path"], "w") as f:
        json.dump({str(k): v for k, v in label_map.items()}, f, indent=2)

    print(f"[INFO] Training complete in {elapsed:.2f}s")
    print(f"[INFO] Model saved  → {CONFIG['model_path']}")
    print(f"[INFO] Labels saved → {CONFIG['labels_path']}")
    return recognizer, label_map


# ─────────────────────────────────────────────
# STEP 4 — EVALUATION ON TEST SET
# ─────────────────────────────────────────────

def evaluate_model(recognizer, label_map: dict):
    """
    Evaluate trained model on images inside dataset/test/<person_name>/.
    Computes accuracy, false positives, false negatives, and per-class breakdown.
    """
    test_path = CONFIG["test_path"]
    if not os.path.exists(test_path):
        print("[WARN] No test directory found. Skipping evaluation.")
        return {}

    face_cascade = cv2.CascadeClassifier(CONFIG["haarcascade_path"])
    reverse_map = {v: int(k) for k, v in label_map.items()}

    results = {
        "total": 0, "correct": 0,
        "false_positives": 0, "false_negatives": 0,
        "per_class": {}
    }

    for person_name in sorted(os.listdir(test_path)):
        person_dir = os.path.join(test_path, person_name)
        if not os.path.isdir(person_dir):
            continue

        true_label = reverse_map.get(person_name, -1)
        per = {"total": 0, "correct": 0, "fp": 0, "fn": 0}

        for img_file in os.listdir(person_dir):
            if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(person_dir, img_file)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, CONFIG["img_size"])
            img = cv2.equalizeHist(img)

            # Try to detect face first (for images with context/background)
            faces_rects = face_cascade.detectMultiScale(img, 1.1, 5, minSize=(40, 40))
            if len(faces_rects):
                x, y, w, h = faces_rects[0]
                face_roi = img[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, CONFIG["img_size"])
            else:
                face_roi = img     # assume whole image is already a face crop

            pred_label, confidence = recognizer.predict(face_roi)
            per["total"] += 1

            if confidence < CONFIG["confidence_threshold"]:
                if pred_label == true_label:
                    per["correct"] += 1
                else:
                    per["fp"] += 1
            else:
                per["fn"] += 1      # too uncertain = rejected

        acc = (per["correct"] / per["total"] * 100) if per["total"] else 0
        per["accuracy"] = round(acc, 2)
        results["per_class"][person_name] = per
        results["total"]   += per["total"]
        results["correct"] += per["correct"]
        results["false_positives"] += per["fp"]
        results["false_negatives"] += per["fn"]

    overall = (results["correct"] / results["total"] * 100) if results["total"] else 0
    results["accuracy"] = round(overall, 2)

    # Print summary
    print("\n" + "="*50)
    print("  EVALUATION RESULTS")
    print("="*50)
    print(f"  Overall Accuracy : {results['accuracy']}%")
    print(f"  Total Samples    : {results['total']}")
    print(f"  Correct          : {results['correct']}")
    print(f"  False Positives  : {results['false_positives']}")
    print(f"  False Negatives  : {results['false_negatives']}")
    print("-"*50)
    for name, p in results["per_class"].items():
        print(f"  {name:<20} Acc: {p['accuracy']}%  ({p['correct']}/{p['total']})")
    print("="*50)

    # Save results JSON
    os.makedirs(CONFIG["results_path"], exist_ok=True)
    result_file = os.path.join(CONFIG["results_path"], "evaluation_results.json")
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] Results saved → {result_file}")

    return results


# ─────────────────────────────────────────────
# STEP 5 — REAL-TIME RECOGNITION
# ─────────────────────────────────────────────

def run_realtime_recognition():
    """
    Load trained model and run live face recognition via webcam.
    Press 'q' to quit, 's' to save a screenshot.
    """
    if not os.path.exists(CONFIG["model_path"]):
        print("[ERROR] No trained model found. Run train() first.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(CONFIG["model_path"])

    with open(CONFIG["labels_path"]) as f:
        label_map = {int(k): v for k, v in json.load(f).items()}

    face_cascade = cv2.CascadeClassifier(CONFIG["haarcascade_path"])
    cap = cv2.VideoCapture(0)

    print("\n[INFO] Starting real-time recognition. Press 'q' to quit, 's' for screenshot.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(60, 60))

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, CONFIG["img_size"])
            face_roi = cv2.equalizeHist(face_roi)

            label_id, confidence = recognizer.predict(face_roi)

            if confidence < CONFIG["confidence_threshold"]:
                name = label_map.get(label_id, "Unknown")
                conf_text = f"{100 - confidence:.1f}% match"
                color = (0, 220, 100)
            else:
                name = "Unknown"
                conf_text = "Low confidence"
                color = (0, 60, 220)

            # Draw bounding box + label
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.rectangle(frame, (x, y-40), (x+w, y), color, -1)
            cv2.putText(frame, name,      (x+6, y-20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,255,255), 1)
            cv2.putText(frame, conf_text, (x+6, y-4),  cv2.FONT_HERSHEY_DUPLEX, 0.4, (220,220,220), 1)

        # Status bar
        ts = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, f"Faces detected: {len(faces)}  |  {ts}", (10, frame.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Facial Recognition — Press 'q' quit | 's' screenshot", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = os.path.join(CONFIG["results_path"], f"screenshot_{int(time.time())}.jpg")
            cv2.imwrite(fname, frame)
            print(f"[INFO] Screenshot saved → {fname}")

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# STEP 6 — STATIC IMAGE RECOGNITION
# ─────────────────────────────────────────────

def recognize_from_image(image_path: str, save_output: bool = True):
    """
    Run face recognition on a single static image file.
    Optionally saves the annotated output to results/.
    """
    if not os.path.exists(CONFIG["model_path"]):
        print("[ERROR] No trained model found. Run train() first.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(CONFIG["model_path"])

    with open(CONFIG["labels_path"]) as f:
        label_map = {int(k): v for k, v in json.load(f).items()}

    face_cascade = cv2.CascadeClassifier(CONFIG["haarcascade_path"])
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
    print(f"[INFO] Detected {len(faces)} face(s) in {image_path}")

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, CONFIG["img_size"])
        face_roi = cv2.equalizeHist(face_roi)

        label_id, confidence = recognizer.predict(face_roi)
        name = label_map.get(label_id, "Unknown") if confidence < CONFIG["confidence_threshold"] else "Unknown"
        conf_pct = max(0, 100 - confidence)

        color = (0, 220, 100) if name != "Unknown" else (0, 60, 220)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.rectangle(frame, (x, y-36), (x+w, y), color, -1)
        cv2.putText(frame, f"{name} ({conf_pct:.1f}%)", (x+4, y-10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1)

        print(f"  → {name} | confidence score: {confidence:.2f}")

    if save_output:
        os.makedirs(CONFIG["results_path"], exist_ok=True)
        out_path = os.path.join(CONFIG["results_path"], "output_" + os.path.basename(image_path))
        cv2.imwrite(out_path, frame)
        print(f"[INFO] Annotated image saved → {out_path}")

    cv2.imshow("Recognition Result", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# MAIN — FULL PIPELINE
# ─────────────────────────────────────────────

def run_full_pipeline():
    """Run train → evaluate pipeline end-to-end on existing dataset."""
    print("\n" + "="*50)
    print("  FACIAL RECOGNITION SYSTEM — FULL PIPELINE")
    print("="*50)

    print("\n[STEP 1] Loading & preprocessing dataset...")
    faces, labels, label_map = load_dataset(CONFIG["dataset_path"])

    if len(faces) == 0:
        print("[ERROR] No training images found. Add images to dataset/train/<person_name>/")
        return

    print("\n[STEP 2] Training LBPH model...")
    recognizer, label_map = train_model(faces, labels, label_map)

    print("\n[STEP 3] Evaluating on test set...")
    evaluate_model(recognizer, label_map)

    print("\n[DONE] Pipeline complete. Run run_realtime_recognition() to start live mode.")


if __name__ == "__main__":
    # ── Change this block to control what runs ──────────────────────
    #
    # Option A: Full train + eval pipeline
    run_full_pipeline()
    #
    # Option B: Collect new face data from webcam
    # collect_faces_from_webcam("YourName", num_samples=60)
    #
    # Option C: Real-time webcam recognition
    # run_realtime_recognition()
    #
    # Option D: Recognize faces in a single image
    # recognize_from_image("path/to/photo.jpg")
    # ────────────────────────────────────────────────────────────────
