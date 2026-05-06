"""
dataset_setup.py
────────────────
Helper to quickly populate the dataset with sample images from a folder
(e.g., photos already on your PC) instead of using the webcam.

Usage:
    python dataset_setup.py --person "Monika" --source "~/Downloads/my_photos"
"""

import cv2
import os
import argparse
import shutil

HAARCASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
IMG_SIZE = (200, 200)


def extract_and_save_faces(source_dir: str, person_name: str, split: float = 0.8):
    """
    From a folder of raw photos, detect faces, crop, resize, and
    split into train / test sets automatically.
    """
    face_cascade = cv2.CascadeClassifier(HAARCASCADE)
    train_dir = f"dataset/train/{person_name}"
    test_dir  = f"dataset/test/{person_name}"
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir,  exist_ok=True)

    images = [f for f in os.listdir(source_dir)
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    saved, skipped = 0, 0

    for idx, img_file in enumerate(images):
        img_path = os.path.join(source_dir, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            skipped += 1
            continue

        faces = face_cascade.detectMultiScale(img, 1.1, 5, minSize=(40, 40))

        if len(faces) == 0:
            # Try the whole image as a face crop
            face = cv2.resize(img, IMG_SIZE)
            faces_to_save = [face]
        else:
            faces_to_save = []
            for (x, y, w, h) in faces:
                face = cv2.resize(img[y:y+h, x:x+w], IMG_SIZE)
                face = cv2.equalizeHist(face)
                faces_to_save.append(face)

        for i, face in enumerate(faces_to_save):
            dest = train_dir if (saved / max(1, len(images))) < split else test_dir
            filename = f"{person_name}_{idx:04d}_{i}.jpg"
            cv2.imwrite(os.path.join(dest, filename), face)
            saved += 1

    print(f"[INFO] '{person_name}': {saved} faces saved ({skipped} skipped)")
    train_count = len(os.listdir(train_dir))
    test_count  = len(os.listdir(test_dir))
    print(f"       Train: {train_count} | Test: {test_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare face dataset from a photo folder")
    parser.add_argument("--person", required=True, help="Person's name (used as label)")
    parser.add_argument("--source", required=True, help="Folder containing raw photos")
    parser.add_argument("--split",  type=float, default=0.8, help="Train/test split ratio")
    args = parser.parse_args()

    src = os.path.expanduser(args.source)
    if not os.path.exists(src):
        print(f"[ERROR] Source folder not found: {src}")
    else:
        extract_and_save_faces(src, args.person, args.split)
