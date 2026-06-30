# Facial Recognition System — OpenCV + LBPH

A real-time facial recognition system built using OpenCV's LBPH (Local Binary Patterns Histogram) algorithm, capable of identifying registered users through a live webcam feed.

## Problem Statement
Traditional authentication methods such as passwords and PINs are vulnerable to theft, sharing, and brute-force attacks. Organizations and individuals need a fast, contactless, and harder-to-forge authentication method. Facial recognition offers a biometric alternative that improves both security and user convenience, but many existing solutions are either too resource-heavy or require cloud dependency, making them impractical for lightweight local applications.

## Tech Stack
- **Language:** Python
- **Computer Vision:** OpenCV (cv2)
- **Recognition Algorithm:** LBPH (Local Binary Patterns Histogram)
- **Face Detection:** Haar Cascade Classifier
- **Data Handling:** NumPy, OS, Pickle

## Approach
1. Built a custom face dataset by capturing multiple images per subject under varying lighting and angles using a webcam.
2. Used Haar Cascade Classifiers to detect and crop faces from each frame in real time.
3. Trained an LBPH face recognizer on the labeled face dataset.
4. Implemented a live recognition pipeline that detects faces from webcam feed and overlays bounding boxes with subject name and confidence score.
5. Tuned recognition thresholds to minimize false positives while maintaining responsiveness.
6. Achieved **94% recognition accuracy** on the test set across multiple subjects.

## Business Solution
This system can be deployed as a low-cost, offline biometric authentication layer for small businesses, smart attendance systems in schools/colleges, or secure access control for personal devices and home automation — without relying on costly cloud-based facial recognition APIs. It offers a privacy-friendly, on-device alternative that keeps biometric data local rather than sending it to third-party servers.

## Results
| Metric | Score |
|---|---|
| Recognition Accuracy | 94% |

## How to Run
```bash
git clone https://github.com/monikag-creator/facial-recognition-system.git
cd facial-recognition-system
pip install -r requirements.txt
python train.py      # to register new faces
python recognize.py  # to start live recognition
```

## Author
**Monika G** — [LinkedIn](https://www.linkedin.com/in/monika-g-4a2904388) | [GitHub](https://github.com/monikag-creator)
