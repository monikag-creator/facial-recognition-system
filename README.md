 Facial Recognition System

> A complete facial recognition pipeline built with **OpenCV** and the **LBPH algorithm** — supporting dataset collection, preprocessing, model training, evaluation, and real-time webcam recognition.
---

 Project Overview
This project implements a facial recognition system from scratch using classical computer vision techniques. It covers the full pipeline:
- **Face Detection** using Haar Cascade Classifiers
- **Preprocessing** with grayscale conversion + histogram equalization
- **Recognition** using Local Binary Pattern Histograms (LBPH)
- **Evaluation** with accuracy, false positive, and false negative metrics
- **Real-time recognition** via webcam

Why LBPH?
| Feature | LBPH | Deep Learning |
|---|---|---|

| Training data needed | ~30–100 images | Thousands |

| Runs without GPU |  Yes |  Often no |

| Speed | Very fast | Moderate–slow |

| Accuracy (small dataset) | Good | Overkill / poor |

| Interpretability | High | Low (black box) |

LBPH is ideal for academic projects and small datasets — exactly our use case.
---

 Approach Explained

 1. Face Detection — Haar Cascade
OpenCV's `haarcascade_frontalface_default.xml` is a trained classifier that:
- Slides a detection window across the image at multiple scales
- Applies a cascade of simple features (Haar wavelets) to reject non-face regions quickly
- Returns bounding boxes of detected faces

 2. Preprocessing Pipeline
Raw image
   ↓  Convert to grayscale       (remove color noise)
   ↓  Resize to 200×200          (normalize dimensions)
   ↓  Histogram Equalization     (improve contrast, handle lighting)
   ↓  Ready for LBPH


 3. LBPH Face Recognition
LBPH encodes a face by:
1. For each pixel, compare it with its 8 circular neighbors
2. Produce a binary code (1 if neighbor > center, else 0) → 8-bit LBP value
3. Divide the face into a grid (8×8 = 64 cells)
4. Build a histogram of LBP values in each cell
5. Concatenate all histograms → a **feature vector** for the face
During recognition, the feature vector of a new face is compared to stored training vectors using **Chi-Square distance**. Lower distance = better match.
**Confidence threshold:** If the LBPH confidence score > 80, the face is classified as "Unknown" (rejected).
---

 How to Run
Prerequisites

```bash
pip install -r requirements.txt
```

> **Note:** You need `opencv-contrib-python` (not just `opencv-python`) for the LBPH recognizer.
---

 Option A — You Already Have Photos

Use `dataset_setup.py` to auto-extract and split faces from a folder:
```bash
python dataset_setup.py --person "Monika" --source "/path/to/your/photos"
python dataset_setup.py --person "Friend" --source "/path/to/friend/photos"
```

Then run the full pipeline:
```bash
python face_recognition_system.py
```

Option B — Capture from Webcam
Edit the `if __name__ == "__main__"` block in `face_recognition_system.py`:
```python
# Capture 60 images of yourself
collect_faces_from_webcam("Monika", num_samples=60)
```
Run it, then switch to `run_full_pipeline()` and run again.
---

Option C — Real-Time Webcam Recognition
After training:
```python
run_realtime_recognition()
```

Controls:
- `q` — quit
- `s` — save screenshot to `results/`
---

 Option D — Single Image
```python
recognize_from_image("path/to/photo.jpg")
```

---

 Results & Insights
 Performance Metrics Explained
| Metric | Formula | Meaning |
|---|---|---|

| Accuracy | correct / total × 100 | Overall recognition rate |

| False Positive | Wrong person accepted | System says "Alice" but it's "Bob" |

| False Negative | Real person rejected | System says "Unknown" but it's Alice |

Expected Results (well-lit, frontal faces)
| Condition | Typical Accuracy |
|---|---|

| Good lighting, frontal, 50+ images/person | 85–95% |

| Mixed lighting, partial angles | 65–80% |

| Very few images (<20/person) | 50–70% |
---

Challenges & Improvements
**Challenges faced:**
- LBPH is sensitive to lighting variations — histogram equalization partially helps
- Significant pose variation (profile faces) causes recognition failures
- Glasses, masks, or hair covering eyes reduces accuracy

**Possible improvements:**
1. **Deep Learning:** Replace LBPH with FaceNet or ArcFace for ~99% accuracy
2. **Data augmentation:** Flip, rotate, brightness-shift images to increase diversity
3. **Dlib landmarks:** Use facial landmark alignment before recognition for pose invariance
4. **Ensemble:** Combine LBPH + EigenFace + Fisherface predictions
---

Tech Stack
| Tool | Purpose |
|---|---|
| Python 3.10+ | Core language |
| OpenCV 4.8+ | Detection, preprocessing, LBPH |
| NumPy | Array operations |
| Matplotlib / Seaborn | Visualization |
| Scikit-learn | Confusion matrix utility |
---

 Author
**Monika** — B.Tech AI/ML, Chennai  
HCL GUVI AIML Program | IITM Pravartak Certification Track
=======
# facial-recognition-system
Facial Recognition project
>>>>>>> f796835874404509df74b8dd51c3ff8148ceff95
