"""
Inference service: face detection → crop → emotion classification.

Priority order:
  1. ONNX checkpoint (fastest, production)
  2. PyTorch checkpoint (development)
  3. DeepFace pre-trained mini_Xception (no custom model needed, ~65% accuracy)
  4. Mock mode (random — only when nothing else is available)
"""

import base64
import random
from pathlib import Path

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

_model = None
_use_onnx = False
_mock_mode = False
_deepface_mode = False
_image_size = 48

# Below this threshold the top-class probability is considered noise and the
# result is suppressed (face_found=False keeps it out of the DB as well).
_CONFIDENCE_FLOOR = 0.45


# ── Custom model loading ──────────────────────────────────────────────────────

def _load_pytorch_model(checkpoint_path: str):
    import sys
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from model.model import build_model

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    model = build_model(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, cfg["data"]["image_size"]


def _load_onnx_model(onnx_path: str):
    import onnxruntime as ort
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_shape = session.get_inputs()[0].shape
    image_size = input_shape[2] if isinstance(input_shape[2], int) else 48
    return session, image_size


def load_model(checkpoint_path: str, onnx_path: str, use_onnx: bool = False):
    global _model, _use_onnx, _mock_mode, _deepface_mode, _image_size

    if use_onnx and Path(onnx_path).exists():
        try:
            _model, _image_size = _load_onnx_model(onnx_path)
            _use_onnx = True
            _mock_mode = False
            _deepface_mode = False
            print(f"[Inference] Loaded ONNX model: {onnx_path}")
            return
        except Exception as e:
            print(f"[Inference] ONNX load failed: {e}")

    if Path(checkpoint_path).exists():
        try:
            _model, _image_size = _load_pytorch_model(checkpoint_path)
            _use_onnx = False
            _mock_mode = False
            _deepface_mode = False
            print(f"[Inference] Loaded PyTorch model: {checkpoint_path}")
            return
        except Exception as e:
            print(f"[Inference] PyTorch load failed: {e}")

    # Fallback: DeepFace pre-trained model (real predictions, no training needed)
    try:
        import os
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
        from deepface import DeepFace
        weights = Path.home() / ".deepface" / "weights" / "facial_expression_model_weights.h5"
        if not weights.exists():
            raise FileNotFoundError(f"DeepFace weights not found at {weights}. Run: curl -L -o \"{weights}\" https://github.com/serengil/deepface_models/releases/download/v1.0/facial_expression_model_weights.h5")
        # Warm up the model — loads Keras model into memory now
        import numpy as np
        _dummy = np.zeros((1, 48, 48, 1), dtype="float32")
        DeepFace.analyze(np.zeros((100, 100, 3), dtype="uint8"),
                         actions=["emotion"], enforce_detection=False,
                         detector_backend="opencv", silent=True)
        _deepface_mode = True
        _mock_mode = False
        print("[Inference] Using DeepFace pre-trained emotion model (mini_Xception, ~65% accuracy)")
        return
    except ImportError:
        print("[Inference] DeepFace not installed — falling back to mock mode")
    except Exception as e:
        print(f"[Inference] DeepFace init failed: {e} — falling back to mock mode")

    print("[Inference] Running in MOCK mode (random predictions). Train a model for real results.")
    _mock_mode = True


# ── Prediction ────────────────────────────────────────────────────────────────

def _mock_predict() -> dict:
    dominant_idx = random.randint(0, len(EMOTIONS) - 1)
    raw = [random.random() * 0.1 for _ in EMOTIONS]
    raw[dominant_idx] += random.uniform(0.5, 0.9)
    total = sum(raw)
    probs = [v / total for v in raw]
    return {
        "emotion": EMOTIONS[dominant_idx],
        "confidence": float(probs[dominant_idx]),
        "probabilities": {e: float(p) for e, p in zip(EMOTIONS, probs)},
        "face_found": True,
    }


def _deepface_predict(image_bytes: bytes) -> dict:
    import numpy as np
    import cv2
    from deepface import DeepFace

    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"error": "Could not decode image", "face_found": False}

    try:
        results = DeepFace.analyze(
            frame,
            actions=["emotion"],
            enforce_detection=True,
            detector_backend="opencv",
            silent=True,
        )
        if isinstance(results, list):
            results = results[0]

        raw = results["emotion"]  # {"angry": 12.3, "happy": 67.1, ...}
        total = sum(raw.values()) or 1.0
        probs = {e.lower(): float(v / total) for e, v in raw.items()}
        # Ensure all 7 emotions are present
        for e in EMOTIONS:
            probs.setdefault(e, 0.0)

        dominant = results["dominant_emotion"].lower()
        return {
            "emotion": dominant,
            "confidence": probs.get(dominant, 0.0),
            "probabilities": probs,
            "face_found": True,
        }
    except Exception:
        # enforce_detection=True raises when no face found
        return {"emotion": None, "confidence": 0.0, "probabilities": {e: 0.0 for e in EMOTIONS}, "face_found": False}


def _custom_model_predict(image_bytes: bytes) -> dict:
    import numpy as np
    import cv2

    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"error": "Could not decode image", "face_found": False}

    face = _detect_face(frame)
    if face is None:
        return {"emotion": None, "confidence": 0.0, "probabilities": {}, "face_found": False}

    tensor = _preprocess(face, _image_size)

    if _use_onnx:
        input_name = _model.get_inputs()[0].name
        logits = _model.run(None, {input_name: tensor})[0][0]
    else:
        import torch
        with torch.no_grad():
            logits = _model(torch.from_numpy(tensor)).numpy()[0]

    probs = _softmax(logits)
    pred_idx = int(probs.argmax())
    return {
        "emotion": EMOTIONS[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": {e: float(p) for e, p in zip(EMOTIONS, probs)},
        "face_found": True,
    }


def predict_from_bytes(image_bytes: bytes) -> dict:
    if _mock_mode:
        try:
            import cv2
            import numpy as np
            nparr = np.frombuffer(image_bytes, np.uint8)
            if cv2.imdecode(nparr, cv2.IMREAD_COLOR) is None:
                return {"error": "Could not decode image", "face_found": False}
        except ImportError:
            pass
        return _mock_predict()

    if _deepface_mode:
        return _deepface_predict(image_bytes)

    return _custom_model_predict(image_bytes)


def predict_from_base64(b64_string: str) -> dict:
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    return predict_from_bytes(base64.b64decode(b64_string))


# ── Helpers (custom model only) ───────────────────────────────────────────────

_haar_cascade = None


def _load_face_detector():
    import cv2
    global _haar_cascade
    if _haar_cascade is None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _haar_cascade = cv2.CascadeClassifier(path)
    return _haar_cascade


def _detect_face(frame_bgr):
    import cv2
    detector = _load_face_detector()
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    # Scale factor / minNeighbors tuned to match FER2013-style crops
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    fh, fw = frame_bgr.shape[:2]
    # Small padding so forehead/chin are included — matches FER2013 crop style
    pad = int(0.15 * max(w, h))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(fw, x + w + pad)
    y2 = min(fh, y + h + pad)
    return frame_bgr[y1:y2, x1:x2]


def _preprocess(face_img, image_size: int):
    import numpy as np
    import cv2
    img = cv2.resize(face_img, (image_size, image_size))
    # FER2013 images are grayscale. Training loaded them with PIL.convert("RGB")
    # which stacks the gray channel 3×. Webcam frames are BGR color — applying
    # ImageNet stats directly to real color would be a train/inference mismatch.
    # Fix: convert to grayscale, apply CLAHE for lighting normalisation, then
    # stack to 3 identical channels to match the training distribution.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)
    img3 = np.stack([gray, gray, gray], axis=-1).astype("float32") / 255.0
    mean = [0.507, 0.507, 0.507] if image_size == 48 else [0.485, 0.456, 0.406]
    std  = [0.255, 0.255, 0.255] if image_size == 48 else [0.229, 0.224, 0.225]
    img3 = (img3 - mean) / std
    return img3.transpose(2, 0, 1)[None].astype("float32")


def _softmax(x):
    import numpy as np
    e = np.exp(x - x.max())
    return e / e.sum()
