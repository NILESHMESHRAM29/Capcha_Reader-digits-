from flask import Flask, request, jsonify
import numpy as np
import cv2
import re
import logging
import ddddocr

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ocr_main = ddddocr.DdddOcr(show_ad=False)
ocr_beta = ddddocr.DdddOcr(beta=True, show_ad=False)
logging.info("ddddocr main + beta models loaded.")


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 — RED LINE REMOVAL  (surgical, minimal collateral damage)
# ─────────────────────────────────────────────────────────────────────────────

def remove_red_line(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove the diagonal red interference line via HSV masking + inpainting.

    KEY CHANGES vs old code
    ───────────────────────
    OLD: dilation kernel (2×3) + inpaintRadius=5
         → mask was over-dilated; inpainting smeared digit pixels that
           overlapped the line, corrupting stroke geometry for OCR.

    NEW: no dilation (raw pixel mask only) + inpaintRadius=2
         → only the actual red pixels are masked.
         → inpaintRadius=2 fills the 1-2 px line without reaching digit strokes.

    HSV range kept wide [0-15 / 155-180] so anti-aliased line edges are still
    captured — but WITHOUT dilation that was the real culprit of smearing.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0,   60, 40])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([155,  60, 40])
    upper_red2 = np.array([180, 255, 255])
    red_mask = (
        cv2.inRange(hsv, lower_red1, upper_red1) |
        cv2.inRange(hsv, lower_red2, upper_red2)
    )

    # ── NO dilation — avoid growing mask into digit strokes ──
    # Only fill real red pixels; radius=2 is enough for a 1-2 px anti-aliased edge
    img_clean = cv2.inpaint(img, red_mask, inpaintRadius=2, flags=cv2.INPAINT_TELEA)
    return img_clean, red_mask


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 — DOT-GRID SUPPRESSION  (new, was completely missing before)
# ─────────────────────────────────────────────────────────────────────────────

def suppress_dot_grid(img_bgr: np.ndarray) -> np.ndarray:
    """
    Attenuate the regular dot-grid background without touching digit strokes.

    Approach: morphological TOP-HAT on the VALUE channel.
    ──────────────────────────────────────────────────────
    The grid dots are small (~2 px diameter) and evenly spaced.
    Digit strokes are wide (4-8 px) connected blobs.

    A BLACKHAT operation with a kernel slightly LARGER than one grid cell
    extracts dark features (dots + digit strokes) against a light background.
    But our bg is medium-gray, so instead we use:

      1. Convert to grayscale
      2. Morphological CLOSE with 3×3 kernel → smooths over individual dots
         (dots are smaller than kernel → filled in) while preserving digit
         strokes (wider than kernel → survive closing)
      3. Subtract: gray - close_result → emphasizes things DARKER than their
         local neighbourhood → dots partially cancel; digit strokes remain
      4. Invert and normalise → black digits on white background

    Why this works for THIS captcha:
    - Grid dots ≈ 2px, spacing ≈ 4px → 3×3 close fills them
    - Digit strokes ≈ 5-8px → NOT filled by 3×3 close → survive subtraction
    - The result is a cleaner binary-like image where digits stand out
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Close over grid dots (kernel must be > dot size but < stroke width)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel_close)

    # Subtract: values where gray < closed → darker than surroundings (= strokes + dots)
    # diff is brightest at digit strokes (they are wider → survive closing → bigger gap)
    diff = cv2.subtract(closed, gray)  # 0-255, clipped; bright = dark feature

    # Stretch contrast
    diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

    # Otsu threshold → clean binary
    _, binary = cv2.threshold(diff_norm, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Invert: digits=black on white (ddddocr handles both but consistent is better)
    binary_inv = cv2.bitwise_not(binary)
    return binary_inv


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3 — UPSCALING  (new, was missing before)
# ─────────────────────────────────────────────────────────────────────────────

def upscale(img: np.ndarray, factor: float = 2.5) -> np.ndarray:
    """
    Upscale to give ddddocr larger digit features to work with.

    Why 2.5×:
    - Original images ≈ 130×40 px → digits ≈ 15-20px tall
    - After 2.5×: ≈ 325×100 px → digits ≈ 37-50px tall
    - ddddocr's CNN was trained at ~32-64px glyph height; 40px original is borderline
    - INTER_CUBIC preserves edge sharpness better than INTER_LINEAR for digit strokes
    - Avoid INTER_NEAREST on color/gray (aliasing); use it only on strict binary masks
    """
    h, w = img.shape[:2]
    new_w, new_h = int(w * factor), int(h * factor)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


# ─────────────────────────────────────────────────────────────────────────────
#  PREPROCESSING VARIANTS  (used in ensemble)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_color_upscaled(image_bytes: bytes) -> bytes:
    """
    Variant A: red-removed COLOR image, 2.5× upscaled.
    Best for beta model — retains all color texture ddddocr was trained on.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cv2.imdecode failed")
    img_clean, _ = remove_red_line(img)
    img_up = upscale(img_clean, factor=2.5)
    _, enc = cv2.imencode('.png', img_up)
    return enc.tobytes()


def preprocess_grid_suppressed(image_bytes: bytes) -> bytes:
    """
    Variant B: dot-grid suppressed binary + 2.5× upscale.
    Best for main model — removes background noise, gives clean digit strokes.
    The red line is removed BEFORE grid suppression so its pixels don't survive
    as dark artifacts in the binary image.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cv2.imdecode failed")
    img_clean, _ = remove_red_line(img)
    binary = suppress_dot_grid(img_clean)          # black digits on white
    binary_up = upscale(binary, factor=2.5)
    _, enc = cv2.imencode('.png', binary_up)
    return enc.tobytes()


def preprocess_clahe_upscaled(image_bytes: bytes) -> bytes:
    """
    Variant C: CLAHE contrast-enhanced grayscale + 2.5× upscale.
    Fallback — useful when digit strokes have low absolute darkness
    but CLAHE brings them out relative to the local grid background.

    Why CLAHE here (not globally):
    - Global histogram equalization flattens the grid dots into noise islands.
    - CLAHE (tileGridSize=8×4, clipLimit=3.0) operates on small tiles that
      roughly correspond to 2-3 digit widths, so it boosts local digit contrast
      without amplifying the grid pattern globally.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cv2.imdecode failed")
    img_clean, _ = remove_red_line(img)
    gray = cv2.cvtColor(img_clean, cv2.COLOR_BGR2GRAY)

    # Tile size = (8,4): small enough to be local, large enough not to amplify dots
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 4))
    enhanced = clahe.apply(gray)

    enhanced_up = upscale(enhanced, factor=2.5)
    _, enc = cv2.imencode('.png', enhanced_up)
    return enc.tobytes()


def preprocess_raw_upscaled(image_bytes: bytes) -> bytes:
    """
    Variant D: raw image (no preprocessing), just upscaled.
    Kept as a safety net — if all preprocessing variants degrade the image
    for some reason, the raw+upscaled is still better than raw alone.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cv2.imdecode failed")
    img_up = upscale(img, factor=2.5)
    _, enc = cv2.imencode('.png', img_up)
    return enc.tobytes()


# ─────────────────────────────────────────────────────────────────────────────
#  ENSEMBLE WITH VOTING
# ─────────────────────────────────────────────────────────────────────────────

def extract_digits_only(raw_text: str) -> str:
    return re.sub(r'[^0-9]', '', raw_text).strip()


def ensemble_classify(image_bytes: bytes, expected_length: int = 6) -> dict:
    """
    Run 8 inference candidates (4 preprocessings × 2 models) and select best.

    ── Candidate order (priority) ──────────────────────────────────────────
    The order matters because the first exact-length match wins.
    We order from "most reliable" to "fallback":

      1. grid-suppressed  × beta   ← cleanest signal, best model
      2. grid-suppressed  × main
      3. color-upscaled   × beta   ← full texture, best model
      4. color-upscaled   × main
      5. CLAHE-upscaled   × beta   ← contrast-boosted fallback
      6. CLAHE-upscaled   × main
      7. raw-upscaled     × beta   ← last resort
      8. raw-upscaled     × main

    ── Selection logic ─────────────────────────────────────────────────────
    1. VOTE: if 3+ candidates agree on the same 6-digit string → return it
       (strong consensus; the correct answer doesn't need to be first)
    2. FIRST EXACT: first candidate with exactly 6 digits
    3. MAJORITY LENGTH: if no 6-digit result, pick most common digit count,
       then return the first candidate with that count
    4. LONGEST: absolute fallback — most digits found

    Why voting before first-exact:
    - In testing, sometimes candidates 3-4 return the right answer while
      candidates 1-2 return 5 digits (line crossing a digit).
    - Voting catches this consensus even when the "best" model failed.
    """
    prep_variants = [
        ("grid_suppressed",  preprocess_grid_suppressed(image_bytes)),
        ("color_upscaled",   preprocess_color_upscaled(image_bytes)),
        ("clahe_upscaled",   preprocess_clahe_upscaled(image_bytes)),
        ("raw_upscaled",     preprocess_raw_upscaled(image_bytes)),
    ]

    candidates = []
    for variant_name, prep_bytes in prep_variants:
        for model_name, ocr_model in [("beta", ocr_beta), ("main", ocr_main)]:
            raw = ocr_model.classification(prep_bytes)
            digits = extract_digits_only(raw)
            candidates.append({
                "raw": raw,
                "digits": digits,
                "variant": variant_name,
                "model": model_name,
            })
            logging.debug(f"  [{variant_name}/{model_name}] raw='{raw}' → digits='{digits}'")

    # ── Strategy 1: Voting (3+ agree) ──
    from collections import Counter
    exact_results = [c["digits"] for c in candidates if len(c["digits"]) == expected_length]
    if exact_results:
        counts = Counter(exact_results)
        top_value, top_count = counts.most_common(1)[0]
        if top_count >= 3:
            winner = next(c for c in candidates if c["digits"] == top_value)
            winner["method"] = f"vote_{top_count}of{len(candidates)}"
            return winner

    # ── Strategy 2: First exact-length match (in priority order) ──
    for c in candidates:
        if len(c["digits"]) == expected_length:
            c["method"] = "first_exact"
            return c

    # ── Strategy 3: Most common digit count ──
    all_lengths = [len(c["digits"]) for c in candidates if len(c["digits"]) > 0]
    if all_lengths:
        most_common_len = Counter(all_lengths).most_common(1)[0][0]
        for c in candidates:
            if len(c["digits"]) == most_common_len:
                c["method"] = "majority_length"
                return c

    # ── Strategy 4: Longest fallback ──
    best = max(candidates, key=lambda c: len(c["digits"]))
    best["method"] = "longest_fallback"
    return best


# ─────────────────────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "model": "ddddocr_ensemble_v2"})


@app.route('/read-captcha', methods=['POST'])
def read_captcha():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Empty filename."}), 400

    try:
        image_bytes = file.read()
        result = ensemble_classify(image_bytes, expected_length=6)

        captcha = result["digits"]
        raw_text = result["raw"]

        logging.info(
            f"Ensemble OCR → raw='{raw_text}', digits='{captcha}' "
            f"({len(captcha)} chars), method={result.get('method')}, "
            f"variant={result.get('variant')}, model={result.get('model')}"
        )

        response = {
            "captcha": captcha,
            "raw": raw_text,
            "source": "ddddocr_ensemble_v2",
            "method": result.get("method"),
            "variant": result.get("variant"),
            "model": result.get("model"),
        }
        if len(captcha) != 6:
            response["warning"] = f"Expected 6 digits, got {len(captcha)}"

        return jsonify(response)

    except Exception as e:
        logging.error(f"Error processing captcha: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=False)