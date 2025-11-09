import re
import json
from pdf2image import convert_from_path
import pytesseract
import requests
from io import BytesIO
import cv2
import numpy as np
from PIL import Image
import fitz
import sys
import argparse

DEEPAI_API_KEY = 'bed0f1b8-4988-4371-9e31-75e705f7f892'
POPPLER_PATH = r"C:\Users\vegas\Downloads\poppler-25.07.0\Library\bin" # path to Poppler bin folder
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # path to tesseract.exe

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

SSN_RE = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
CC_RE = re.compile(r'\b(?:\d[ -]*?){13,16}\b')
CONFIDENTIAL_KEYWORDS = ['confidential', 'internal', 'do not distribute', 'for internal', 'employee']
UNSAFE_KEYWORDS = ['child sexual abuse', 'self-harm', 'suicide', 'explosive', 'hateful', 'terror']

def extract_embedded_images(pdf_path):
    images = []
    with fitz.open(pdf_path) as doc:
        for page_index, page in enumerate(doc, start=1):
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                pil_img = Image.open(BytesIO(image_bytes)).convert("RGB")
                images.append((page_index, pil_img))
    return images

def deepai_detect_violence_bytes(img_bytes):
    try:
        response = requests.post(
            "https://api.deepai.org/api/violent-content-detection",
            files={'image': img_bytes},
            headers={'api-key': DEEPAI_API_KEY},
            timeout=30
        )
        data = response.json()
        score = data.get('output', {}).get('violent', 0.0)
        return float(score)
    except Exception as e:
        print(f"[Warning] DeepAI request failed: {e}")
        return 0.0

def detect_violence_regions(pil_img):
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    h, w = cv_img.shape[:2]

    # Use Tesseract to get text boxes
    data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)
    # Build a mask where text exists
    mask = np.zeros((h, w), dtype=np.uint8)

    n_boxes = len(data['text'])
    for i in range(n_boxes):
        conf = data['conf'][i]
        text_val = data['text'][i].strip() if data['text'][i] else ''
        
        try:
            conf_val = int(conf)
        except:
            conf_val = -1
        if conf_val > 30 and text_val:
            x, y, bw, bh = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            # Expand box slightly to be safe
            pad_x = max(2, int(bw * 0.05))
            pad_y = max(2, int(bh * 0.05))
            x0 = max(0, x - pad_x)
            y0 = max(0, y - pad_y)
            x1 = min(w, x + bw + pad_x)
            y1 = min(h, y + bh + pad_y)
            cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)

    inv_mask = cv2.bitwise_not(mask)

    # Find contours of non-text regions
    # Clean small noise with morphology
    kernel = np.ones((5,5), np.uint8)
    inv_mask_clean = cv2.morphologyEx(inv_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(inv_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_score = 0.0
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = bw * bh
        # Filter out tiny regions (noise) and extremely large regions that are mostly page background
        if area < 4000:  # tuneable threshold
            continue
        # Optional: ignore regions that are nearly full-page whitespace (e.g., area > 90% page)
        if area > 0.98 * (w * h):
            continue

        roi = cv_img[y:y+bh, x:x+bw]
        pil_crop = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        buffered = BytesIO()
        pil_crop.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()

        score = deepai_detect_violence_bytes(img_bytes)
        if score > max_score:
            max_score = score

    return float(max_score)

def classify_pdf(path):
    try:
        pages = convert_from_path(path, dpi=200, poppler_path=POPPLER_PATH)
    except Exception as e:
        print(f"Error converting PDF: {e}")
        return None

    doc_evidence = []
    label = "Public"

    for i, img in enumerate(pages, start=1):
        try:
            text = pytesseract.image_to_string(img)
        except Exception as e:
            print(f"OCR failed on page {i}: {e}")
            continue

        page_hits = []
        for kw in UNSAFE_KEYWORDS:
            if kw.lower() in text.lower():
                page_hits.append({'type':'UNSAFE_TEXT', 'snippet': kw, 'page': i})
        
        
        embedded_images = extract_embedded_images(path)
        for (pnum, pil_img) in embedded_images:
            gore_score = detect_violence_regions(pil_img)  # or deepai_detect_violence_bytes variant
            if gore_score > 0.1:
                page_hits.append({
                    'type': 'UNSAFE_IMAGE',
                    'snippet': 'Gore/Violence detected',
                    'page': pnum,
                    'score': gore_score
                })

        if page_hits:
            doc_evidence.extend(page_hits)

        # Check for SSN
        ssn_match = SSN_RE.search(text)
        if ssn_match:
            page_hits.append({'type':'SSN', 'snippet': ssn_match.group(), 'page':i})

        # Check for credit card
        cc_match = CC_RE.search(text)
        if cc_match:
            page_hits.append({'type':'CC', 'snippet': cc_match.group(), 'page':i})

        # Check for confidential keywords
        for kw in CONFIDENTIAL_KEYWORDS:
            if kw.lower() in text.lower():
                page_hits.append({'type':'CONFIDENTIAL', 'snippet':kw, 'page':i})

        if page_hits:
            doc_evidence.extend(page_hits)

    # Decide label conservatively
    types = {e['type'] for e in doc_evidence}
    if 'SSN' in types or 'CC' in types:
        label = "Highly Sensitive"
    elif 'UNSAFE_TEXT' in types or 'UNSAFE_IMAGE' in types:
        label = 'Unsafe'
    elif 'CONFIDENTIAL' in types:
        label = "Confidential"
    else:
        label = "Public"

    out = {
        'file': path,
        'label': label,
        'evidence': doc_evidence
    }
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify a PDF document by sensitivity level.")
    parser.add_argument("pdf_path", help="Path to the PDF file to analyze")
    args = parser.parse_args()
    pdf_path = args.pdf_path

    result = classify_pdf(pdf_path)

    if result:
        print(json.dumps(result, indent=2))
    else:
        sys.exit("Error: could not process file.")
        
    
