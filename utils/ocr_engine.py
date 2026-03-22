import cv2
import numpy as np
import easyocr
import pytesseract
import PIL.Image
import json
import os
import re
import time
from google import genai

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Initialize Local OCR Engines (The Backup)
print("Initializing Local Engines: EasyOCR & Tesseract...")
easy_reader = easyocr.Reader(['en'], gpu=False)

# Initialize Gemini with the working 2.5 Lite model
API_KEY = "AIzaSyAYVPo5vjv7s6XDcz-8t3Puw97xRBsQ4qs"  # Put your active key here
client = genai.Client(api_key=API_KEY)

class MedicineOCRExtractor:
    @classmethod
    def find_patterns(cls, text):
        res = {"medicine_name": "NOT DETECTED", "batch": "NOT DETECTED", 
               "mfg_date": "NOT DETECTED", "expiry": "NOT DETECTED", "lic_no": "NOT DETECTED"}
        
        batch_match = re.search(r'(?:B\.?NO\.?|BATCH\s*N?[O0]?|BN|L[O0]T)[\s\:\-\.]*([A-Z0-9]+)', text, re.I)
        if batch_match: res["batch"] = batch_match.group(1)
        return res

def extract_medicine_text(img_path):
    print(f"🔍 Analyzing: {os.path.basename(img_path)}")
    
    # STAGE 1: LOCAL OCR
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tess_text = pytesseract.image_to_string(gray)
    easy_text = " ".join(easy_reader.readtext(img, detail=0))
    combined_raw = f"{tess_text} {easy_text}"

    # STAGE 2: AI REFINEMENT (Using the working Lite model)
    raw_img = PIL.Image.open(img_path)
    
    # THE STRICT MASTER PROMPT
 # THE STRICT MASTER PROMPT
    prompt = """
    Look at the medicine foil and extract the primary brand details.
    
    RULES:
    1. Return EXACTLY ONE single JSON object. Do NOT return an array or list.
    2. "medicine_name": The single main brand name. Ignore generic chemical names.
    3. "batch": Look for "B.No", "BN", or standalone codes. Remove prefixes.
    4. "mfg_date": Convert dates like "MAY 24" to "05/2024". MUST be MM/YYYY.
    5. "expiry": Convert dates like "APR 27" to "04/2027". MUST be MM/YYYY.
    6. "lic_no": Look for "M.L.", "Lic", or "Mfg Lic" followed by a number.
    
    Return EXACTLY this format, replacing the placeholders with the actual detected data:
    {
        "medicine_name": "DETECTED_NAME",
        "batch": "DETECTED_BATCH",
        "mfg_date": "MM/YYYY",
        "expiry": "MM/YYYY",
        "lic_no": "DETECTED_LIC"
    }
    """
    
    max_retries = 3 
    
    for attempt in range(max_retries):
        try:
            # We are using the model that just worked for you!
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite", 
                contents=[prompt, raw_img], 
                config={'response_mime_type': 'application/json'}
            )
            
            # This ensures it reads the single dictionary correctly
            data = json.loads(response.text)
            
            # If the AI stubbornly returns a list anyway, just grab the first item
            if isinstance(data, list):
                data = data[0]
                
            return data

        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg and attempt < max_retries - 1:
                print(f"⏳ Rate limit hit! Waiting 15 seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(15)
                continue
            else:
                print(f"⚠️ AI Unavailable. Switching to Fallback...")
                res = MedicineOCRExtractor.find_patterns(combined_raw)
                words = easy_text.split()
                valid_words = [w for w in words if len(w) > 3 and not w.isnumeric()]
                if valid_words: res["medicine_name"] = valid_words[0]
                return res