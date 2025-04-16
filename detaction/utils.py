import re
import os
import csv
import cv2
import numpy as np
from time import timezone
from paddleocr import PaddleOCR


def paddle_ocr(frame, x1, y1, x2, y2):
    cropped_frame = frame[y1:y2, x1:x2]
 
    gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
  
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)

    gaussian_blur = cv2.GaussianBlur(clahe_img, (0, 0), 3)
    unsharp_img = cv2.addWeighted(clahe_img, 2.5, gaussian_blur, -0.5, 0)

    thresh = cv2.adaptiveThreshold(unsharp_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
 
    preprocessed_images = [
        cropped_frame,    
        clahe_img,        
        unsharp_img,      
        thresh,           
    ]
    
    # Initialize PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, use_gpu=False, lang='en', ocr_version='PP-OCRv3')
    
    all_results = []
 
    for img in preprocessed_images:
        results = ocr.ocr(img, det=False, rec=True, cls=False)
        if results:
            all_results.extend(results)
    best_text = ""
    best_score = 0
    
    for res in all_results:
        if not res or len(res) == 0:
            continue
            
        text = res[0][0]
        score = res[0][1]
        
        if np.isnan(score) or score < 0.5:
            continue

        clean_text = re.sub(r'[^A-Za-z0-9\s\-]', '', text).strip()
        clean_text = re.sub(r'\s+', ' ', clean_text) 
        
        if not clean_text:
            continue
            
        if score > best_score:
            best_score = score
            best_text = clean_text
    if best_text:
        normalized_text = standardize_license_plate(best_text)
        return normalized_text, best_score
    
    return "", 0


def standardize_license_plate(plate_text):
    raw_text = re.sub(r'[\s\-]', '', plate_text.upper())  # Remove spaces and dashes, uppercase input

    # OCR Misread Character Replacements
    replacements = {
        'O': '0',  # Letter O to number 0
        'I': '1',  # Letter I to number 1
        'Z': '2',  # Letter Z to number 2
        'S': '5',  # Letter S to number 5
        'G': '6',  # Letter G to number 6
        'B': '8'   # Letter B to number 8
    }

    # **Pakistan Format: ABC 1234**
    pakistani_plate_pattern = re.match(r'^([A-Z]{2,3})(\d{1,4})$', raw_text)
    if pakistani_plate_pattern:
        prefix, number = pakistani_plate_pattern.groups()
        for old, new in replacements.items():
            number = number.replace(old, new)
        return f"{prefix} {number}"

    # **India Format: XX00XX0000**
    indian_plate_pattern = re.match(r'^([A-Z]{2})(\d{1,2})([A-Z]{1,3})(\d{1,4})$', raw_text)
    if indian_plate_pattern:
        state_code, region_code, series, number = indian_plate_pattern.groups()
        for old, new in replacements.items():
            region_code = region_code.replace(old, new)
            number = number.replace(old, new)
        return f"{state_code} {region_code} {series} {number}"

    # **Malaysia / Indonesia / Philippines: ABC 1234**
    general_plate_pattern = re.match(r'^([A-Z]{1,3})(\d{3,4})([A-Z]{0,2})$', raw_text)
    if general_plate_pattern:
        prefix, number, suffix = general_plate_pattern.groups()
        for old, new in replacements.items():
            number = number.replace(old, new)
        return f"{prefix} {number} {suffix}".strip()

    # If format does not match, apply general OCR corrections
    corrected_text = raw_text
    for old, new in replacements.items():
        corrected_text = corrected_text.replace(old, new)

    return corrected_text
def normalize_plate(plate):
    ocr_corrections = {
        'O': '0', 'Q': '0', 'D': 'Q', 'I': '1', 'Z': '2', 
        'S': '5', 'G': '6', 'B': '8'
    }
    plate = re.sub(r'[^A-Za-z0-9\u4E00-\u9FFF]', '', plate.upper())  # Keep Asian characters
    plate = ''.join(ocr_corrections.get(char, char) for char in plate)  # Apply OCR corrections
    return plate

def levenshtein(s1, s2):
    """ Compute the Levenshtein distance with OCR-aware substitutions. """
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            if c1 == c2:
                cost = 0
            elif (c1 in "0OQ" and c2 in "0OQ") or (c1 in "1I" and c2 in "1I") or (c1 in "2Z" and c2 in "2Z") or (c1 in "Q" and c2 in "D"):
                cost = 0.5  # OCR substitution penalty
            else:
                cost = 1

            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + cost
            current_row.append(min(insertions, deletions, substitutions))

        previous_row = current_row

    return previous_row[-1]

def is_same_license_plate(plate1, plate2, threshold=0.9):
    clean1, clean2 = normalize_plate(plate1), normalize_plate(plate2)
    
    if abs(len(clean1) - len(clean2)) > 2:  
        return False
    
    edit_distance = levenshtein(clean1, clean2)
    max_len = max(len(clean1), len(clean2))
    
    similarity = 1 - (edit_distance / max_len if max_len > 0 else 0)
    
    return similarity >= threshold

def is_same_numplate(plate1, plate2, plate3, threshold=0.9):
    return (
        is_same_license_plate(plate1, plate2, threshold) and
        is_same_license_plate(plate2, plate3, threshold) and
        is_same_license_plate(plate1, plate3, threshold)
    )
def update_license_plate_db(detected_plate, confidence, vehicle_id, detection_records=None):
    if detection_records is None:
        detection_records = []

    standardized_plate = standardize_license_plate(detected_plate)

    for record in detection_records:
        if is_same_license_plate(standardized_plate, record['plate_number']):
            if confidence > record['confidence']:
                record['plate_number'] = standardized_plate
                record['confidence'] = confidence
            if vehicle_id not in record['vehicle_ids']:
                record['vehicle_ids'].append(vehicle_id)
                
            return record 
    new_record = {
        'plate_number': standardized_plate,
        'confidence': confidence,
        'vehicle_ids': [vehicle_id],
        'first_seen': timezone.now()
    }
    
    detection_records.append(new_record)
    return new_record

def save_license_plates(license_plates, start_time, end_time):
    if not license_plates:
        print("No license plates detected in this interval.")
        return
    
    csv_file_path = "data_store/vehicle_license_plates.csv"
    file_exists = os.path.exists(csv_file_path)
    
    existing_entries = set()
    if file_exists:
        with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  
            existing_entries = {(row[1], row[2]) for row in reader}
    
    with open(csv_file_path, mode="a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        
        if not file_exists:
            writer.writerow(["Timestamp", "Vehicle ID", "License Plate", "file_name","Confidence"])
        
        for plate_info in license_plates:
            if plate_info['plate']:
                if plate_info['vehicle_id'] not in existing_entries and plate_info['plate'] not in existing_entries:
                    writer.writerow([
                        end_time.isoformat(), 
                        plate_info['vehicle_id'], 
                        plate_info['plate'],
                        plate_info['crop_filename'], 
                        plate_info['confidence']
                    ])
                    existing_entries.add((plate_info['vehicle_id'], plate_info['plate']))