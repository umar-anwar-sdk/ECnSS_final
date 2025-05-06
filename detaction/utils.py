import re
import os
import cv2
import numpy as np
from paddleocr import PaddleOCR
from collections import Counter

def enhanced_paddle_ocr(frame, x1, y1, x2, y2, ocr_model=None):
    """
    Enhanced license plate recognition using PaddleOCR with optimized preprocessing
    
    Args:
        frame: The original image frame
        x1, y1, x2, y2: Coordinates of the license plate region
        ocr_model: PaddleOCR model instance (will be created if not provided)
    
    Returns:
        tuple: (recognized_text, confidence_score)
    """
    # Initialize OCR model if not provided
    if ocr_model is None:
        # Use PP-OCRv4 with default settings
        ocr_model = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            ocr_version='PP-OCRv4',
            drop_score=0.5,  # Minimum confidence threshold
            use_gpu=False,
            show_log=False,
            enable_mkldnn=False,
            cpu_threads=10
        )
    
    # Add small margin to avoid cutting off characters
    height, width = frame.shape[:2]
    margin = 5
    y1 = max(0, y1 - margin)
    y2 = min(height, y2 + margin)
    x1 = max(0, x1 - margin)
    x2 = min(width, x2 + margin)
    
    # Crop the license plate region
    cropped_frame = frame[y1:y2, x1:x2]
    
    # Check if crop is valid
    if cropped_frame.size == 0:
        print("Warning: Invalid crop region")
        return "", 0
    
    # Resize plate image to optimal size for OCR (maintaining aspect ratio)
    aspect_ratio = cropped_frame.shape[1] / cropped_frame.shape[0]
    target_height = 64  # Optimal height for license plate OCR
    target_width = int(target_height * aspect_ratio)
    resized_frame = cv2.resize(cropped_frame, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
    
    # Skip saving debug images
    
    # Create list of preprocessing methods
    candidates = []
    
    # Method 0: Original resized frame
    candidates.append(("original", resized_frame))
    
    # Method 1: Grayscale
    gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    candidates.append(("gray", gray))
    
    # Method 2: CLAHE enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    candidates.append(("clahe", clahe_img))
    
    # Method 3: Adaptive thresholding (from second implementation)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    candidates.append(("thresh", thresh))
    
    # Method 4: Inverted adaptive thresholding (from second implementation)
    thresh_inv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
    candidates.append(("thresh_inv", thresh_inv))
    
    # Method 5: Morphological operations (from second implementation)
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(thresh_inv, cv2.MORPH_OPEN, kernel)
    candidates.append(("opening", opening))
    
    # Method 6: Bilateral filter (preserves edges while removing noise)
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    candidates.append(("bilateral", bilateral))
    
    # Method 7: Otsu's thresholding
    _, otsu = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidates.append(("otsu", otsu))
    
    # Method 8: Sharpening for text
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(clahe_img, -1, kernel)
    candidates.append(("sharpened", sharpened))
    
    # Method 9: Contrast adjustment
    alpha = 1.3  # Contrast control
    beta = 10    # Brightness control
    adjusted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    candidates.append(("adjusted", adjusted))
    
    # Skip saving preprocessing debug images
    
    # Store all OCR results
    all_results = []
    
    # Process each candidate image
    for name, img in candidates:
        # Convert grayscale images to RGB for PaddleOCR
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if len(img.shape) == 2 else img
        
        # Run OCR with and without detection
        # First run without detection (assuming cropped plate)
        results_no_det = ocr_model.ocr(img_rgb, det=False, rec=True, cls=False)
        
        # Then run with detection (in case the plate contains multiple text regions)
        results_with_det = ocr_model.ocr(img_rgb, det=True, rec=True, cls=True)
        
        # Process results without detection
        if results_no_det and len(results_no_det) > 0:
            for res in results_no_det[0]:
                if res:
                    text, score = res
                    clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                    if len(clean_text) >= 4:  # Most license plates have at least 4 characters
                        all_results.append((clean_text, score, name, "no_det"))
        
        # Process results with detection
        if results_with_det and len(results_with_det) > 0:
            res = results_with_det[0]
            if res:
                # Sort by vertical position (top to bottom)
                sorted_boxes = sorted(res, key=lambda x: (x[0][0][1], x[0][0][0]))
                
                # Get combined text from all boxes
                combined_text = ""
                total_score = 0
                box_count = 0
                
                for box in sorted_boxes:
                    if len(box) >= 2:
                        text = box[1][0]
                        score = box[1][1]
                        combined_text += text
                        total_score += score
                        box_count += 1
                
                if box_count > 0:
                    avg_score = total_score / box_count
                    clean_text = re.sub(r'[^A-Z0-9]', '', combined_text.upper())
                    if len(clean_text) >= 4:
                        all_results.append((clean_text, avg_score, name, "with_det"))
    
    # Sort results by confidence score (highest first)
    all_results.sort(key=lambda x: x[1], reverse=True)
    
    # Log results for debugging
    print("\nTop OCR Results:")
    for text, score, method, det_mode in all_results[:5]:  # Show top 5 results
        print(f"Method: {method}, Detection: {det_mode}, Text: {text}, Score: {score:.4f}")
    
    # Use voting system for more reliable results
    if len(all_results) >= 3:
        # Get the most common text among top results
        top_results = all_results[:min(10, len(all_results))]  # Consider up to top 10 results
        texts = [text for text, _, _, _ in top_results]
        
        # Count occurrences of each text
        text_counts = Counter(texts)
        most_common = text_counts.most_common(1)
        
        if most_common and most_common[0][1] >= 2:  # If a text appears at least twice
            best_text = most_common[0][0]
            best_score = max([score for text, score, _, _ in top_results if text == best_text])
            print(f"Using voting system - Text: {best_text}, Score: {best_score:.4f}")
            corrected_text = smart_standardize_license_plate(best_text)
            return corrected_text, best_score
    
    # If voting doesn't give clear result, use highest confidence
    if all_results:
        best_text, best_score, best_method, det_mode = all_results[0]
        print(f"Best method: {best_method}, Detection: {det_mode}, Text: {best_text}, Score: {best_score:.4f}")
        
        corrected_text = smart_standardize_license_plate(best_text)
        return corrected_text, best_score
    
    return "", 0

def smart_standardize_license_plate(plate_text):
    """
    Smarter standardization of license plate text that considers
    plate format and common OCR errors
    """
    # Remove spaces and convert to uppercase
    raw_text = re.sub(r'[\s\-]', '', plate_text.upper())
    
    # Only keep alphanumeric characters
    raw_text = ''.join(c for c in raw_text if c.isalnum())
    
    # Dictionary of commonly confused characters
    corrected_text = ""
    
    for i, char in enumerate(raw_text):
        # Check for common OCR confusion patterns
        if char == 'O' and (i > 0 and any(c.isdigit() for c in raw_text[max(0, i-2):i])):
            # O is likely a 0 if near digits
            corrected_text += '0'
        elif char == 'I' and (i > 0 and any(c.isdigit() for c in raw_text[max(0, i-2):i])):
            # I is likely a 1 if near digits
            corrected_text += '1'
        elif char == '1' and (i < len(raw_text)-1 and all(c.isalpha() for c in raw_text[max(0, i-1):min(len(raw_text), i+2)])):
            # 1 is likely an I if surrounded by letters
            corrected_text += 'I'
        elif char == '0' and (i < len(raw_text)-1 and all(c.isalpha() for c in raw_text[max(0, i-1):min(len(raw_text), i+2)])):
            # 0 is likely an O if surrounded by letters
            corrected_text += 'O'
        elif char == '8' and (i < len(raw_text)-1 and all(c.isalpha() for c in raw_text[max(0, i-1):min(len(raw_text), i+2)])):
            # 8 is likely a B if surrounded by letters
            corrected_text += 'B'
        elif char == 'B' and (i > 0 and any(c.isdigit() for c in raw_text[max(0, i-2):i])):
            # B is likely an 8 if near digits
            corrected_text += '8'
        elif char == 'S' and (i > 0 and any(c.isdigit() for c in raw_text[max(0, i-2):i])):
            # S is likely a 5 if near digits
            corrected_text += '5'
        elif char == 'Z' and (i > 0 and any(c.isdigit() for c in raw_text[max(0, i-2):i])):
            # Z is likely a 2 if near digits
            corrected_text += '2'
        elif char == 'D' and (i > 0 and any(c.isdigit() for c in raw_text[max(0, i-2):i])):
            # D is likely a 0 if near digits
            corrected_text += '0'
        else:
            corrected_text += char
    
    # Validate against common license plate formats
    if is_valid_license_plate(corrected_text):
        return corrected_text
    
    # If doesn't match common format, return the original but uppercase and clean
    return raw_text

def is_valid_license_plate(text):
    """
    Check if the text matches common license plate formats
    Returns True if valid format, False otherwise
    """
    # Common US license plate formats
    patterns = [
        r'^[A-Z]{3}\d{3,4}$',        # ABC1234
        r'^\d{3,4}[A-Z]{3}$',        # 123ABC
        r'^[A-Z]{2}\d{3,5}$',        # AB12345
        r'^\d{1,3}[A-Z]{3}\d{1,3}$', # 1ABC123
        r'^[A-Z]\d{3,7}$',           # A1234567
        r'^\d{1,3}[A-Z]{2}\d{1,4}$', # 12AB1234
        r'^[A-Z]{1,3}\d{1,5}[A-Z]$', # ABC12A
        r'^[A-Z]{2}[0-9]{2}[A-Z]{3}$' # AA11BBB (UK format)
    ]
    
    for pattern in patterns:
        if re.match(pattern, text):
            return True
    
    # Check basic length and composition as fallback
    if 4 <= len(text) <= 8 and any(c.isdigit() for c in text) and any(c.isalpha() for c in text):
        return True
        
    return False

def save_debug_images(preprocessed_images, output_dir="debug_images"):
    """
    Save preprocessed images for debugging purposes - disabled by default
    
    Args:
        preprocessed_images: Dictionary of preprocessed images
        output_dir: Directory to save images
    """
    # This function is kept for compatibility but doesn't save images by default
    pass

def preprocess_license_plate(frame, x1, y1, x2, y2, save_debug=False):
    """
    Preprocess license plate for OCR
    
    Args:
        frame: The original image frame
        x1, y1, x2, y2: Coordinates of the license plate region
        save_debug: Whether to save debug images
    
    Returns:
        dict: Dictionary containing all preprocessed images
    """
    # Add margin to avoid cutting off characters
    height, width = frame.shape[:2]
    margin = 5
    y1 = max(0, y1 - margin)
    y2 = min(height, y2 + margin)
    x1 = max(0, x1 - margin)
    x2 = min(width, x2 + margin)
    
    # Crop the license plate region
    cropped_frame = frame[y1:y2, x1:x2]
    
    # Check if crop is valid
    if cropped_frame.size == 0:
        print("Warning: Invalid crop region")
        return {}
    
    # Resize plate image to optimal size for OCR (maintaining aspect ratio)
    aspect_ratio = cropped_frame.shape[1] / cropped_frame.shape[0]
    target_height = 64  # Optimal height for license plate OCR
    target_width = int(target_height * aspect_ratio)
    resized_frame = cv2.resize(cropped_frame, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
    
    # Create results dictionary with optimized preprocessing methods
    results = {"original": resized_frame}
    
    # Add grayscale
    gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    results["gray"] = gray
    
    # CLAHE enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    results["clahe"] = clahe_img
    
    # Adaptive thresholding (regular and inverted)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 11, 2)
    results["thresh"] = thresh
    
    thresh_inv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)
    results["thresh_inv"] = thresh_inv
    
    # Morphological operations
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(thresh_inv, cv2.MORPH_OPEN, kernel)
    results["opening"] = opening
    
    # Bilateral filter
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    results["bilateral"] = bilateral
    
    # Otsu's thresholding
    _, otsu = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    results["otsu"] = otsu
    
    # Sharpening
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(clahe_img, -1, kernel)
    results["sharpened"] = sharpened
    
    # Contrast adjustment
    alpha = 1.3  # Contrast control
    beta = 10    # Brightness control
    adjusted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    results["adjusted"] = adjusted
    
    # Skip debug image saving
    
    return results

def test_license_plate_recognition(image_path, x1, y1, x2, y2):
    """
    Test license plate recognition on a single image
    
    Args:
        image_path: Path to the image
        x1, y1, x2, y2: Coordinates of the license plate region
    
    Returns:
        tuple: (recognized_text, confidence_score)
    """
    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not load image from {image_path}")
        return "", 0
    
    # Initialize OCR model with default settings
    ocr_model = PaddleOCR(
        use_angle_cls=True,
        lang='en',
        ocr_version='PP-OCRv4',
        drop_score=0.5,
        use_gpu=False,
        show_log=False,
        enable_mkldnn=False,
        cpu_threads=10
    )
    
    # Process license plate
    text, score = enhanced_paddle_ocr(frame, x1, y1, x2, y2, ocr_model)
    
    print(f"License plate: {text}, Confidence: {score:.4f}")
    return text, score

def visualize_result(image_path, text, score, x1=None, y1=None, x2=None, y2=None):
    """
    Create a visualization of the OCR results
    
    Args:
        image_path: Path to the image
        text: Recognized text
        score: Confidence score
        x1, y1, x2, y2: Optional coordinates to draw bounding box
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Draw bounding box if coordinates are provided
    if all(v is not None for v in [x1, y1, x2, y2]):
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Add text to the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, f"Text: {text}", (10, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(img, f"Confidence: {score:.2f}", (10, 70), font, 1, (0, 255, 0), 2)
    
    # Save the result
    result_path = os.path.splitext(image_path)[0] + "_result.jpg"
    cv2.imwrite(result_path, img)
    print(f"Result visualization saved to: {result_path}")