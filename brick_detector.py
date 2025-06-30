import cv2
import numpy as np
import os
import pickle

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REFERENCE_IMAGE_DIR = os.path.join(BASE_DIR, "Brick photos")
DESCRIPTORS_FILE = os.path.join(BASE_DIR, "reference_descriptors.pkl")

# ORB Parameters
N_FEATURES_PER_IMAGE = 500  # Max features to detect in each reference image
MATCHER_THRESHOLD = 0.75  # Lowe's ratio test threshold
MIN_MATCH_COUNT = 10  # Minimum number of good matches to consider it a detection

# Color Segmentation Parameters (Green) - These may need tuning
LOWER_GREEN = np.array([35, 50, 50])  # Lower HSV bound for green
UPPER_GREEN = np.array([85, 255, 255]) # Upper HSV bound for green
MIN_CONTOUR_AREA = 500 # Minimum area for a green contour to be considered

orb = cv2.ORB_create(nfeatures=N_FEATURES_PER_IMAGE)
bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False) # crossCheck=False for knnMatch

def get_reference_descriptors():
    """
    Loads or computes and saves descriptors for all reference brick images.
    """
    if os.path.exists(DESCRIPTORS_FILE):
        with open(DESCRIPTORS_FILE, 'rb') as f:
            try:
                data = pickle.load(f)
                # Quick check to see if the format is as expected
                if isinstance(data, list) and all(isinstance(item, tuple) and len(item) == 2 for item in data):
                    print("Loaded reference descriptors from cache.")
                    return data
                else:
                    print("Cached descriptors file is in an unexpected format. Recomputing.")
            except (pickle.UnpicklingError, EOFError) as e:
                print(f"Error loading cached descriptors: {e}. Recomputing.")

    print("Computing reference descriptors...")
    reference_descriptors = []
    if not os.path.exists(REFERENCE_IMAGE_DIR):
        print(f"Error: Reference image directory not found at {REFERENCE_IMAGE_DIR}")
        return []

    image_files = [f for f in os.listdir(REFERENCE_IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"No images found in {REFERENCE_IMAGE_DIR}")
        return []

    print(f"Found {len(image_files)} potential image files in {REFERENCE_IMAGE_DIR}.")

    for i, filename in enumerate(image_files):
        print(f"Processing reference image {i+1}/{len(image_files)}: {filename}...")
        img_path = os.path.join(REFERENCE_IMAGE_DIR, filename)

        print(f"  Reading image: {img_path}")
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  Warning: Could not read reference image {filename}. Skipping.")
            continue
        print(f"  Image {filename} read successfully. Shape: {img.shape}")

        print(f"  Detecting and computing ORB features for {filename}...")
        kp, des = orb.detectAndCompute(img, None)
        print(f"  ORB feature computation done for {filename}.")

        if des is not None and len(des) > 0:
            reference_descriptors.append((filename, des))
            print(f"  Found {len(des)} descriptors for {filename}.")
        else:
            print(f"  Warning: No descriptors found for reference image {filename}.")

    print(f"Finished processing all reference images. Total descriptors collected for {len(reference_descriptors)} images.")
    if not reference_descriptors:
        print("No valid reference descriptors were collected. Please check your images in Brick photos/")

    print(f"Attempting to save {len(reference_descriptors)} reference descriptors to {DESCRIPTORS_FILE}...")
    with open(DESCRIPTORS_FILE, 'wb') as f:
        pickle.dump(reference_descriptors, f)
    print(f"Saved {len(reference_descriptors)} reference descriptors to {DESCRIPTORS_FILE}.")
    return reference_descriptors

# Load reference descriptors when the module is imported
# This is the standard behavior for when the module is imported by app.py
REFERENCE_DESCRIPTORS = get_reference_descriptors()

def detect_brick(query_image_path):
    """
    Detects if a green brick is present in the query image.
    Args:
        query_image_path (str): Path to the query image.
    Returns:
        bool: True if a brick is detected, False otherwise.
        str: A status message.
    """
    if not REFERENCE_DESCRIPTORS:
        return False, "Error: No reference descriptors loaded. Cannot perform matching."

    query_img_bgr = cv2.imread(query_image_path)
    if query_img_bgr is None:
        return False, f"Error: Could not read query image at {query_image_path}"

    query_img_gray = cv2.cvtColor(query_img_bgr, cv2.COLOR_BGR2GRAY)
    query_img_hsv = cv2.cvtColor(query_img_bgr, cv2.COLOR_BGR2HSV)

    # 1. Color Segmentation (Optional Pre-filter)
    green_mask = cv2.inRange(query_img_hsv, LOWER_GREEN, UPPER_GREEN)
    # Morphological Operations
    kernel = np.ones((5,5),np.uint8)
    green_mask = cv2.erode(green_mask, kernel, iterations = 1)
    green_mask = cv2.dilate(green_mask, kernel, iterations = 2)

    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_brick_in_any_contour = False
    status_messages = []

    if not contours:
        status_messages.append("No significant green regions found after initial filtering.")
        print("No green contours found, attempting detection on whole image.")
        query_kp, query_des = orb.detectAndCompute(query_img_gray, None)
        if query_des is None or len(query_des) == 0:
            return False, "No features found in the query image (whole image)."

        for ref_filename, ref_des in REFERENCE_DESCRIPTORS:
            matches = bf_matcher.knnMatch(query_des, ref_des, k=2)
            good_matches = []
            if not matches or (len(matches[0]) < 2 if matches else False):
                 status_messages.append(f"Not enough match candidates for {ref_filename} (whole image).")
                 continue
            for m, n in matches:
                if m.distance < MATCHER_THRESHOLD * n.distance:
                    good_matches.append(m)

            status_messages.append(f"Found {len(good_matches)} good matches with {ref_filename} (whole image).")
            if len(good_matches) >= MIN_MATCH_COUNT:
                detected_brick_in_any_contour = True
                status_messages.append(f"Brick detected based on whole image match with {ref_filename}.")
                break

        final_status = " ".join(status_messages)
        return detected_brick_in_any_contour, final_status

    significant_contours_found = False
    for contour in contours:
        if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
            continue
        significant_contours_found = True

        contour_mask = np.zeros_like(query_img_gray)
        cv2.drawContours(contour_mask, [contour], -1, (255), thickness=cv2.FILLED)
        masked_query_img_gray = cv2.bitwise_and(query_img_gray, query_img_gray, mask=contour_mask)

        query_kp, query_des = orb.detectAndCompute(masked_query_img_gray, None)

        if query_des is None or len(query_des) == 0:
            status_messages.append(f"No features found in a green contour of area {cv2.contourArea(contour)}.")
            continue

        status_messages.append(f"Processing green contour of area {cv2.contourArea(contour)}, found {len(query_des)} features.")

        for ref_filename, ref_des in REFERENCE_DESCRIPTORS:
            matches = bf_matcher.knnMatch(query_des, ref_des, k=2)
            good_matches = []

            if not matches or not all(len(p) == 2 for p in matches if isinstance(p, list) or isinstance(p, tuple)):
                status_messages.append(f"Not enough match candidates or invalid match format for {ref_filename} with current contour.")
                continue

            for m, n in matches:
                if m.distance < MATCHER_THRESHOLD * n.distance:
                    good_matches.append(m)

            status_messages.append(f"Found {len(good_matches)} good matches with {ref_filename} for current contour.")
            if len(good_matches) >= MIN_MATCH_COUNT:
                detected_brick_in_any_contour = True
                status_messages.append(f"Brick detected in contour, matching {ref_filename}.")
                break

        if detected_brick_in_any_contour:
            break

    final_status = " ".join(status_messages)
    if not detected_brick_in_any_contour and not significant_contours_found :
         final_status += " No brick detected."
    elif not detected_brick_in_any_contour and significant_contours_found:
         final_status += " Green regions found, but no match to reference bricks."

    return detected_brick_in_any_contour, final_status

if __name__ == '__main__':
    # This section is for direct script testing.
    # It's useful for initially generating the reference_descriptors.pkl file or for debugging.
    print("Running brick_detector.py directly for testing or initial descriptor generation...")

    # The module-level call to get_reference_descriptors() should have already run.
    # This __main__ block can now be used to verify or perform specific tests.
    if not REFERENCE_DESCRIPTORS:
        print("WARNING: REFERENCE_DESCRIPTORS is empty. This might happen if .pkl file was missing and generation failed,")
        print("or if the initial call at module load didn't populate it.")
        print("Attempting to call get_reference_descriptors() explicitly from __main__ as a fallback...")
        # This re-assignment ensures that if the module-level one failed silently (it shouldn't with current logs),
        # we try again and populate it for any __main__ tests.
        globals()['REFERENCE_DESCRIPTORS'] = get_reference_descriptors() # Update global if needed
        if not REFERENCE_DESCRIPTORS:
            print("ERROR: Failed to load or compute reference descriptors even from __main__.")
        else:
            print(f"Successfully loaded/computed {len(REFERENCE_DESCRIPTORS)} descriptors from __main__ fallback.")
    else:
        print(f"REFERENCE_DESCRIPTORS already populated with {len(REFERENCE_DESCRIPTORS)} sets (likely from cache on import).")

    # Perform a test detection if reference images are available
    if REFERENCE_DESCRIPTORS and os.path.exists(REFERENCE_IMAGE_DIR):
        image_files = [f for f in os.listdir(REFERENCE_IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if image_files:
            test_query_image_name = image_files[0] # Use the first reference image as a query
            test_query_image_path = os.path.join(REFERENCE_IMAGE_DIR, test_query_image_name)

            print(f"\n--- Testing detect_brick with: {test_query_image_path} ---")
            detected, status = detect_brick(test_query_image_path)
            print(f"Detection result: {detected}")
            print(f"Status: {status}")
            print("--- Test detection finished ---")
        else:
            print("No images found in reference directory to run a test detection.")
    else:
        print("Skipping direct test detection: No reference descriptors or reference images directory not found.")

    print("\nbrick_detector.py direct execution finished.")

print("brick_detector.py loaded (globally or by import)") # General message
