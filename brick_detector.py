import cv2
import numpy as np
import os
import pickle

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REFERENCE_IMAGE_DIR = os.path.join(BASE_DIR, "Brick photos")
DESCRIPTORS_FILE = os.path.join(BASE_DIR, "reference_descriptors.pkl") # Will store (filename, descriptors, shape)

# ORB Parameters
N_FEATURES_PER_IMAGE = 500
MATCHER_THRESHOLD = 0.75
MIN_MATCH_COUNT = 13

# Color Segmentation Parameters (Green)
LOWER_GREEN = np.array([35, 50, 50])
UPPER_GREEN = np.array([85, 255, 255])
MIN_CONTOUR_AREA = 500

orb = cv2.ORB_create(nfeatures=N_FEATURES_PER_IMAGE)
bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

def get_reference_data():
    """
    Loads or computes and saves descriptors and shapes for all reference brick images.
    The saved data in .pkl file is a list of tuples: (filename, descriptors, shape).
    Shape is (height, width).
    """
    if os.path.exists(DESCRIPTORS_FILE):
        try:
            with open(DESCRIPTORS_FILE, 'rb') as f:
                data = pickle.load(f)
            # Validate structure: list of (filename, descriptors, shape)
            if isinstance(data, list) and data and \
               isinstance(data[0], tuple) and len(data[0]) == 3 and \
               isinstance(data[0][1], np.ndarray) and \
               isinstance(data[0][2], tuple) and len(data[0][2]) == 2:
                print("Loaded reference data (descriptors, shape) from cache.")
                return data
            else:
                print("Cached data file is in an old/invalid format. Recomputing.")
        except Exception as e:
            print(f"Error loading or validating cached data: {e}. Recomputing.")

    print("Computing reference descriptors and shapes...")
    reference_data_list = []

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
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  Warning: Could not read reference image {filename}. Skipping.")
            continue
        print(f"  Image {filename} read successfully. Shape: {img.shape}")

        _, descriptors = orb.detectAndCompute(img, None) # We only need descriptors for matching now
        print(f"  ORB descriptor computation done for {filename}.")

        if descriptors is not None and len(descriptors) > 0:
            img_shape = img.shape[:2]
            reference_data_list.append((filename, descriptors, img_shape))
            print(f"  Found {len(descriptors)} descriptors for {filename}. Shape: {img_shape}")
        else:
            print(f"  Warning: No descriptors found for reference image {filename}.")

    print(f"Finished processing all reference images. Total data entries collected: {len(reference_data_list)}.")
    if not reference_data_list:
        print("No valid reference data was collected.")
        return []

    print(f"Attempting to save {len(reference_data_list)} reference data entries to {DESCRIPTORS_FILE}...")
    with open(DESCRIPTORS_FILE, 'wb') as f:
        pickle.dump(reference_data_list, f)
    print(f"Saved {len(reference_data_list)} reference data entries to {DESCRIPTORS_FILE}.")
    return reference_data_list

REFERENCE_DATA = get_reference_data() # Stores (filename, descriptors, shape)

def detect_brick(query_image_path):
    if not REFERENCE_DATA:
        return False, "Error: No reference data loaded.", None, None, "unknown"

    query_img_bgr = cv2.imread(query_image_path)
    if query_img_bgr is None:
        return False, f"Error: Could not read query image: {query_image_path}", None, None, "unknown"

    query_img_gray = cv2.cvtColor(query_img_bgr, cv2.COLOR_BGR2GRAY)
    query_img_hsv = cv2.cvtColor(query_img_bgr, cv2.COLOR_BGR2HSV)

    bounding_box_final = None # Will be [x,y,w,h] from cv2.boundingRect
    centroid_final = None
    orientation_final = "unknown"
    detected_status_final = False
    status_messages = []
    potential_detections = []

    green_mask = cv2.inRange(query_img_hsv, LOWER_GREEN, UPPER_GREEN)
    kernel = np.ones((5,5),np.uint8)
    green_mask_morphed = cv2.erode(green_mask, kernel, iterations = 1)
    green_mask_morphed = cv2.dilate(green_mask_morphed, kernel, iterations = 2)
    contours, _ = cv2.findContours(green_mask_morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    significant_contours_found = False
    if contours:
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for contour_idx, contour in enumerate(sorted_contours):
            if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
                if not significant_contours_found and contour_idx == 0:
                    status_messages.append(f"Largest green region (Area: {cv2.contourArea(contour):.0f}) is below min area {MIN_CONTOUR_AREA}.")
                break
            significant_contours_found = True
            contour_mask_for_features = np.zeros_like(query_img_gray)
            cv2.drawContours(contour_mask_for_features, [contour], -1, (255), thickness=cv2.FILLED)
            masked_query_img_gray = cv2.bitwise_and(query_img_gray, query_img_gray, mask=contour_mask_for_features)
            query_kp_contour, query_des_contour = orb.detectAndCompute(masked_query_img_gray, None)

            if query_des_contour is None or len(query_des_contour) < MIN_MATCH_COUNT / 2:
                status_messages.append(f"Contour {contour_idx} (Area {cv2.contourArea(contour):.0f}): Few features ({len(query_des_contour if query_des_contour is not None else [])}).")
                continue
            status_messages.append(f"Contour {contour_idx} (Area {cv2.contourArea(contour):.0f}, {len(query_kp_contour) if query_kp_contour else 0} features):")
            best_match_for_this_contour = {'score': -1, 'ref_filename': None}

            for ref_filename, ref_des, _ in REFERENCE_DATA: # We don't use ref_shape or ref_kp here anymore
                if ref_des is None: continue
                matches = bf_matcher.knnMatch(query_des_contour, ref_des, k=2)
                if not matches or not matches[0] or len(matches[0]) < 2: continue
                good_matches = [m for m, n in matches if m.distance < MATCHER_THRESHOLD * n.distance]

                if len(good_matches) >= MIN_MATCH_COUNT and len(good_matches) > best_match_for_this_contour['score']:
                    best_match_for_this_contour['score'] = len(good_matches)
                    best_match_for_this_contour['ref_filename'] = ref_filename

            if best_match_for_this_contour['ref_filename']: # A feature match was made for this contour
                ref_filename = best_match_for_this_contour['ref_filename']
                score = best_match_for_this_contour['score']
                status_messages.append(f"  Best match in contour: {ref_filename} ({score} matches).")

                x_br, y_br, w_br, h_br = cv2.boundingRect(contour)
                bbox = [x_br, y_br, w_br, h_br] # Use [x,y,w,h] format
                moments = cv2.moments(contour)
                cx = int(moments["m10"]/moments["m00"]) if moments["m00"]!=0 else x_br+w_br//2
                cy = int(moments["m01"]/moments["m00"]) if moments["m00"]!=0 else y_br+h_br//2
                potential_detections.append({'score': score,
                                             'status_detail':f"Feature match in contour with {ref_filename}",
                                             'bounding_box':bbox, 'centroid':(cx,cy), 'source': 'contour_features'})
                break # Processed the best contour with a feature match

    if not potential_detections: # No feature match in contours, or no significant contours
        status_messages.append("No confident feature match in contours or no significant contours. Trying whole image.")
        query_kp_whole, query_des_whole = orb.detectAndCompute(query_img_gray, None)

        if query_des_whole is None or len(query_des_whole) < MIN_MATCH_COUNT:
            status_messages.append(f"Not enough features in whole query image ({len(query_des_whole if query_des_whole is not None else [])}).")
        else:
            status_messages.append(f"Whole image ({len(query_kp_whole) if query_kp_whole else 0} features):")
            best_match_for_whole = {'score': -1, 'ref_filename': None, 'matched_query_kp': None}

            for ref_filename, ref_des, _ in REFERENCE_DATA:
                if ref_des is None: continue
                matches = bf_matcher.knnMatch(query_des_whole, ref_des, k=2)
                if not matches or not matches[0] or len(matches[0]) < 2: continue
                good_matches = [m for m,n in matches if m.distance < MATCHER_THRESHOLD * n.distance]

                if len(good_matches) >= MIN_MATCH_COUNT and len(good_matches) > best_match_for_whole['score']:
                    best_match_for_whole['score'] = len(good_matches)
                    best_match_for_whole['ref_filename'] = ref_filename
                    # Store matched keypoints from query image
                    best_match_for_whole['matched_query_kp'] = [query_kp_whole[m.queryIdx] for m in good_matches]

            if best_match_for_whole['ref_filename']:
                ref_filename = best_match_for_whole['ref_filename']
                score = best_match_for_whole['score']
                status_messages.append(f"  Best whole image match: {ref_filename} ({score} matches).")

                # Bounding box around matched keypoints in query image - very rough
                if best_match_for_whole['matched_query_kp']:
                    points = np.float32([kp.pt for kp in best_match_for_whole['matched_query_kp']])
                    if len(points) > 0:
                        x_br, y_br, w_br, h_br = cv2.boundingRect(points)
                        bbox = [x_br, y_br, w_br, h_br]
                        cx = x_br + w_br // 2
                        cy = y_br + h_br // 2
                        potential_detections.append({'score':score,
                                                     'status_detail':f"Feature match on WHOLE IMAGE with {ref_filename}",
                                                     'bounding_box':bbox, 'centroid':(cx,cy), 'source': 'whole_image_features'})

    if potential_detections:
        best_detection = sorted(potential_detections, key=lambda x: x['score'], reverse=True)[0]
        detected_status_final = True
        status_messages.insert(0, best_detection['status_detail'])
        bounding_box_final = best_detection['bounding_box'] # This is [x,y,w,h]
        centroid_final = best_detection['centroid']
        if bounding_box_final: # bounding_box_final is [x,y,w,h]
            w_rect, h_rect = bounding_box_final[2], bounding_box_final[3]
            if w_rect == 0 or h_rect == 0: orientation_final = "degenerate_bbox"
            else:
                actual_width = max(w_rect, h_rect) # Use actual width/height from upright rect
                actual_height = min(w_rect, h_rect)
                aspect_ratio = actual_width / actual_height if actual_height > 0 else float('inf')
                SIDEWAYS_THRESHOLD = 1.6
                HEAD_ON_THRESHOLD_MAX = 1.3
                if aspect_ratio > SIDEWAYS_THRESHOLD: orientation_final = "sideways"
                elif aspect_ratio < HEAD_ON_THRESHOLD_MAX: orientation_final = "head-on"
                else: orientation_final = "angled"
            status_messages.append(f"Orientation ({aspect_ratio:.2f} from w:{w_rect:.0f},h:{h_rect:.0f}): {orientation_final}")
    elif significant_contours_found :
        status_messages.append("Significant green regions found, but no feature match to reference bricks.")
    else:
        status_messages.append("No brick detected (no significant green or feature matches).")

    final_status_message = " | ".join(status_messages)
    return detected_status_final, final_status_message, bounding_box_final, centroid_final, orientation_final

if __name__ == '__main__':
    print("Running brick_detector.py directly for testing or initial descriptor generation...")
    if not REFERENCE_DATA: # Changed variable name
        print("WARNING: REFERENCE_DATA is empty.")
        globals()['REFERENCE_DATA'] = get_reference_data() # Changed variable name
        if not REFERENCE_DATA: print("ERROR: Failed to load or compute reference data even from __main__.")
        else: print(f"Successfully loaded/computed {len(REFERENCE_DATA)} data entries from __main__ fallback.")
    else: print(f"REFERENCE_DATA already populated with {len(REFERENCE_DATA)} sets.")

    if REFERENCE_DATA and os.path.exists(REFERENCE_IMAGE_DIR):
        image_files = [f for f in os.listdir(REFERENCE_IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if image_files:
            test_query_image_name = image_files[0]
            test_query_image_path = os.path.join(REFERENCE_IMAGE_DIR, test_query_image_name)
            print(f"\n--- Testing detect_brick with: {test_query_image_path} ---")
            detected, status, bbox, centroid, orientation = detect_brick(test_query_image_path)
            print(f"Detection result: {detected}")
            print(f"Status: {status}")
            print(f"Bounding Box (x,y,w,h): {bbox}")
            print(f"Centroid: {centroid}")
            print(f"Orientation: {orientation}")
            print("--- Test detection finished ---")
            dummy_black_img_path = "dummy_black_test.jpg"
            cv2.imwrite(dummy_black_img_path, np.zeros((100,100,3), dtype=np.uint8))
            print(f"\n--- Testing detect_brick with: {dummy_black_img_path} ---")
            detected, status, bbox, centroid, orientation = detect_brick(dummy_black_img_path)
            print(f"Detection result: {detected}")
            print(f"Status: {status}")
            print(f"Bounding Box (x,y,w,h): {bbox}")
            print(f"Centroid: {centroid}")
            print(f"Orientation: {orientation}")
            print("--- Test detection finished ---")
            os.remove(dummy_black_img_path)
        else: print("No images found in reference directory to run a test detection.")
    else: print("Skipping direct test detection: No reference data or images directory.")
    print("\nbrick_detector.py direct execution finished.")

print("brick_detector.py loaded (globally or by import)")
