import cv2
import numpy as np
import os
import pickle

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REFERENCE_IMAGE_DIR = os.path.join(BASE_DIR, "Brick photos")
DESCRIPTORS_FILE = os.path.join(BASE_DIR, "reference_descriptors.pkl")

# ORB Parameters
N_FEATURES_PER_IMAGE = 500
MATCHER_THRESHOLD = 0.75
MIN_MATCH_COUNT = 13 # Increased from 10

# Color Segmentation Parameters (Green)
LOWER_GREEN = np.array([35, 50, 50])
UPPER_GREEN = np.array([85, 255, 255])
MIN_CONTOUR_AREA = 500

orb = cv2.ORB_create(nfeatures=N_FEATURES_PER_IMAGE)
bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

def get_reference_descriptors():
    """
    Loads or computes and saves keypoints, descriptors, and shapes for all reference brick images.
    The saved data will be a list of tuples: (filename, descriptors, keypoints, shape).
    Shape is (height, width).
    """
    if os.path.exists(DESCRIPTORS_FILE):
        try:
            with open(DESCRIPTORS_FILE, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, list) and data and \
                   isinstance(data[0], tuple) and len(data[0]) == 4 and \
                   (not data[0][2] or (isinstance(data[0][2], list) and (not data[0][2] or hasattr(data[0][2][0], 'pt')))) and \
                   isinstance(data[0][3], tuple) and len(data[0][3]) == 2: # check keypoints is a list
                    print("Loaded reference data (des, kp, shape) from cache.")
                    return data
                else:
                    print("Cached descriptors file is in an old format or invalid. Recomputing with new structure.")
        except Exception as e: # More generic catch for any unpickling or validation issue
            print(f"Error loading or validating cached descriptors: {e}. Recomputing with new structure.")

    print("Computing reference descriptors, keypoints, and shapes...")
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

        keypoints, descriptors = orb.detectAndCompute(img, None)
        print(f"  ORB feature computation done for {filename}.")

        if descriptors is not None and len(descriptors) > 0:
            img_shape = img.shape[:2] # (height, width)
            reference_data_list.append((filename, descriptors, keypoints, img_shape))
            print(f"  Found {len(descriptors)} descriptors and {len(keypoints)} keypoints for {filename}. Shape: {img_shape}")
        else:
            print(f"  Warning: No descriptors found for reference image {filename}.")

    print(f"Finished processing all reference images. Total data entries collected: {len(reference_data_list)}.")
    if not reference_data_list:
        print("No valid reference data (des, kp, shape) was collected.")

    print(f"Attempting to save {len(reference_data_list)} reference data entries to {DESCRIPTORS_FILE}...")
    with open(DESCRIPTORS_FILE, 'wb') as f:
        pickle.dump(reference_data_list, f)
    print(f"Saved {len(reference_data_list)} reference data entries to {DESCRIPTORS_FILE}.")
    return reference_data_list

REFERENCE_DESCRIPTORS = get_reference_descriptors()

def detect_brick(query_image_path):
    """
    Detects if a green brick is present in the query image.
    Also calculates its bounding box, centroid, and orientation if detected.
    Args:
        query_image_path (str): Path to the query image.
    Returns:
        bool: True if a brick is detected, False otherwise.
        str: A status message.
        list | None: List of 4 corner points [(x,y), ...] for the bounding box, or None.
        tuple | None: (cX, cY) for the centroid, or None.
        str: Orientation status ("head-on", "sideways", "angled", "unknown").
    """
    if not REFERENCE_DESCRIPTORS:
        return False, "Error: No reference data loaded.", None, None, "unknown"

    query_img_bgr = cv2.imread(query_image_path)
    if query_img_bgr is None:
        return False, f"Error: Could not read query image: {query_image_path}", None, None, "unknown"

    query_img_gray = cv2.cvtColor(query_img_bgr, cv2.COLOR_BGR2GRAY)
    query_img_hsv = cv2.cvtColor(query_img_bgr, cv2.COLOR_BGR2HSV)
    query_img_height, query_img_width = query_img_gray.shape[:2]

    bounding_box_final = None
    centroid_final = None
    orientation_final = "unknown"
    detected_status_final = False
    status_messages = []

    potential_detections = [] # List to store dicts of potential matches

    # 1. Color Segmentation
    green_mask = cv2.inRange(query_img_hsv, LOWER_GREEN, UPPER_GREEN)
    kernel = np.ones((5,5),np.uint8)
    green_mask_morphed = cv2.erode(green_mask, kernel, iterations = 1)
    green_mask_morphed = cv2.dilate(green_mask_morphed, kernel, iterations = 2)
    contours, _ = cv2.findContours(green_mask_morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 2. Process significant contours for feature matching
    significant_contours_found = False
    if contours:
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for contour_idx, contour in enumerate(sorted_contours):
            if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
                if not significant_contours_found and contour_idx == 0: # Only log for the largest if it's too small
                    status_messages.append(f"Largest green region (Area: {cv2.contourArea(contour):.0f}) is below min area {MIN_CONTOUR_AREA}.")
                break # Since sorted, no need to check smaller ones
            significant_contours_found = True

            contour_mask_for_features = np.zeros_like(query_img_gray)
            cv2.drawContours(contour_mask_for_features, [contour], -1, (255), thickness=cv2.FILLED)
            masked_query_img_gray = cv2.bitwise_and(query_img_gray, query_img_gray, mask=contour_mask_for_features)

            query_kp_contour, query_des_contour = orb.detectAndCompute(masked_query_img_gray, None)

            if query_des_contour is None or len(query_des_contour) < MIN_MATCH_COUNT / 2: # Looser threshold for features within contour
                status_messages.append(f"Contour {contour_idx} (Area {cv2.contourArea(contour):.0f}): Few features ({len(query_des_contour if query_des_contour is not None else [])}).")
                continue

            status_messages.append(f"Contour {contour_idx} (Area {cv2.contourArea(contour):.0f}, {len(query_kp_contour)} features):")

            best_match_for_this_contour = {'score': -1, 'data': None}

            for ref_filename, ref_des, ref_kp, ref_shape in REFERENCE_DESCRIPTORS:
                if ref_des is None or not ref_kp : continue
                matches = bf_matcher.knnMatch(query_des_contour, ref_des, k=2)

                # Ensure matches are valid before trying to access matches[0]
                if not matches or not matches[0] or len(matches[0]) < 2:
                    # status_messages.append(f"  Skipping {ref_filename} for contour due to insufficient match candidates.")
                    continue
                good_matches = [m for m, n in matches if m.distance < MATCHER_THRESHOLD * n.distance]

                if len(good_matches) >= MIN_MATCH_COUNT and len(good_matches) > best_match_for_this_contour['score']:
                    best_match_for_this_contour['score'] = len(good_matches)
                    best_match_for_this_contour['data'] = (ref_filename, good_matches, query_kp_contour, ref_kp, ref_shape)

            if best_match_for_this_contour['data']:
                ref_filename, good_matches, q_kp, r_kp, r_shape = best_match_for_this_contour['data']
                status_messages.append(f"  Best match in contour: {ref_filename} ({len(good_matches)} matches).")

                src_pts = np.float32([r_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([q_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                if len(src_pts) >= 4 and len(dst_pts) >=4: # findHomography needs at least 4 points
                    M_homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    if M_homography is not None:
                        h_ref, w_ref = r_shape
                        pts_ref_corners = np.float32([[0,0],[0,h_ref-1],[w_ref-1,h_ref-1],[w_ref-1,0]]).reshape(-1,1,2)
                        dst_corners = cv2.perspectiveTransform(pts_ref_corners, M_homography)
                        bbox = [tuple(pt[0].astype(int)) for pt in dst_corners]
                        moments = cv2.moments(dst_corners.astype(np.int32))
                        cx = int(moments["m10"]/moments["m00"]) if moments["m00"]!=0 else 0
                        cy = int(moments["m01"]/moments["m00"]) if moments["m00"]!=0 else 0
                        potential_detections.append({'score':len(good_matches), 'status_detail':f"Homography match in contour with {ref_filename}", 'bounding_box':bbox, 'centroid':(cx,cy)})
                    else: # Homography failed, use contour bounding rectangle
                        x_br, y_br, w_br, h_br = cv2.boundingRect(contour)
                        bbox = [[x_br,y_br],[x_br+w_br,y_br],[x_br+w_br,y_br+h_br],[x_br,y_br+h_br]]
                        moments = cv2.moments(contour)
                        cx = int(moments["m10"]/moments["m00"]) if moments["m00"]!=0 else x_br+w_br//2
                        cy = int(moments["m01"]/moments["m00"]) if moments["m00"]!=0 else y_br+h_br//2
                        potential_detections.append({'score':len(good_matches)/2, 'status_detail':f"Contour BRect (homography failed) with {ref_filename}", 'bounding_box':bbox, 'centroid':(cx,cy)})
                else:
                    status_messages.append(f"  Not enough points for homography with {ref_filename} (need >=4, got {len(src_pts)}).")


    # 3. Fallback to whole image feature matching if no confident contour detections
    if not potential_detections:
        status_messages.append("No confident feature match in contours. Trying whole image.")
        query_kp_whole, query_des_whole = orb.detectAndCompute(query_img_gray, None)

        if query_des_whole is None or len(query_des_whole) < MIN_MATCH_COUNT:
            status_messages.append(f"Not enough features in whole query image ({len(query_des_whole if query_des_whole is not None else [])}).")
        else:
            status_messages.append(f"Whole image ({len(query_kp_whole)} features):")
            best_match_for_whole = {'score': -1, 'data': None}

            for ref_filename, ref_des, ref_kp, ref_shape in REFERENCE_DESCRIPTORS:
                if ref_des is None or not ref_kp: continue
                matches = bf_matcher.knnMatch(query_des_whole, ref_des, k=2)

                if not matches or not matches[0] or len(matches[0]) < 2:
                    # status_messages.append(f"  Skipping {ref_filename} for whole image due to insufficient match candidates.")
                    continue
                good_matches = [m for m,n in matches if m.distance < MATCHER_THRESHOLD * n.distance]

                if len(good_matches) >= MIN_MATCH_COUNT and len(good_matches) > best_match_for_whole['score']:
                    best_match_for_whole['score'] = len(good_matches)
                    best_match_for_whole['data'] = (ref_filename, good_matches, query_kp_whole, ref_kp, ref_shape)

            if best_match_for_whole['data']:
                ref_filename, good_matches, q_kp, r_kp, r_shape = best_match_for_whole['data']
                status_messages.append(f"  Best whole image match: {ref_filename} ({len(good_matches)} matches).")
                src_pts = np.float32([r_kp[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
                dst_pts = np.float32([q_kp[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)

                if len(src_pts) >= 4 and len(dst_pts) >=4:
                    M_homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    if M_homography is not None:
                        h_ref, w_ref = r_shape
                        pts_ref_corners = np.float32([[0,0],[0,h_ref-1],[w_ref-1,h_ref-1],[w_ref-1,0]]).reshape(-1,1,2)
                        dst_corners = cv2.perspectiveTransform(pts_ref_corners, M_homography)
                        bbox = [tuple(pt[0].astype(int)) for pt in dst_corners]
                        moments = cv2.moments(dst_corners.astype(np.int32))
                        cx = int(moments["m10"]/moments["m00"]) if moments["m00"]!=0 else 0
                        cy = int(moments["m01"]/moments["m00"]) if moments["m00"]!=0 else 0
                        potential_detections.append({'score':len(good_matches), 'status_detail':f"Homography match on WHOLE IMAGE with {ref_filename}", 'bounding_box':bbox, 'centroid':(cx,cy)})
                    else: # Homography failed for whole image
                         status_messages.append(f"  Whole image homography failed with {ref_filename}.")
                         # Optional: Fallback to bounding box of matched keypoints if homography fails
                         points_for_bbox = np.float32([ q_kp[m.queryIdx].pt for m in good_matches ])
                         x_br_kp, y_br_kp, w_br_kp, h_br_kp = cv2.boundingRect(points_for_bbox)
                         bbox = [[x_br_kp,y_br_kp],[x_br_kp+w_br_kp,y_br_kp],[x_br_kp+w_br_kp,y_br_kp+h_br_kp],[x_br_kp,y_br_kp+h_br_kp]]
                         cx = x_br_kp + w_br_kp // 2
                         cy = y_br_kp + h_br_kp // 2
                         potential_detections.append({'score':len(good_matches)/2, 'status_detail':f"Keypoint BRect (whole image homography failed) with {ref_filename}", 'bounding_box':bbox, 'centroid':(cx,cy)})
                else:
                     status_messages.append(f"  Not enough points for whole image homography with {ref_filename} (need >=4, got {len(src_pts)}).")


    # 4. Final Decision based on potential_detections
    if potential_detections:
        best_detection = sorted(potential_detections, key=lambda x: x['score'], reverse=True)[0]
        detected_status_final = True
        status_messages.insert(0, best_detection['status_detail']) # Prepend best status
        bounding_box_final = best_detection['bounding_box']
        centroid_final = best_detection['centroid']

        # --- Orientation Detection (Step 8) ---
        if bounding_box_final:
            # For a perspective transform, dst_corners gives 4 points.
            # We need width and height. For a non-rectangular quadrilateral,
            # we can use minAreaRect or approximate from the perspective corners.
            # Simpler: use cv2.boundingRect on the perspective corners for an upright width/height.
            # Or, if it's already a simple [x,y,w,h] style box (from contour fallback), use that directly.

            # Ensure bounding_box_final is a NumPy array of points for minAreaRect
            pts_for_orientation = np.array(bounding_box_final, dtype=np.int32)

            # Get the minimum area rectangle which can be rotated
            rect = cv2.minAreaRect(pts_for_orientation)
            # box_points = cv2.boxPoints(rect) # These are the 4 corners of the minAreaRect
            # box_points = np.int0(box_points)

            # (center (x,y), (width, height), angle of rotation)
            _, (w_rect, h_rect), _ = rect

            if w_rect == 0 or h_rect == 0:
                orientation_final = "degenerate_bbox"
            else:
                # Ensure width is always the larger dimension for consistent aspect ratio
                actual_width = max(w_rect, h_rect)
                actual_height = min(w_rect, h_rect)
                aspect_ratio = actual_width / actual_height

                # Define thresholds (these are heuristics and may need tuning)
                # Assuming "sideways" means the long side is presented (wide aspect ratio)
                # Assuming "head-on" means the short side is presented (narrower, more square/tall aspect ratio)
                SIDEWAYS_THRESHOLD = 1.6  # e.g., if width is > 1.6 times height
                HEAD_ON_THRESHOLD_MAX = 1.3 # e.g., if width is < 1.3 times height (closer to square or tall)
                                          # This also implies height is the dominant visual dimension for "head-on"

                if aspect_ratio > SIDEWAYS_THRESHOLD:
                    orientation_final = "sideways"
                elif aspect_ratio < HEAD_ON_THRESHOLD_MAX: # Closer to square or tall
                    orientation_final = "head-on"
                else:
                    orientation_final = "angled"
            status_messages.append(f"Orientation based on aspect ratio ({aspect_ratio:.2f} from w:{w_rect:.0f},h:{h_rect:.0f}): {orientation_final}")

    elif significant_contours_found : # Green found, but no feature match
        status_messages.append("Significant green regions found, but no feature match to reference bricks.")
    else: # No green, no features
        status_messages.append("No brick detected (no significant green or feature matches).")

    final_status_message = " | ".join(status_messages)
    # Ensure return types are consistent even if no detection
    return detected_status_final, final_status_message, bounding_box_final, centroid_final, orientation_final


if __name__ == '__main__':
    print("Running brick_detector.py directly for testing or initial descriptor generation...")
    if not REFERENCE_DESCRIPTORS:
        print("WARNING: REFERENCE_DESCRIPTORS is empty.")
        globals()['REFERENCE_DESCRIPTORS'] = get_reference_descriptors()
        if not REFERENCE_DESCRIPTORS:
            print("ERROR: Failed to load or compute reference descriptors even from __main__.")
        else:
            print(f"Successfully loaded/computed {len(REFERENCE_DESCRIPTORS)} descriptors from __main__ fallback.")
    else:
        print(f"REFERENCE_DESCRIPTORS already populated with {len(REFERENCE_DESCRIPTORS)} sets.")

    if REFERENCE_DESCRIPTORS and os.path.exists(REFERENCE_IMAGE_DIR):
        image_files = [f for f in os.listdir(REFERENCE_IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if image_files:
            test_query_image_name = image_files[0]
            test_query_image_path = os.path.join(REFERENCE_IMAGE_DIR, test_query_image_name)

            print(f"\n--- Testing detect_brick with: {test_query_image_path} ---")
            detected, status, bbox, centroid, orientation = detect_brick(test_query_image_path)
            print(f"Detection result: {detected}")
            print(f"Status: {status}")
            print(f"Bounding Box: {bbox}")
            print(f"Centroid: {centroid}")
            print(f"Orientation: {orientation}") # Will be 'unknown' for now
            print("--- Test detection finished ---")
        else:
            print("No images found in reference directory to run a test detection.")
    else:
        print("Skipping direct test detection: No reference descriptors or images directory.")

    print("\nbrick_detector.py direct execution finished.")

print("brick_detector.py loaded (globally or by import)")
