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

def _keypoints_to_tuples(keypoints):
    if keypoints is None:
        return []
    return [(kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in keypoints]

def _tuples_to_keypoints(kp_tuples):
    if kp_tuples is None:
        return []
    return [cv2.KeyPoint(x=t[0][0], y=t[0][1], _size=t[1], _angle=t[2],
                         _response=t[3], _octave=t[4], _class_id=t[5]) for t in kp_tuples]

def get_reference_descriptors():
    """
    Loads or computes and saves keypoints (in a pickleable format), descriptors,
    and shapes for all reference brick images.
    The saved data in .pkl file is a list of tuples:
    (filename, descriptors, list_of_keypoint_tuples, shape).
    The function returns a list of tuples with live KeyPoint objects:
    (filename, descriptors, list_of_cv2.KeyPoint_objects, shape).
    Shape is (height, width).
    """
    if os.path.exists(DESCRIPTORS_FILE):
        try:
            with open(DESCRIPTORS_FILE, 'rb') as f:
                pickled_data = pickle.load(f)

            # Validate structure: list of (filename, descriptors, list_of_kp_tuples, shape)
            if isinstance(pickled_data, list) and pickled_data and \
               isinstance(pickled_data[0], tuple) and len(pickled_data[0]) == 4 and \
               isinstance(pickled_data[0][1], np.ndarray) and \
               isinstance(pickled_data[0][2], list) and \
               (not pickled_data[0][2] or (isinstance(pickled_data[0][2][0], tuple) and len(pickled_data[0][2][0]) == 6)) and \
               isinstance(pickled_data[0][3], tuple) and len(pickled_data[0][3]) == 2:

                reconstructed_data_for_runtime = []
                for filename, des, kp_tuples, shape_val in pickled_data:
                    reconstructed_kps = _tuples_to_keypoints(kp_tuples)
                    reconstructed_data_for_runtime.append((filename, des, reconstructed_kps, shape_val))
                print("Loaded and reconstructed reference data (des, kp, shape) from cache.")
                return reconstructed_data_for_runtime
            else:
                print("Cached descriptors file is in an old/invalid format. Recomputing.")
        except Exception as e:
            print(f"Error loading or validating cached descriptors: {e}. Recomputing.")

    print("Computing reference descriptors, keypoints (pickleable), and shapes...")
    reference_data_to_pickle = [] # This will store data with keypoints as tuples

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

        keypoints, descriptors = orb.detectAndCompute(img, None) # These are live KeyPoint objects
        print(f"  ORB feature computation done for {filename}.")

        if descriptors is not None and len(descriptors) > 0:
            img_shape = img.shape[:2]
            pickleable_kps = _keypoints_to_tuples(keypoints) # Convert live KPs to tuples for pickling
            reference_data_to_pickle.append((filename, descriptors, pickleable_kps, img_shape))
            print(f"  Found {len(descriptors)} descriptors and {len(keypoints)} keypoints for {filename}. Shape: {img_shape}")
        else:
            print(f"  Warning: No descriptors found for reference image {filename}.")

    print(f"Finished processing all reference images. Total data entries collected: {len(reference_data_to_pickle)}.")
    if not reference_data_to_pickle:
        print("No valid reference data was collected.")
        return [] # Return empty list if nothing was processed

    print(f"Attempting to save {len(reference_data_to_pickle)} reference data entries to {DESCRIPTORS_FILE}...")
    with open(DESCRIPTORS_FILE, 'wb') as f:
        pickle.dump(reference_data_to_pickle, f) # Save the pickleable version
    print(f"Saved {len(reference_data_to_pickle)} reference data entries to {DESCRIPTORS_FILE}.")

    # For immediate runtime use after computation, convert the just-computed pickleable KPs back to live KPs.
    reconstructed_data_for_runtime = []
    for fname, des, kp_tuples, shape_val in reference_data_to_pickle:
        reconstructed_kps = _tuples_to_keypoints(kp_tuples)
        reconstructed_data_for_runtime.append((fname, des, reconstructed_kps, shape_val))
    return reconstructed_data_for_runtime


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
    if not REFERENCE_DESCRIPTORS: # This now contains live KeyPoint objects
        return False, "Error: No reference data loaded.", None, None, "unknown"

    query_img_bgr = cv2.imread(query_image_path)
    if query_img_bgr is None:
        return False, f"Error: Could not read query image: {query_image_path}", None, None, "unknown"

    query_img_gray = cv2.cvtColor(query_img_bgr, cv2.COLOR_BGR2GRAY)
    query_img_hsv = cv2.cvtColor(query_img_bgr, cv2.COLOR_BGR2HSV)
    # query_img_height, query_img_width = query_img_gray.shape[:2] # Not directly used yet

    bounding_box_final = None
    centroid_final = None
    orientation_final = "unknown"
    detected_status_final = False
    status_messages = []

    potential_detections = []

    # 1. Color Segmentation
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

            status_messages.append(f"Contour {contour_idx} (Area {cv2.contourArea(contour):.0f}, {len(query_kp_contour)} features):")

            best_match_for_this_contour = {'score': -1, 'data': None}

            for ref_filename, ref_des, ref_kp_live, ref_shape in REFERENCE_DESCRIPTORS: # ref_kp_live are KeyPoint objects
                if ref_des is None or not ref_kp_live : continue
                matches = bf_matcher.knnMatch(query_des_contour, ref_des, k=2)

                if not matches or not matches[0] or len(matches[0]) < 2:
                    continue
                good_matches = [m for m, n in matches if m.distance < MATCHER_THRESHOLD * n.distance]

                if len(good_matches) >= MIN_MATCH_COUNT and len(good_matches) > best_match_for_this_contour['score']:
                    best_match_for_this_contour['score'] = len(good_matches)
                    best_match_for_this_contour['data'] = (ref_filename, good_matches, query_kp_contour, ref_kp_live, ref_shape)

            if best_match_for_this_contour['data']:
                ref_filename, good_matches, q_kp, r_kp_live, r_shape = best_match_for_this_contour['data']
                status_messages.append(f"  Best match in contour: {ref_filename} ({len(good_matches)} matches).")

                src_pts = np.float32([r_kp_live[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([q_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

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
                        potential_detections.append({'score':len(good_matches), 'status_detail':f"Homography match in contour with {ref_filename}", 'bounding_box':bbox, 'centroid':(cx,cy)})
                    else:
                        x_br, y_br, w_br, h_br = cv2.boundingRect(contour)
                        bbox = [[x_br,y_br],[x_br+w_br,y_br],[x_br+w_br,y_br+h_br],[x_br,y_br+h_br]]
                        moments = cv2.moments(contour)
                        cx = int(moments["m10"]/moments["m00"]) if moments["m00"]!=0 else x_br+w_br//2
                        cy = int(moments["m01"]/moments["m00"]) if moments["m00"]!=0 else y_br+h_br//2
                        potential_detections.append({'score':len(good_matches)/2, 'status_detail':f"Contour BRect (homography failed) with {ref_filename}", 'bounding_box':bbox, 'centroid':(cx,cy)})
                else:
                    status_messages.append(f"  Not enough points for homography with {ref_filename} (need >=4, got {len(src_pts)}).")

    if not potential_detections:
        status_messages.append("No confident feature match in contours. Trying whole image.")
        query_kp_whole, query_des_whole = orb.detectAndCompute(query_img_gray, None)

        if query_des_whole is None or len(query_des_whole) < MIN_MATCH_COUNT :
            status_messages.append(f"Not enough features in whole query image ({len(query_des_whole if query_des_whole is not None else [])}).")
        else:
            status_messages.append(f"Whole image ({len(query_kp_whole)} features):")
            best_match_for_whole = {'score': -1, 'data': None}

            for ref_filename, ref_des, ref_kp_live, ref_shape in REFERENCE_DESCRIPTORS: # ref_kp_live are KeyPoint objects
                if ref_des is None or not ref_kp_live: continue
                matches = bf_matcher.knnMatch(query_des_whole, ref_des, k=2)

                if not matches or not matches[0] or len(matches[0]) < 2:
                    continue
                good_matches = [m for m,n in matches if m.distance < MATCHER_THRESHOLD * n.distance]

                if len(good_matches) >= MIN_MATCH_COUNT and len(good_matches) > best_match_for_whole['score']:
                    best_match_for_whole['score'] = len(good_matches)
                    best_match_for_whole['data'] = (ref_filename, good_matches, query_kp_whole, ref_kp_live, ref_shape)

            if best_match_for_whole['data']:
                ref_filename, good_matches, q_kp, r_kp_live, r_shape = best_match_for_whole['data']
                status_messages.append(f"  Best whole image match: {ref_filename} ({len(good_matches)} matches).")
                src_pts = np.float32([r_kp_live[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
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
                    else:
                         status_messages.append(f"  Whole image homography failed with {ref_filename}.")
                         points_for_bbox = np.float32([ q_kp[m.queryIdx].pt for m in good_matches ])
                         x_br_kp, y_br_kp, w_br_kp, h_br_kp = cv2.boundingRect(points_for_bbox)
                         bbox = [[x_br_kp,y_br_kp],[x_br_kp+w_br_kp,y_br_kp],[x_br_kp+w_br_kp,y_br_kp+h_br_kp],[x_br_kp,y_br_kp+h_br_kp]]
                         cx = x_br_kp + w_br_kp // 2
                         cy = y_br_kp + h_br_kp // 2
                         potential_detections.append({'score':len(good_matches)/2, 'status_detail':f"Keypoint BRect (whole image homography failed) with {ref_filename}", 'bounding_box':bbox, 'centroid':(cx,cy)})
                else:
                     status_messages.append(f"  Not enough points for whole image homography with {ref_filename} (need >=4, got {len(src_pts)}).")

    if potential_detections:
        best_detection = sorted(potential_detections, key=lambda x: x['score'], reverse=True)[0]
        detected_status_final = True
        status_messages.insert(0, best_detection['status_detail'])
        bounding_box_final = best_detection['bounding_box']
        centroid_final = best_detection['centroid']

        if bounding_box_final:
            pts_for_orientation = np.array(bounding_box_final, dtype=np.int32)
            rect = cv2.minAreaRect(pts_for_orientation)
            _, (w_rect, h_rect), _ = rect

            if w_rect == 0 or h_rect == 0:
                orientation_final = "degenerate_bbox"
            else:
                actual_width = max(w_rect, h_rect)
                actual_height = min(w_rect, h_rect)
                aspect_ratio = actual_width / actual_height if actual_height > 0 else float('inf') # Avoid division by zero

                SIDEWAYS_THRESHOLD = 1.6
                HEAD_ON_THRESHOLD_MAX = 1.3

                if aspect_ratio > SIDEWAYS_THRESHOLD:
                    orientation_final = "sideways"
                elif aspect_ratio < HEAD_ON_THRESHOLD_MAX:
                    orientation_final = "head-on"
                else:
                    orientation_final = "angled"
            status_messages.append(f"Orientation ({aspect_ratio:.2f} from w:{w_rect:.0f},h:{h_rect:.0f}): {orientation_final}")

    elif significant_contours_found :
        status_messages.append("Significant green regions found, but no feature match to reference bricks.")
    else:
        status_messages.append("No brick detected (no significant green or feature matches).")

    final_status_message = " | ".join(status_messages)
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
            # Test with a known brick image
            test_query_image_name = image_files[0]
            test_query_image_path = os.path.join(REFERENCE_IMAGE_DIR, test_query_image_name)
            print(f"\n--- Testing detect_brick with: {test_query_image_path} ---")
            detected, status, bbox, centroid, orientation = detect_brick(test_query_image_path)
            print(f"Detection result: {detected}")
            print(f"Status: {status}")
            print(f"Bounding Box: {bbox}")
            print(f"Centroid: {centroid}")
            print(f"Orientation: {orientation}")
            print("--- Test detection finished ---")

            # Test with a dummy non-brick image (e.g., black image)
            dummy_black_img_path = "dummy_black_test.jpg"
            cv2.imwrite(dummy_black_img_path, np.zeros((100,100,3), dtype=np.uint8))
            print(f"\n--- Testing detect_brick with: {dummy_black_img_path} ---")
            detected, status, bbox, centroid, orientation = detect_brick(dummy_black_img_path)
            print(f"Detection result: {detected}")
            print(f"Status: {status}")
            print(f"Bounding Box: {bbox}")
            print(f"Centroid: {centroid}")
            print(f"Orientation: {orientation}")
            print("--- Test detection finished ---")
            os.remove(dummy_black_img_path)
        else:
            print("No images found in reference directory to run a test detection.")
    else:
        print("Skipping direct test detection: No reference descriptors or images directory.")

    print("\nbrick_detector.py direct execution finished.")

print("brick_detector.py loaded (globally or by import)")
