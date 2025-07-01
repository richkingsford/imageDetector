from flask import Flask, request, render_template, redirect, url_for, flash
import os
import cv2 # Used for reading images and drawing
import numpy as np # Used for polylines points
from werkzeug.utils import secure_filename
# Correctly import the new function and global variable names
from brick_detector import detect_brick, get_reference_data, REFERENCE_IMAGE_DIR, REFERENCE_DATA

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Needed for flashing messages

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Ensure reference images folder exists and has images
if not os.path.exists(REFERENCE_IMAGE_DIR):
    os.makedirs(REFERENCE_IMAGE_DIR)
    # You might want to add a message here to tell the user to add reference images
    print(f"Warning: Reference image directory '{REFERENCE_IMAGE_DIR}' was created. Please add your 25 brick photos there.")
elif not any(fname.lower().endswith(('.png', '.jpg', '.jpeg')) for fname in os.listdir(REFERENCE_IMAGE_DIR)):
    print(f"Warning: Reference image directory '{REFERENCE_IMAGE_DIR}' is empty or contains no images. Detection will likely fail.")


# Pre-load reference data on startup
print("Initializing reference data for the web app...")
# Ensure correct function and variable names from brick_detector are used
from brick_detector import REFERENCE_DATA # Changed from REFERENCE_DESCRIPTORS
if not REFERENCE_DATA: # Changed from REFERENCE_DESCRIPTORS
    print("Warning: No reference data loaded. The detector might not work correctly.")


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                file.save(filepath)

                # Basic image validation (e.g., can it be opened by OpenCV?)
                img_check = cv2.imread(filepath)
                if img_check is None:
                    flash('Invalid image file. Could not be processed.', 'error')
                    os.remove(filepath) # Clean up invalid file
                    return redirect(request.url)

                # Perform detection
                detected, status_message, bbox_coords, centroid_coords, orientation = detect_brick(filepath)

                # For debugging
                print(f"Detection for {filename}: Detected={detected}, Status='{status_message}', BBox={bbox_coords}, Centroid={centroid_coords}, Orient='{orientation}'")

                processed_image_filename = filename # Default to original if no drawing
                if detected and bbox_coords:
                    img_to_draw_on = cv2.imread(filepath)
                    if img_to_draw_on is not None:
                        # Draw bounding box (assuming list of 4 (x,y) tuples or points)
                        # cv2.polylines expects an array of points of shape (NumPoints, 1, 2)
                        if len(bbox_coords) == 4 and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in bbox_coords):
                            pts = np.array(bbox_coords, np.int32)
                            pts = pts.reshape((-1, 1, 2))
                            cv2.polylines(img_to_draw_on, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                        elif len(bbox_coords) == 4 and isinstance(bbox_coords[0], int): # for [x,y,w,h] from simple contour
                             x,y,w,h = bbox_coords
                             cv2.rectangle(img_to_draw_on, (x,y), (x+w, y+h), (0,255,0), 2)


                        # Draw centroid
                        if centroid_coords:
                            cv2.circle(img_to_draw_on, (int(centroid_coords[0]), int(centroid_coords[1])), radius=5, color=(0, 0, 255), thickness=-1)

                        # Add orientation text
                        cv2.putText(img_to_draw_on, f"Orientation: {orientation}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

                        # Save the image with drawings
                        name, ext = os.path.splitext(filename)
                        processed_image_filename = f"{name}_processed{ext}"
                        processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], processed_image_filename)
                        cv2.imwrite(processed_filepath, img_to_draw_on)
                        print(f"Saved processed image to: {processed_filepath}")
                    else:
                        print(f"Warning: Could not re-read image {filepath} to draw on.")

                flash_categories = {
                    'detected': 'success' if detected else 'warning',
                    'status': 'info',
                    'bbox': 'info' if bbox_coords else 'secondary',
                    'centroid': 'info' if centroid_coords else 'secondary',
                    'orientation': 'info'
                }

                flash(f'Detection Result: {"Green brick DETECTED!" if detected else "Green brick NOT detected."}', flash_categories['detected'])
                flash(f'Details: {status_message}', flash_categories['status'])
                if bbox_coords:
                    flash(f'Bounding Box: {bbox_coords}', flash_categories['bbox'])
                if centroid_coords:
                    flash(f'Centroid: {centroid_coords}', flash_categories['centroid'])
                flash(f'Orientation: {orientation}', flash_categories['orientation'])

                return redirect(url_for('upload_file', filename=processed_image_filename, original_filename=filename if processed_image_filename != filename else None))

            except Exception as e:
                import traceback
                traceback.print_exc() # Print full traceback to server console
                flash(f'An error occurred during processing: {str(e)}', 'error')
                if os.path.exists(filepath): # Clean up if file was saved before error
                    os.remove(filepath)
                return redirect(request.url)
        else:
            flash('File type not allowed. Please upload JPG, JPEG, or PNG files.', 'error')
            return redirect(request.url)

    # For GET request, or after POST and redirect
    processed_filename = request.args.get('filename') # This will be the _processed filename
    original_filename = request.args.get('original_filename')

    # Determine which image to display: the processed one if available, else original if that's what 'filename' refers to
    display_filename = processed_filename

    return render_template('index.html',
                           uploaded_filename=display_filename)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    # This is a simple way to display the uploaded image if needed,
    # but ensure it's safe and doesn't allow directory traversal.
    # For now, we are not directly displaying it on a separate page from here.
    # If you want to display it, use send_from_directory.
    from flask import send_from_directory
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except FileNotFoundError:
        flash("File not found.", "error")
        return redirect(url_for('upload_file'))


if __name__ == '__main__':
    # Ensure the reference image directory exists
    if not os.path.exists(REFERENCE_IMAGE_DIR):
        os.makedirs(REFERENCE_IMAGE_DIR)
        print(f"Created reference image directory: {REFERENCE_IMAGE_DIR}")
        print("Please place your 25 brick photos in this directory.")

    if len(REFERENCE_DATA) == 0 : # Changed from REFERENCE_DESCRIPTORS
        print("\n\nWARNING: No reference images or data found.")
        print(f"Please ensure your 25 brick photos are in the '{REFERENCE_IMAGE_DIR}' directory.")
        print("The application will run, but detection will not work correctly.\n\n")
    else:
        print(f"\n\nSuccessfully loaded {len(REFERENCE_DATA)} reference image data entries.") # Changed variable and description
        print(f"Reference images should be in: {os.path.abspath(REFERENCE_IMAGE_DIR)}")
        print(f"Uploads will be stored in: {os.path.abspath(UPLOAD_FOLDER)}")
        print(f"Descriptor cache file: {os.path.abspath('reference_descriptors.pkl')}\n\n")


    app.run(debug=True, host='0.0.0.0', port=5001)
print("app.py loaded")
