from flask import Flask, request, render_template, redirect, url_for, flash
import os
import cv2 # For image size checking, not heavy processing here
from werkzeug.utils import secure_filename
from brick_detector import detect_brick, get_reference_descriptors, REFERENCE_IMAGE_DIR

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


# Pre-load reference descriptors on startup
print("Initializing reference descriptors for the web app...")
REFERENCE_DESCRIPTORS = get_reference_descriptors()
if not REFERENCE_DESCRIPTORS:
    print("Warning: No reference descriptors loaded. The detector might not work correctly.")
    # Depending on desired behavior, you could prevent app startup or flash a persistent warning.


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
                detected, status_message = detect_brick(filepath)

                # For debugging, let's see the status message
                print(f"Detection status for {filename}: {status_message}")

                if detected:
                    flash('Green brick DETECTED!', 'success')
                    flash(f'Details: {status_message}', 'info')
                else:
                    flash('Green brick NOT detected.', 'warning')
                    flash(f'Details: {status_message}', 'info')

                # Optionally, remove the uploaded file after processing to save space
                # os.remove(filepath)
                # For now, keep it for easier debugging if needed

                return redirect(url_for('upload_file', filename=filename)) # Or redirect to a result page

            except Exception as e:
                flash(f'An error occurred during processing: {str(e)}', 'error')
                if os.path.exists(filepath): # Clean up if file was saved before error
                    os.remove(filepath)
                return redirect(request.url)
        else:
            flash('File type not allowed. Please upload JPG, JPEG, or PNG files.', 'error')
            return redirect(request.url)

    # For GET request, or after POST and redirect
    uploaded_filename = request.args.get('filename')
    return render_template('index.html', uploaded_filename=uploaded_filename)


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

    if len(REFERENCE_DESCRIPTORS) == 0 :
        print("\n\nWARNING: No reference images or descriptors found.")
        print(f"Please ensure your 25 brick photos are in the '{REFERENCE_IMAGE_DIR}' directory.")
        print("The application will run, but detection will not work correctly.\n\n")
    else:
        print(f"\n\nSuccessfully loaded {len(REFERENCE_DESCRIPTORS)} reference image descriptors.")
        print(f"Reference images should be in: {os.path.abspath(REFERENCE_IMAGE_DIR)}")
        print(f"Uploads will be stored in: {os.path.abspath(UPLOAD_FOLDER)}")
        print(f"Descriptor cache file: {os.path.abspath('reference_descriptors.pkl')}\n\n")


    app.run(debug=True, host='0.0.0.0', port=5001)
print("app.py loaded")
