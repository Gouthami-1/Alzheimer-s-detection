import os
import time
import uuid
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from PIL import Image, UnidentifiedImageError

import config
import predict

app = Flask(__name__)
app.config["SECRET_KEY"] = "change-this-secret-key"       # needed for flash messages
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024        # max upload size: 5 MB

os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(config.RESULTS_FOLDER, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS


def is_valid_image(path):
    """Check the file really is an image (not just renamed to .jpg)."""
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, OSError):
        return False


def model_files_exist():
    return os.path.exists(config.MODEL_PATH) and os.path.exists(config.CLASS_NAMES_PATH)


def cleanup_old_files(folder, max_age_seconds=3600):
    """Uploads are temporary: delete files older than one hour."""
    now = time.time()
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if os.path.isfile(path) and now - os.path.getmtime(path) > max_age_seconds:
            try:
                os.remove(path)
            except OSError:
                pass


def remove_file(path):
    try:
        os.remove(path)
    except OSError:
        pass


@app.route("/")
def index():
    return render_template("index.html", model_ready=model_files_exist())


@app.route("/predict", methods=["POST"])
def upload_and_predict():
    if "mri_image" not in request.files:
        flash("No file was uploaded.", "danger")
        return redirect(url_for("index"))

    file = request.files["mri_image"]
    if file.filename == "":
        flash("Please choose an MRI image first.", "danger")
        return redirect(url_for("index"))

    original_name = secure_filename(file.filename)
    if not allowed_file(original_name):
        flash("Invalid file type. Please upload a JPG, JPEG or PNG image.", "danger")
        return redirect(url_for("index"))

    cleanup_old_files(config.UPLOAD_FOLDER)
    cleanup_old_files(config.RESULTS_FOLDER)

    filename = f"{uuid.uuid4().hex}_{original_name}"
    save_path = os.path.join(config.UPLOAD_FOLDER, filename)
    file.save(save_path)

    if not is_valid_image(save_path):
        remove_file(save_path)
        flash("The uploaded file is not a valid image.", "danger")
        return redirect(url_for("index"))

    try:
        result = predict.predict_image(save_path)
    except FileNotFoundError as error:
        remove_file(save_path)
        flash(str(error), "danger")
        return redirect(url_for("index"))
    except Exception:
        app.logger.exception("Prediction failed")
        remove_file(save_path)
        flash("Something went wrong while processing the image. Please try another image.", "danger")
        return redirect(url_for("index"))

    gradcam_image = None
    try:
        gradcam_name = f"{os.path.splitext(filename)[0]}_gradcam.png"
        predict.generate_gradcam(
            save_path, os.path.join(config.RESULTS_FOLDER, gradcam_name), result["class_index"])
        gradcam_image = f"results/{gradcam_name}"
    except Exception:
        app.logger.exception("Grad-CAM failed (prediction is still shown)")

    return render_template(
        "result.html",
        uploaded_image=f"uploads/{filename}",
        gradcam_image=gradcam_image,
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
    )


@app.errorhandler(413)
def file_too_large(error):
    flash("File is too large. Maximum allowed size is 5 MB.", "danger")
    return redirect(url_for("index"))


if __name__ == "__main__":
    try:
        predict.load_resources()
        print("Model loaded successfully.")
    except Exception as error:
        print(f"WARNING: model not loaded -> {error}")
    # use_reloader=False prevents TensorFlow from loading the model twice
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)