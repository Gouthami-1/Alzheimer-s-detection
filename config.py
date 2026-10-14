import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------- Dataset paths ----------
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "validation")
TEST_DIR = os.path.join(DATASET_DIR, "test")

# ---------- Model / output paths ----------
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "alzheimer_model.keras")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# ---------- Flask folders ----------
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
RESULTS_FOLDER = os.path.join(BASE_DIR, "static", "results")

# ---------- Image settings (same for training and prediction) ----------
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
BATCH_SIZE = 32          # lower to 16 if you get out-of-memory errors
SEED = 42

# ---------- Training settings ----------
EPOCHS_FROZEN = 10       # phase 1: only the new classifier head is trained
EPOCHS_FINE_TUNE = 10    # phase 2: last layers of MobileNetV2 are fine-tuned
LEARNING_RATE_FROZEN = 1e-3
LEARNING_RATE_FINE_TUNE = 1e-5
FINE_TUNE_LAYERS = 30    # how many of the last MobileNetV2 layers to unfreeze

# ---------- File types ----------
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}          # for web upload
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")  # for counting dataset files

# ---------- Optional: nicer names shown in the web app ----------
# Keys must be EXACTLY your dataset folder names. If a folder name is not
# listed here, the folder name itself is shown. The dictionary can be empty.
DISPLAY_NAMES = {
    "MildDemented": "Mild Demented",
    "ModerateDemented": "Moderate Demented",
    "NonDemented": "Non Demented",
    "VeryMildDemented": "Very Mild Demented",
}


def pretty_name(class_name):
    return DISPLAY_NAMES.get(class_name, class_name)