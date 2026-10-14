# Alzheimer's Disease Detection Using Deep Learning

AI-based MRI image classification with MobileNetV2 transfer learning, Flask and Grad-CAM.

> **Disclaimer:** research/educational project. Not a medical diagnosis tool.

## Setup (Windows PowerShell, Python 3.10)
```powershell
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset
Place images as `dataset/train|validation|test/<ClassName>/image.jpg`.
Class folder names must be identical in all three splits.
No validation folder? Run `python prepare_validation.py` once.

## Run
```powershell
python train.py       # trains and saves model/alzheimer_model.keras
python evaluate.py    # test metrics, confusion matrix -> outputs/
python app.py         # open http://127.0.0.1:5000
```

## Outputs
- `model/alzheimer_model.keras`, `model/class_names.json`
- `outputs/accuracy_graph.png`, `loss_graph.png`, `confusion_matrix.png`
- `outputs/test_metrics.json`, `classification_report.txt`, `training_history.json`