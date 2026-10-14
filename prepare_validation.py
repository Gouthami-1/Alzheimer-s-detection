import os
import random
import shutil

import config

VAL_FRACTION = 0.15
random.seed(config.SEED)


def main():
    if not os.path.isdir(config.TRAIN_DIR):
        raise SystemExit(f"Train folder not found: {config.TRAIN_DIR}")
    if os.path.isdir(config.VAL_DIR) and any(os.scandir(config.VAL_DIR)):
        print("validation folder already has content. Nothing was changed.")
        return

    for class_name in sorted(os.listdir(config.TRAIN_DIR)):
        source = os.path.join(config.TRAIN_DIR, class_name)
        if not os.path.isdir(source):
            continue
        files = [f for f in os.listdir(source) if f.lower().endswith(config.IMAGE_EXTENSIONS)]
        random.shuffle(files)
        n_val = max(1, int(len(files) * VAL_FRACTION))
        destination = os.path.join(config.VAL_DIR, class_name)
        os.makedirs(destination, exist_ok=True)
        for name in files[:n_val]:
            shutil.move(os.path.join(source, name), os.path.join(destination, name))
        print(f"{class_name}: moved {n_val} of {len(files)} images to validation")


if __name__ == "__main__":
    main()