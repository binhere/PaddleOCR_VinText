import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from utils import get_rotate_crop_image

# ===== CONFIG =====
JSON_DIR = "grouped_labels_final" # or `grouped_labels_manual` or `grouped_labels_auto`
REC_ROOT = "rec_textline_data"

SPLITS = {
    "train": "train_images",
    "test": "test_images",
    "unseen": "unseen_images"
}

# Create output folders
os.makedirs(REC_ROOT, exist_ok=True)
for split_folder in SPLITS.values():
    os.makedirs(os.path.join(REC_ROOT, split_folder), exist_ok=True)

# ===== Main loop =====
for fname in tqdm(os.listdir(JSON_DIR), desc="Converting JSON → PaddleOCR rec data"):
    if not fname.endswith(".json"):
        continue

    json_path = os.path.join(JSON_DIR, fname)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    im_path = data["im_path"]
    lines = data["lines"]

    # detect split from path
    if "train_images" in im_path:
        split = "train"
    elif "test_images" in im_path:
        split = "test"
    elif "unseen_images" in im_path:
        split = "unseen"
    else:
        print(f"⚠️ Unknown split for {im_path}, skipping.")
        continue

    label_file = os.path.join(REC_ROOT, f"{split}_labels.txt")
    image_out_dir = os.path.join(REC_ROOT, SPLITS[split])

    img = cv2.imread(im_path)
    if img is None:
        print(f"❌ Failed to read image: {im_path}")
        continue

    base_name = os.path.splitext(os.path.basename(im_path))[0]
    for i, line in enumerate(lines):
        box = np.array(line["line_box"], dtype=np.float32)
        text = line["text"].strip()

        if not text:
            continue

        # auto group word by pretrain may be inaccurate, e.g. group words which are not in the same line
        # we assume that text lines often have width > height, disable rotation to remove inaccurate lines
        # we accept the tradeoff to miss some lines that will be correct after rotation
        # only set `do_rotate=True` if you use `02-2_label_tool.py` *WITHOUT* canvas reference
        # because labeling with canvas reference already results in the correct text orientation
        do_rotate = False
        skip_height_exceeds_width = True if not do_rotate else False
        crop = get_rotate_crop_image(img, box, rotate_90_counter_clockwise=do_rotate)
        if skip_height_exceeds_width:
            crop_h, crop_w = crop.shape[:2]
            if crop_h > crop_w:
                continue

        save_name = f"{base_name}_textline_{i}.jpg"
        save_path = os.path.join(image_out_dir, save_name)
        cv2.imwrite(save_path, crop)

        rel_path = os.path.join(SPLITS[split], save_name).replace("\\", "/")

        with open(label_file, "a", encoding="utf-8") as lf:
            lf.write(f"{rel_path}\t{text}\n")

print("\n✅ All recognition crops and label files generated!")
