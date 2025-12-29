import os
import cv2
import numpy as np
from tqdm import tqdm
from utils import parse_vintext_label, get_rotate_crop_image

# Root paths
# ----------------------------
img_root_base = "vietnamese_original"
label_root = os.path.join(img_root_base, "labels")
save_root_base = "rec_word_data"

os.makedirs(save_root_base, exist_ok=True)

# Process each split
# ----------------------------
splits = ["train", "test", "unseen"]

for split in splits:
    img_root = os.path.join(img_root_base, f"{split}_images")
    save_root = os.path.join(save_root_base, f"{split}_images")
    os.makedirs(save_root, exist_ok=True)

    # Label file for this split should be inside ./rec_word_data/
    crop_label_path = os.path.join(save_root_base, f"{split}_labels.txt")
    crop_label = open(crop_label_path, "w", encoding="utf-8")

    img_files = sorted(os.listdir(img_root))
    for idx, img_name in enumerate(tqdm(img_files, desc=f"Processing {split} images")):
        img_path = os.path.join(img_root, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[Warning] Cannot read {img_name}, skipped.")
            continue

        img_id = int(img_name.replace("im", "").replace(".jpg", ""))
        label_name = f"gt_{img_id}.txt"
        label_path = os.path.join(label_root, label_name)

        if not os.path.exists(label_path):
            print(f"[Warning] Missing label: {label_name}")
            continue

        words = parse_vintext_label(label_path)

        for j, word in enumerate(words):
            pts = np.float32(word["bbox"])
            text = word["text"]

            try:
                # Notice:
                # 1. Some words'coordinates are not labeled correctly, so the crop might not be able to read
                # 2. Processing single words where height often exceeds width. Disable auto-rotation to maintain original orientation.
                crop = get_rotate_crop_image(img, pts, rotate_90_counter_clockwise=False)
            except Exception as e:
                print(f"[Error] {img_name} word {j}: {e}")
                continue

            crop_filename = f"{os.path.splitext(img_name)[0]}_word_{j:03d}.jpg"
            crop_save_path = os.path.join(save_root, crop_filename)
            cv2.imwrite(crop_save_path, crop)

            crop_label.write(f"{split}_images/{crop_filename}\t{text}\n")

    crop_label.close()
    print(f"✅ Done! Cropped word data saved to: {save_root}")
    print(f"✅ Label file saved to: {crop_label_path}")