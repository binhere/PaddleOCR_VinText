# optional script to merge `rec_textline_data` and `rec_word_data`
# skip this script if you only use one of them

import os
import shutil

# === INPUTS ===
src_dirs = ["rec_textline_data", "rec_word_data"] 
dst_dir = "rec_merged_data"                        

# === FOLDER NAMES TO MERGE ===
subfolders = ["train_images", "test_images", "unseen_images"]
txt_files = ["train_labels.txt", "test_labels.txt", "unseen_labels.txt"]

os.makedirs(dst_dir, exist_ok=True)

# --- Merge subfolders ---
for sub in subfolders:
    dst_sub = os.path.join(dst_dir, sub)
    os.makedirs(dst_sub, exist_ok=True)
    
    for src in src_dirs:
        src_sub = os.path.join(src, sub)
        if not os.path.exists(src_sub):
            continue
        
        for fname in os.listdir(src_sub):
            src_path = os.path.join(src_sub, fname)
            dst_path = os.path.join(dst_sub, fname)

            # if same name exists, add prefix to avoid overwriting
            if os.path.exists(dst_path):
                name, ext = os.path.splitext(fname)
                dst_path = os.path.join(dst_sub, f"{name}_{os.path.basename(src)}{ext}")

            shutil.copy2(src_path, dst_path)

# --- Merge .txt files ---
for txt_name in txt_files:
    dst_txt = os.path.join(dst_dir, txt_name)
    
    with open(dst_txt, "w", encoding="utf-8") as fout:
        for src in src_dirs:
            src_txt = os.path.join(src, txt_name)
            if os.path.exists(src_txt):
                with open(src_txt, "r", encoding="utf-8") as fin:
                    fout.write(fin.read().rstrip() + "\n")
