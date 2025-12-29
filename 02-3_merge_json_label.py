# Optional script to merge `grouped_labels_auto` and `grouped_labels_manual`
# if you only create `grouped_labels_auto` skip this script

import os
import shutil

auto_dir   = "grouped_labels_auto"
manual_dir = "grouped_labels_manual"
final_dir  = "grouped_labels_final"

# make sure output folder exists
os.makedirs(final_dir, exist_ok=True)

# step 1: copy all from auto
for fname in os.listdir(auto_dir):
    if fname.endswith(".json"):
        src = os.path.join(auto_dir, fname)
        dst = os.path.join(final_dir, fname)
        shutil.copy2(src, dst)

# step 2: copy all from manual (overwrite if duplicate)
for fname in os.listdir(manual_dir):
    if fname.endswith(".json"):
        src = os.path.join(manual_dir, fname)
        dst = os.path.join(final_dir, fname)
        shutil.copy2(src, dst)
