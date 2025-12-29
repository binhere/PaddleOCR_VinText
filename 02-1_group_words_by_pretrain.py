import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from paddleocr import TextDetection
from utils import parse_vintext_label, group_to_lines

detector = TextDetection(model_name="PP-OCRv5_mobile_det")

LABEL_DIR = "vietnamese_original/labels"
IMAGE_ROOTS = {
    "train": "vietnamese_original/train_images",
    "test": "vietnamese_original/test_images",
    "unseen": "vietnamese_original/unseen_images"
}

OUTPUT_DIR = "grouped_labels_auto"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# remove existing `log.json`
log_data = []
log_path = "log.json"
if os.path.exists(log_path):
	os.remove(log_path)

start = 176 # start from specific image that you want to continue 
end = 2000

for idx in tqdm(range(start, end+1), desc="Processing"):
    
	if 1<= idx <= 1200:
		im_path = os.path.join(IMAGE_ROOTS["train"], f"im{idx:04d}.jpg")
	elif idx <= 1500:
		im_path = os.path.join(IMAGE_ROOTS["test"], f"im{idx:04d}.jpg")
	else:
		im_path = os.path.join(IMAGE_ROOTS["unseen"], f"im{idx:04d}.jpg")

	gt_path = f"vietnamese_original/labels/gt_{idx}.txt"

	output = detector.predict(im_path)
	dt_polys = output[0]['dt_polys']
	paddle_lines = dt_polys

	word_boxes = parse_vintext_label(gt_path)
	line_groups = group_to_lines(
    	word_boxes, 
        paddle_lines, 
        use_extra_ref=False, # do *NOT* use predicted polygon of PaddleOCR TextDetection as reference
        min_words_per_line=2, 
        min_chars_to_split=26,
		mode="auto" # always be `auto` when using this script with pretrained PaddleOCR TextDetection
	) 
 
	out_data = {
		"im_path": im_path.replace("\\", "/"), # normalize path
		"lines": line_groups
	}
 
	### log images which have number of returned lines < 2 to review afterward (optional, comment out if not needed or can customize)
	# num_returned_lines = 2
	# if len(line_groups) <= num_returned_lines:
	# 	log_data.append(
	# 		{
	# 			"im_path": im_path.replace("\\", "/"), 
	# 			"gt_path": gt_path.replace("\\", "/"),
	# 			"num_lines": len(line_groups)
	# 		}
	# 	)
	# with open(log_path, "w", encoding="utf-8") as f:
	# 	json.dump(log_data, f, ensure_ascii=False, indent=2)

	out_path = os.path.join(OUTPUT_DIR, f"grouped_{idx}.json")
	with open(out_path, "w", encoding="utf-8") as f:
		json.dump(out_data, f, ensure_ascii=False, indent=2)
  
	