# import os
# import json
# import numpy as np
# from PIL import Image, ImageDraw
# import streamlit as st
# from streamlit_drawable_canvas import st_canvas
# from shapely.geometry import Polygon, Point
# from shapely.ops import unary_union
# import cv2
# from PIL import ImageFont
# from PIL import ImageOps
# from utils import parse_vintext_label, group_to_lines, get_rotate_crop_image


# # ---------- Config ----------
# LABEL_DIR = "vietnamese_original/labels"
# IMAGE_ROOTS = {
#     "train": "vietnamese_original/train_images",
#     "test": "vietnamese_original/test_images",
#     "unseen": "vietnamese_original/unseen_images"
# }

# OUTPUT_DIR = "grouped_labels_manual"
# os.makedirs(OUTPUT_DIR, exist_ok=True)


# # ---------- Helpers ----------
# def get_image_and_label(idx):
#     """Find correct folder and paths based on image index (1-based)."""
#     if 1 <= idx <= 1200:
#         folder = IMAGE_ROOTS["train"]
#     elif idx <= 1500:
#         folder = IMAGE_ROOTS["test"]
#     else:
#         folder = IMAGE_ROOTS["unseen"]

#     img_name = f"im{idx:04d}.jpg"      # im0001.jpg → im2000.jpg
#     label_name = f"gt_{idx}.txt"       # gt_1.txt → gt_2000.txt
#     img_path = os.path.join(folder, img_name)
#     label_path = os.path.join(LABEL_DIR, label_name)
#     return folder, img_path, label_path


# # ---------- Streamlit UI ----------
# st.set_page_config(layout="wide")
# st.header("🖋️ Text Line Grouping Tool")

# # Session index
# if "idx" not in st.session_state:
#     st.session_state.idx = 1 # continue from specific image # 166

# folder, img_path, label_path = get_image_and_label(st.session_state.idx)

# if not os.path.exists(img_path):
#     st.error(f"❌ Image not found: {img_path}")
#     st.stop()
# if not os.path.exists(label_path):
#     st.warning(f"⚠️ Label not found: {label_path}")

# image = Image.open(img_path)
# image = ImageOps.exif_transpose(image).convert("RGB")  # ✅ Fix auto-rotation
# orig_w, orig_h = image.size

# # Resize large image to fit Streamlit
# if orig_h > orig_w:
#     MAX_W, MAX_H = 800, 1000
# MAX_W, MAX_H = 1000, 800
# scale = min(MAX_W / orig_w, MAX_H / orig_h, 1.0)
# canvas_w, canvas_h = int(orig_w * scale), int(orig_h * scale)
# resized = image.resize((canvas_w, canvas_h))

# # Draw ground truth boxes on image
# words = parse_vintext_label(label_path)

# # Create overlay as a Pillow image to draw
# overlay = resized.copy()
# draw = ImageDraw.Draw(overlay)

# for w in words:
#     pts = [(x * scale, y * scale) for (x, y) in w["bbox"]]
#     draw.polygon(pts, outline="red", width=2)

# st.write(f"### Image {st.session_state.idx:04d}  ({folder})  |  {len(words)} words")

# # Drawing canvas
# canvas = st_canvas(
#     fill_color="rgba(0, 255, 0, 0.3)",
#     stroke_width=2,
#     stroke_color="lime",
#     background_image=overlay,
#     update_streamlit=True,
#     height=canvas_h,
#     width=canvas_w,
#     drawing_mode="polygon",
#     key=f"canvas_{st.session_state.idx}"
# )

# # --- Controls ---
# col1, col2, col3 = st.columns(3, gap="small", vertical_alignment='center')

# if col1.button("✅ Save Groups"):
#     if canvas.json_data and "objects" in canvas.json_data:
#         polys = []

#         for obj in canvas.json_data["objects"]:
#             if obj.get("type") != "path":
#                 continue

#             path_data = obj.get("path", [])
#             pts = []
#             for seg in path_data:
#                 if isinstance(seg, list) and len(seg) >= 3:
#                     cmd, x, y = seg[:3]
#                     if cmd in ("M", "L"):  # Move or Line commands
#                         pts.append((x / scale, y / scale))

#             if len(pts) >= 3:
#                 polys.append(pts)

#         st.write(f"🧩 Found {len(polys)} polygons")

#         if not polys:
#             st.warning("⚠️ No polygons found — please draw before saving.")
#         else:
#             # === Save grouped texts ===
#             # Load VinText words
#             label_path = os.path.join(LABEL_DIR, f"gt_{st.session_state.idx}.txt")
            
#             words = parse_vintext_label(label_path)
#             grouped_lines = group_to_lines(words, polys, limit_word=False)
            
#             out_data = {
#                 "im_path": img_path.replace("\\", "/"), # normalize path
#                 "lines": grouped_lines
#             }

#             out_path = os.path.join(OUTPUT_DIR, f"grouped_{st.session_state.idx}.json")
#             with open(out_path, "w", encoding="utf-8") as f:
#                 json.dump(out_data, f, ensure_ascii=False, indent=2)
#             st.success(f"✅ Saved {len(grouped_lines)} line groups → {out_path}")
            
#             # === Show Cropped Preview ===
#             st.markdown("### 🔍 Cropped Line Preview")
#             for g in grouped_lines:
#                 box = np.array(g["line_box"], dtype=np.float32)
#                 crop = get_rotate_crop_image(np.array(image), box)
#                 st.image(crop, caption=g["text"], use_container_width=False)
        
#     else:
#         st.warning("⚠️ No canvas data found. Try drawing again.")

# if col2.button("⬅️ Previous Image"):
#     st.session_state.idx -= 1
#     st.rerun()

# if col3.button("➡️ Next Image"):
#     st.session_state.idx += 1
#     st.rerun()

# st.markdown("---")
# st.write("""
# - Red boxes = original OCR word boxes  
# - Draw green polygons to group them into lines  
# - Click **Save Groups** when done → will auto-group words by polygon intersection  
# - Image is resized to fit the window but groups saved in original coordinates
# """)

import os
import re
import json
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from utils import parse_vintext_label, group_to_lines, get_rotate_crop_image


# ---------- Config ----------
LABEL_DIR = "vietnamese_original/labels"
IMAGE_ROOTS = {
    "train": "vietnamese_original/train_images",
    "test": "vietnamese_original/test_images",
    "unseen": "vietnamese_original/unseen_images"
}
OUTPUT_DIR = "grouped_labels_manual"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Display settings
MAX_CANVAS_WIDTH = 1000
MAX_CANVAS_HEIGHT = 800
STARTING_IMAGE_IDX = 1  # Change this to start from a specific image # 160


# ---------- Helpers ----------
def get_image_and_label(idx):
    """Find correct folder and paths based on image index (1-based)."""
    if 1 <= idx <= 1200:
        folder = IMAGE_ROOTS["train"]
    elif idx <= 1500:
        folder = IMAGE_ROOTS["test"]
    else:
        folder = IMAGE_ROOTS["unseen"]

    img_name = f"im{idx:04d}.jpg"
    label_name = f"gt_{idx}.txt"
    img_path = os.path.join(folder, img_name)
    label_path = os.path.join(LABEL_DIR, label_name)
    
    return folder, img_path, label_path


def resize_image_for_canvas(image, max_width, max_height):
    """Resize image to fit canvas while maintaining aspect ratio."""
    orig_w, orig_h = image.size
    scale = min(max_width / orig_w, max_height / orig_h, 1.0)
    canvas_w = int(orig_w * scale)
    canvas_h = int(orig_h * scale)
    resized = image.resize((canvas_w, canvas_h))
    
    return resized, scale, canvas_w, canvas_h


def draw_word_boxes(image, words, scale):
    """Draw ground truth boxes on image."""
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    
    for word in words:
        pts = [(x * scale, y * scale) for (x, y) in word["bbox"]]
        draw.polygon(pts, outline="red", width=2)
    
    return overlay


def extract_polygons_from_canvas(canvas_data, scale):
    """Extract polygon coordinates from canvas JSON data."""
    if not canvas_data or "objects" not in canvas_data:
        return []
    
    polygons = []
    for obj in canvas_data["objects"]:
        if obj.get("type") != "path":
            continue
        
        path_data = obj.get("path", [])
        pts = []
        
        for seg in path_data:
            if isinstance(seg, list) and len(seg) >= 3:
                cmd, x, y = seg[:3]
                if cmd in ("M", "L"):  # Move or Line commands
                    pts.append((x / scale, y / scale))
        
        if len(pts) >= 3:
            polygons.append(pts)
    
    return polygons


def save_grouped_lines(idx, img_path, words, polygons, is_split=False, use_canvas_ref=False):
    """Save grouped text lines to JSON file."""
    
    # we want to group at least 2 words per line
    # for individual words, we can address this by script `02_create_word_data.py`
    min_chars = 26 if is_split == True else None
    grouped_lines = group_to_lines(
        words, 
        polygons, 
        min_words_per_line = 2, 
        min_chars_to_split = min_chars,
        use_extra_ref = use_canvas_ref,
        mode="manual" # always be manual when using this script
    )

    out_data = {
        "im_path": img_path.replace("\\", "/"),
        "lines": grouped_lines
    }
    
    out_path = os.path.join(OUTPUT_DIR, f"grouped_{idx}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
    
    return grouped_lines, out_path


def display_cropped_previews(image, grouped_lines, rotate_90_counter_clockwise=False):
    """Display cropped line previews."""
    st.markdown("### 🔍 Cropped Line Preview")
    for group in grouped_lines:
        box = np.array(group["line_box"], dtype=np.float32)
        crop = get_rotate_crop_image(np.array(image), box, rotate_90_counter_clockwise=rotate_90_counter_clockwise)
        st.image(crop, caption=group["text"], use_container_width=False)
    

def extract_idx_from_path(gt_path: str):
    """Extract integer index from a path like '.../gt_179.txt'."""
    match = re.search(r"gt_(\d+)\.txt", gt_path)
    return int(match.group(1)) if match else None


# ---------- Streamlit UI ----------
st.set_page_config(layout="wide")

# --- initialize session state ---
if "idx" not in st.session_state:
    st.session_state.idx = STARTING_IMAGE_IDX
if "review_idx" not in st.session_state:
    st.session_state.review_idx = 0
if "review_data" not in st.session_state:
    st.session_state.review_data = None
if "load_json_1st" not in st.session_state:
    st.session_state.load_json_1st = False

# --- sidebar ---
with st.sidebar:
    uploaded_file = st.file_uploader(
        "JSON log",
        type=["json"],
        accept_multiple_files=False,
        key="json_uploader"
    )

    # when JSON is uploaded for the first time
    if uploaded_file is not None and not st.session_state.load_json_1st:
        st.session_state.review_data = json.load(uploaded_file)
        review_data = st.session_state.review_data
        st.session_state.review_idx = 0 # Reset to first item
        idx = extract_idx_from_path(review_data[0]['gt_path'])
        st.session_state.idx = idx
        st.session_state.load_json_1st = True   # Mark that we already processed the upload
        st.rerun()                              # Force to refresh everything
        
    # Reset when JSON is removed
    elif uploaded_file is None:
        st.session_state.review_data = None
        st.session_state.load_json_1st = False

    # Always show the number input with current idx value
    input_idx = st.number_input(
        "Image Index",
        min_value=1,
        max_value=2000,
        value=st.session_state.idx,  # Changed from 'min' to current idx
        step=1,
        disabled=True if uploaded_file else False,
        key="idx_input"
    )
    
    # Only update idx if number input changed and no JSON is loaded
    if input_idx != st.session_state.idx and uploaded_file is None:
        st.session_state.idx = input_idx
        st.rerun()

# Sync Before Rendering: get current image/label
if st.session_state.review_data is not None:
    st.session_state.idx = extract_idx_from_path(
        st.session_state.review_data[st.session_state.review_idx]['gt_path']
    )
    
folder, img_path, label_path = get_image_and_label(st.session_state.idx)
    

# Validate paths
if not os.path.exists(img_path):
    st.error(f"❌ Image not found: {img_path}")
    st.stop()
if not os.path.exists(label_path):
    st.warning(f"⚠️ Label not found: {label_path}")

# Load and prepare image
image = Image.open(img_path)
image = ImageOps.exif_transpose(image).convert("RGB")

# Resize image for canvas
resized, scale, canvas_w, canvas_h = resize_image_for_canvas(image, MAX_CANVAS_WIDTH, MAX_CANVAS_HEIGHT)

# Draw ground truth boxes
words = parse_vintext_label(label_path)
overlay = draw_word_boxes(resized, words, scale)

st.write(f"#### Image {st.session_state.idx:04d}  ({folder})  |  {len(words)} words")

# Drawing canvas
canvas = st_canvas(
    fill_color="rgba(0, 255, 0, 0.3)",
    stroke_width=2,
    stroke_color="lime",
    background_image=overlay,
    update_streamlit=True,
    height=canvas_h,
    width=canvas_w,
    drawing_mode="polygon",
    key=f"canvas_{st.session_state.idx}"
)

# --- Controls ---
col1, col2, col3 = st.columns(3, gap="medium", vertical_alignment='center') # toggle function
col4, col5, col6 = st.columns(3, gap="medium", vertical_alignment='center') # button navigation
        
rotate_enabled = col1.toggle(
    label="🔄 90°", 
    value=False, 
    key="rotate_toggle", 
    help="""For preview only: Rotate crops 90° CCW when height > width to check orientation. 
    Coordinates saved unchanged - apply rotation in post-processing script."""
)

split_enabled = col2.toggle(
    label="✂️ Split", 
    value=False, 
    key="split_toggle", 
    help="split text into multiple subsets per text line"
)

use_external_ref = col3.toggle(
    label="🔗 Refer canvas", 
    value=False,
    key="ref_toggle", 
    help="""When enabled, first point of each polygon you draw on canvas as a reference point. 
    If disabled, use first point of first word of sorted of original cordinates as reference.
    Do not support for splitting, only for entire line alignment."""
)

if col4.button("✅ Save", help='save (or override existing) json file'):
    polygons = extract_polygons_from_canvas(canvas.json_data, scale)
    st.write(f"🧩 Found {len(polygons)} polygons")
    
    if not polygons:
        st.warning("⚠️ No polygons found — please draw before saving.")
    else:
        # get split status
        is_split = st.session_state.get("split_toggle", False)
        # get canvas reference status
        use_canvas_ref = st.session_state.get("ref_toggle", False)
        
        # save json files for future use
        grouped_lines, out_path = save_grouped_lines(st.session_state.idx, img_path, words, polygons, is_split, use_canvas_ref)
        st.success(f"✅ Saved {len(grouped_lines)} line groups → {out_path}")
        
        # preview crop rotation, only affects the preview display, not the saved data
        # the point order in saved json files is unchanged, but, we will crop, rotate, save and split them into subsets afterward in `02_create_textline_data.py`  
        rotate_90 = st.session_state.get("rotate_toggle", False)
        display_cropped_previews(image, grouped_lines, rotate_90_counter_clockwise=rotate_90)

if col5.button("⬅️ Back"):
    if st.session_state.review_data is not None:
        st.session_state.review_idx = max(0, st.session_state.review_idx - 1)
        st.session_state.idx = extract_idx_from_path(
            st.session_state.review_data[st.session_state.review_idx]['gt_path']
        )
    else:
        st.session_state.idx = max(1, st.session_state.idx - 1)
    st.rerun()

if col6.button("➡️ Next"):
    if st.session_state.review_data is not None:
        st.session_state.review_idx = min(
            len(st.session_state.review_data) - 1, st.session_state.review_idx + 1
        )
        st.session_state.idx = extract_idx_from_path(
            st.session_state.review_data[st.session_state.review_idx]['gt_path']
        )
    else:
        st.session_state.idx = min(2000, st.session_state.idx + 1)
    st.rerun()

st.markdown("---")
st.write("""
- Red boxes = original OCR word boxes  
- Draw green polygons to group them into lines  
- Click **Save** when done
""")