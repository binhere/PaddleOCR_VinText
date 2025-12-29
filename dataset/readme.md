# Dataset preparation

1. `01_create_word_data.py`- convert single word dataset from Vintext style to Paddle style

    input:
    * original Vintext dataset

    output:
    * `rec_word_data` dataset

2. `02-1_group_words_by_pretrain.py`- automatically group words to text lines by using pretrained model from PaddleOCR -> accelerate textline data preparation

    input:
    * original Vintext dataset

    output:
    * `grouped_labels_auto` folder containing JSON files <br>
    * `log_keep.json` - optional output to review/re-annotate manually in step 3

3. `02-2_label_tool.py` - optional, give you full control to create your own dataset mannually

    input:
    * original Vintext dataset

    output:
    * `grouped_labels_manual` folder containing JSON files 

4. `02-3_merge_json_label.py` - optional, help you merge `grouped_labels_auto` and `grouped_labels_manual`. if you only use script `02-1` to create `grouped_labels_auto` skip this step

    input:
    * 2 JSON folders: `grouped_labels_auto` and `grouped_labels_manual`
    
    output:
    * `grouped_labels_final` folder - merged version from auto and manual text line folders

5. `02-4_create_textline_data.py` - convert JSON files to Paddle Text Recognition dataset

    input:
    * `grouped_labels_final` folder (or `grouped_labels_manual` or `grouped_labels_auto`)

    output:
    * `rec_textline_data` dataset

6. `03_merge_data.py` - optional, this help you merge `rec_textline_data` and `rec_word_data`. Skip this step if you only use one of them

    input:
    * 2 datasets: `rec_textline_data` and `rec_word_data`

    output:
    * `rec_merged_data` dataset

7. `data_insight.ipynb` - optional, explore dataset

    output:
    * `dataset_analysis.csv`


# Tool usage

streamlit run 02-2_label_tool.py