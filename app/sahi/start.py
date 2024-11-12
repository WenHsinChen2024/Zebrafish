from sahi.predict import predict
from sahi.predict import predict_spine
from sahi.predict import predict_fish
from sahi.scripts.coco_evaluation import evaluate
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image_as_pil
import csv
import os
import shutil
import subprocess

csv_filename = "./outfile/fishscore.csv"
if os.path.exists(csv_filename):
    with open(csv_filename, "w", newline="") as file:
        file.truncate(0)  # 清空檔案

file_to_delete = "./outfile/autolabel_fish"
if os.path.exists(file_to_delete):
    shutil.rmtree(file_to_delete)

file_to_delete = "./outfile/autolabel_spine"
if os.path.exists(file_to_delete):
    shutil.rmtree(file_to_delete)

file_to_delete = "./outfile/our_parts"
if os.path.exists(file_to_delete):
    shutil.rmtree(file_to_delete)
    #print(f"檔案 {file_to_delete} 已刪除")
file_to_delete = "./outfile/our_spine"
if os.path.exists(file_to_delete):
    shutil.rmtree(file_to_delete)
    #print(f"檔案 {file_to_delete} 已刪除")
folder_to_delete = "./run"
if os.path.exists(folder_to_delete):
    shutil.rmtree(folder_to_delete)
    
x=predict_fish(
    detection_model= None,
    model_type = "mmdet",
    model_path = "../mmdetection_CJHo/work_dirs/paper_our_fish_f217-232/best.pth",
    model_config_path = "../mmdetection_CJHo/work_dirs/paper_our_fish_f217-232/paper_our_fish_f217-232.py",
    model_confidence_threshold = 0.2,
    model_device = "gpu",
    model_category_mapping = {"0": "Abdomen","1": "Back","2": "Head","3": "Spine"},
    model_category_remapping = None,
    source = "./fishimage",#待檢測的資料集位置
    no_standard_prediction = True,
    no_sliced_prediction = False,
    image_size= None,
    slice_height = 4000,
    slice_width = 4000,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    postprocess_type="GREEDYNMM",
    postprocess_match_metric="IOS",
    postprocess_match_threshold = 0.2,
    postprocess_class_agnostic = False,
    novisual = False,
    view_video = False,
    frame_skip_interval = 0,
    export_pickle = False,
    export_crop = True,
    dataset_json_path = "",
    project = "runs",
    name = "our_parts",
    visual_bbox_thickness = 1,
    visual_text_size = 1,
    visual_text_thickness = 1,
    visual_export_format = "jpg",
    verbose = 1,
    return_dict = True,
    force_postprocess_type = False,
    )

subprocess.run(["python", "coco2labelme_fish.pyc"])
print("coco2labelme fish Finish!")

if os.path.exists('./fishimage'):
    if not os.path.exists('./outfile/autolabel_fish'):
        os.makedirs('./outfile/autolabel_fish')
    
    for filename in os.listdir('./fishimage'):
        file_path = os.path.join('./fishimage', filename)
        if os.path.isfile(file_path):
            shutil.move(file_path, os.path.join('./outfile/autolabel_fish', filename))



x=predict_spine(
    detection_model= None,
    model_type = "mmdet",
    model_path = "../mmdetection_CJHo/work_dirs/paper_our_marspine_217-232/best.pth",
    model_config_path = "../mmdetection_CJHo/work_dirs/paper_our_marspine_217-232/paper_our_marspine_217-232.py",
    model_confidence_threshold = 0.55,
    model_device = "gpu",
    model_category_mapping = {"0": "a","1": "b","2": "r"},
    model_category_remapping = None,
    source = "./runs/our_parts/crops_spine_image",
    no_standard_prediction = True,
    no_sliced_prediction = False,
    image_size= None,
    slice_height = 512,
    slice_width = 512,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    postprocess_type="GREEDYNMM",
    postprocess_match_metric="IOS",
    postprocess_match_threshold = 0.2,
    postprocess_class_agnostic = True,
    novisual = False,
    view_video = False,
    frame_skip_interval = 0,
    export_pickle = False,
    export_crop = False,
    dataset_json_path = "",
    project = "runs",
    name = "our_spine",
    visual_bbox_thickness = 2,
    visual_text_size = 1,
    visual_text_thickness = 2,
    visual_export_format = "jpg",
    verbose = 1,
    return_dict = True,
    force_postprocess_type = False,
    )

subprocess.run(["python", "coco2labelme_spine.pyc"])
print("coco2labelme spine Finish!")

if os.path.exists('./runs/our_parts/crops_spine_image'):
    if not os.path.exists('./outfile/autolabel_spine'):
        os.makedirs('./outfile/autolabel_spine')
    
    for filename in os.listdir('./runs/our_parts/crops_spine_image'):
        file_path = os.path.join('./runs/our_parts/crops_spine_image', filename)
        if os.path.isfile(file_path):
            shutil.move(file_path, os.path.join('./outfile/autolabel_spine', filename))

if os.path.exists('./fishimage'):
    if not os.path.exists('./outfile/autolabel_spine'):
        os.makedirs('./outfile/autolabel_spine')
    
    for filename in os.listdir('./fishimage'):
        file_path = os.path.join('./fishimage', filename)
        if os.path.isfile(file_path):
            shutil.move(file_path, os.path.join('./outfile/autolabel_spine', filename))

#move files under /runs to /outfile
            
source_folder = './runs'
destination_folder = './outfile'

items = os.listdir(source_folder)
#print(items)

for item in items:
    try:
        item_path = os.path.join(source_folder, item)
        if os.path.isdir(item_path):
            shutil.move(item_path, os.path.join(destination_folder, item))
        else:
            shutil.move(item_path, destination_folder)
    except:
        print('Failed to move the file.')

spine_folder = './outfile/autolabel_spine'
fish_folder = './outfile/autolabel_fish'

# 获取autolabel_spine和autolabel_fish文件夹中的jpg文件列表
spine_files = [file[:-4] for file in os.listdir(spine_folder) if file.endswith('.jpg')]
print(spine_files)
for i in range(len(spine_files)):
    if spine_files[i].endswith('_spine'):
        spine_files[i] = spine_files[i].replace('_spine', '')
        print(spine_files[i])
spine_files = set(spine_files)
fish_files = set([file[:-4] for file in os.listdir(fish_folder) if file.endswith('.jpg')])
print(spine_files)
print(fish_files)
# 找到autolabel_fish中有但autolabel_spine中没有的文件名
missing_fish_files = fish_files - spine_files

# 写入fishscore.csv文件
with open('./outfile/fishscore.csv', 'a+') as csv_file:
    for file in missing_fish_files:
        csv_file.write('\n'+'name,'+file+',error' + '\n')

# 删除fishscore.csv中的空行
with open('./outfile/fishscore.csv', 'r') as csv_file:
    lines = csv_file.readlines()

# 重新写入文件，跳过空行
with open('./outfile/fishscore.csv', 'w') as csv_file:
    csv_file.writelines([line for line in lines if line.strip()])


print("Zebrafish scoring completed!")
