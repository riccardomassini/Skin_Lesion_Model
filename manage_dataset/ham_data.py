from manage_dataset.validate_image import is_image_valid
from collections import defaultdict
import pandas as pd
import shutil
import os

"""
Download from Task 3:
- Training Data
- Test Data
- Training Ground Truth
- Test Ground Truth

from here: https://challenge.isic-archive.com/data/#2018
"""

def ham_training_data():
    src_path = "ALL_DATA/"
    dst_path = "TRAIN_DATA/DERM/"
    src_csv = src_path + "ISIC2018_Task3_Training_GroundTruth.csv"
    src_image = src_path + "ISIC2018_Task3_Training_Input/"

    df = pd.read_csv(src_csv)

    class_counts = defaultdict(int)
    image_class_dict = {}

    for _, row in df.iterrows():
        image_name = row['image']
        for col in df.columns[1:]:
            class_value = int(row[col])
            if class_value == 1:
                class_counts[col] += 1
                image_class_dict[image_name] = col
                break

    type_images = defaultdict(list)

    for image_name, image_type in image_class_dict.items():
        type_images[image_type].append(image_name)

    selected_images = {}

    for image_type, images in type_images.items():
        selected_images[image_type] = images[:min(class_counts.values())]

    final_list = []

    for images in selected_images.values():
        final_list.extend(images)

    for el in final_list:
        if el + ".jpg" not in os.listdir(dst_path) and is_image_valid(src_image + el + ".jpg"):
            shutil.copy(src_image + el + ".jpg", dst_path + el + ".jpg")

def ham_test_data():
    src_path = "ALL_DATA/"
    dst_path = "TEST_DATA/DERM/"
    src_csv = src_path + "ISIC2018_Task3_Test_GroundTruth.csv"
    src_image = src_path + "ISIC2018_Task3_Test_Input/"

    df = pd.read_csv(src_csv)

    class_counts = defaultdict(int)
    image_class_dict = {}

    for _, row in df.iterrows():
        image_name = row['image']
        for col in df.columns[1:]:
            class_value = int(row[col])
            if class_value == 1:
                class_counts[col] += 1
                image_class_dict[image_name] = col
                break

    type_images = defaultdict(list)

    for image_name, image_type in image_class_dict.items():
        type_images[image_type].append(image_name)

    selected_images = {}

    for image_type, images in type_images.items():
        selected_images[image_type] = images[:min(class_counts.values())]

    final_list = []

    for images in selected_images.values():
        final_list.extend(images)

    for el in final_list:
        if el + ".jpg" not in os.listdir(dst_path) and is_image_valid(src_image + el + ".jpg"):
            shutil.copy(src_image + el + ".jpg", dst_path + el + ".jpg")