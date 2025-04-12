from manage_dataset.validate_image import is_image_valid
import shutil
import os

src_train_path = "ALL_DATA/hospital_varese/train/"
src_test_path = "ALL_DATA/hospital_varese/test/"
dst_train_path = "TRAIN_DATA/PAN/"
dst_test_path = "TEST_DATA/PAN/"

def hospital_va_training_data():
    for img in os.listdir(src_train_path):
        if img not in os.listdir(dst_train_path) and is_image_valid(src_train_path + img):
            shutil.copy(src_train_path + img, dst_train_path + img)

def hospital_va_test_data():
    for img in os.listdir(src_test_path):
        if img not in os.listdir(dst_test_path) and is_image_valid(src_test_path + img):
            shutil.copy(src_test_path + img, dst_test_path + img)