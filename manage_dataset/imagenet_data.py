from manage_dataset.validate_image import is_image_valid
import shutil
import os


def imagenet_training_data():
    src_path = "ALL_DATA/imagenet-mini/train/"
    dts_path = "TRAIN_DATA/OTHER/"

    img_for_class = 4

    for class_dir in os.listdir(src_path):
        copied_count = 0
        for img in os.listdir(src_path + class_dir):
            if is_image_valid(src_path + class_dir + "/" + img):
                shutil.copy(src_path + class_dir + "/" + img, dts_path + img)
                copied_count += 1

            if copied_count >= img_for_class:
                break


def imagenet_test_data():
    src_path = "ALL_DATA/imagenet-mini/val/"
    dts_path = "TEST_DATA/OTHER/"

    img_for_class = 1

    for class_dir in os.listdir(src_path):
        copied_count = 0
        for img in os.listdir(src_path + class_dir):
            if is_image_valid(src_path + class_dir + "/" + img):
                shutil.copy(src_path + class_dir + "/" + img, dts_path + img)
                copied_count += 1

            if copied_count >= img_for_class:
                break