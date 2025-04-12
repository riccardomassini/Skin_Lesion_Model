from manage_dataset.validate_image import is_image_valid
import shutil
import os


def padufes_training_data():
    derm_clinic_path = "TRAIN_DATA/DERM_CLINIC/"
    derm_clinic_padufes_1 = "ALL_DATA/imgs_part_1/"
    derm_clinic_padufes_2 = "ALL_DATA/imgs_part_2/"
    derm_clinic_padufes_3 = "ALL_DATA/imgs_part_3/"

    for img in os.listdir(derm_clinic_padufes_1):
        if img not in os.listdir(derm_clinic_path) and is_image_valid(derm_clinic_padufes_1 + img):
            shutil.copy(derm_clinic_padufes_1 + img, derm_clinic_path + img)

    for img in os.listdir(derm_clinic_padufes_2):
        if img not in os.listdir(derm_clinic_path) and is_image_valid(derm_clinic_padufes_2 + img):
            shutil.copy(derm_clinic_padufes_2 + img, derm_clinic_path + img)

    for img in os.listdir(derm_clinic_padufes_3):
        if img not in os.listdir(derm_clinic_path) and is_image_valid(derm_clinic_padufes_3 + img):
            shutil.copy(derm_clinic_padufes_3 + img, derm_clinic_path + img)