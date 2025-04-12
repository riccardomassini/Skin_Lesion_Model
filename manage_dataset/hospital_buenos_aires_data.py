from manage_dataset.validate_image import is_image_valid
import pandas as pd
import shutil
import os


def hospital_ba_test_data():
    src_path = "ALL_DATA/hospital_buenos_aires/"
    path_image = src_path + "images/"
    path_csv = src_path + "metadata.csv"
    dst_path = "TEST_DATA/DERM_CLINIC/"

    df = pd.read_csv(path_csv)

    image_dict = df.groupby("image_type")["isic_id"].apply(list).to_dict()

    new_keys = {
        list(image_dict.keys())[0]: "clinic_close",
        list(image_dict.keys())[1]: "clinic_overview",
        list(image_dict.keys())[2]: "dermoscopic"
    }

    image_dict = {new_keys.get(k): v for k, v in image_dict.items()}

    clinical_close_images = [img_id + '.jpg' for img_id in image_dict["clinic_close"]]
    clinical_overview_images = [img_id + '.jpg' for img_id in image_dict["clinic_overview"]]

    for img in os.listdir(path_image):
        if (img in clinical_overview_images or img in clinical_close_images) and img not in os.listdir(dst_path) and is_image_valid(path_image + img):
            shutil.copy(path_image + img, dst_path + img)