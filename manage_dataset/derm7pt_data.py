from manage_dataset.init_derm7pt.dataset import Derm7PtDataset, Derm7PtDatasetGroupInfrequent
from manage_dataset.validate_image import is_image_valid
import os
import shutil
import pandas as pd


def manage_derm7pt():
    dir_release = "ALL_DATA/release_v0/"
    dir_meta = dir_release + "meta/"
    dir_images = dir_release + "images/"
    meta_df = pd.read_csv(dir_meta + "meta.csv")
    train_indexes = list(pd.read_csv(dir_meta + "train_indexes.csv")["indexes"])
    valid_indexes = list(pd.read_csv(dir_meta + "valid_indexes.csv")["indexes"])
    test_indexes = list(pd.read_csv(dir_meta + "test_indexes.csv")["indexes"])

    derm_data = Derm7PtDataset(dir_images=dir_images,
                               metadata_df=meta_df.copy(),  # Copy as is modified.
                               train_indexes=train_indexes, valid_indexes=valid_indexes,
                               test_indexes=test_indexes)

    derm_data_group = Derm7PtDatasetGroupInfrequent(dir_images=dir_images,
                                                    metadata_df=meta_df.copy(),  # Copy as is modified.
                                                    train_indexes=train_indexes,
                                                    valid_indexes=valid_indexes,
                                                    test_indexes=test_indexes)

    # TEST DERM DATA
    test_derm_paths = derm_data.get_img_paths(data_type='test', img_type='derm')

    # TRAIN DERM DATA
    train_derm_paths = derm_data.get_img_paths(data_type='train', img_type='derm')

    # TEST CLINIC DATA
    test_clinic_paths = derm_data.get_img_paths(data_type='test', img_type='clinic')

    # TRAIN CLINIC DATA
    train_clinic_paths = derm_data_group.get_img_paths(data_type='train', img_type='clinic')

    return train_derm_paths, train_clinic_paths, test_derm_paths, test_clinic_paths


def derm7pt_training_data():
    my_train_data_clinic = "TRAIN_DATA/DERM_CLINIC/"
    my_train_data_derm = "TRAIN_DATA/DERM/"

    train_derm_paths, train_clinic_paths, _, _ = manage_derm7pt()

    for train in train_derm_paths:
        name = train.split("/")[-1]
        if name not in os.listdir(my_train_data_derm) and is_image_valid(train):
            shutil.copy(train, my_train_data_derm + name)

    for train in train_clinic_paths:
        name = train.split("/")[-1]
        if name not in os.listdir(my_train_data_clinic) and is_image_valid(train):
            shutil.copy(train, my_train_data_clinic + name)


def derm7pt_test_data():
    my_test_data_clinic = "TEST_DATA/DERM_CLINIC/"
    my_test_data_derm = "TEST_DATA/DERM/"

    _, _, test_derm_paths, test_clinic_paths = manage_derm7pt()

    for test in test_derm_paths:
        name = test.split("/")[-1]
        if name not in os.listdir(my_test_data_derm) and is_image_valid(test):
            shutil.copy(test, my_test_data_derm + name)

    for test in test_clinic_paths:
        name = test.split("/")[-1]
        if name not in os.listdir(my_test_data_clinic) and is_image_valid(test):
            shutil.copy(test, my_test_data_clinic + name)