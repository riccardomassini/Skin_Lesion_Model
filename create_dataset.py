from manage_dataset.bcn_data import bcn_training_data, bcn_test_data
from manage_dataset.ham_data import ham_training_data, ham_test_data
from manage_dataset.derm7pt_data import derm7pt_training_data, derm7pt_test_data
from manage_dataset.padufes_data import padufes_training_data
from manage_dataset.hospital_buenos_aires_data import hospital_test_data
from manage_dataset.imagenet_data import imagenet_training_data, imagenet_test_data
from manage_dataset.fitzpatrick import fitzpatrick_training_data, fitzpatrick_test_data
from manage_dataset.degraded_data import degraded_test_data
import os

def is_folder_empty(folder):
    return not os.listdir(folder)

classes = ["DERM", "DERM_CLINIC", "LOW_QUALITY", "OTHER", "PAN"]

for el in classes:
    os.makedirs(f"TEST_DATA/{el}", exist_ok=True)

for el in classes[:2] + classes[3:]:
    os.makedirs(f"TRAIN_DATA/{el}", exist_ok=True)

if is_folder_empty("TRAIN_DATA/DERM"):
    print("ADDING BCN DATA ...")
    bcn_training_data()
    bcn_test_data()

    print("ADDING HAM DATA ...")
    ham_training_data()
    ham_test_data()

if is_folder_empty("TRAIN_DATA/DERM") or is_folder_empty("TRAIN_DATA/DERM_CLINIC"):
    print("ADDING DERM7PT DATA ...")
    derm7pt_training_data()
    derm7pt_test_data()

if is_folder_empty("TRAIN_DATA/DERM_CLINIC"):
    print("ADDING PAD-UFES DATA ...")
    padufes_training_data()

    print("ADDING HOSPITAL DE BUENOS AIRES DATA ...")
    hospital_test_data()

if is_folder_empty("TRAIN_DATA/OTHER"):
    print("ADDING IMAGENET DATA ...")
    imagenet_training_data()
    imagenet_test_data()

if is_folder_empty("TRAIN_DATA/PAN"):
    print("ADDING FITZPATRICK DATA ...")
    fitzpatrick_training_data()
    fitzpatrick_test_data()

if is_folder_empty("TEST_DATA/LOW_QUALITY"):
    print("ADDING DEGRADED DATA ...")
    degraded_test_data()