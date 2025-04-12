from requests import get
from io import BytesIO
from PIL import Image
import pandas as pd
import os


def fitzpatrick_training_data():
    dst_train_data = "TRAIN_DATA/PAN/"
    dst_test_data = "TEST_DATA/PAN/"

    headers = {"User-Agent": "myagent"}

    df = pd.read_csv("ALL_DATA/fitzpatrick.csv")
    train_url_list = df["train_url"].dropna().tolist()

    url_count = 0

    for url in train_url_list:
        try:
            name = url.split("/")[-1]
            if name not in os.listdir(dst_train_data) and name not in os.listdir(dst_test_data):
                image_resp = get(url, headers=headers).content
                img = Image.open(BytesIO(image_resp))
                name = url.split("/")[-1]
                img.save(dst_train_data + name)
                url_count += 1
                print(f"Immagine {name} salvata, contatore={url_count}!")
        except Exception as e:
            print(f"Errore: {e}")
            pass



def fitzpatrick_test_data():
    dst_train_data = "TRAIN_DATA/PAN/"
    dst_test_data = "TEST_DATA/PAN/"

    headers = {"User-Agent": "myagent"}

    df = pd.read_csv("ALL_DATA/fitzpatrick.csv")
    test_url_list = df["test_url"].dropna().tolist()

    url_count = 0

    for url in test_url_list:
        try:
            name = url.split("/")[-1]
            if name not in os.listdir(dst_train_data) and name not in os.listdir(dst_test_data):
                image_resp = get(url, headers=headers).content
                img = Image.open(BytesIO(image_resp))
                name = url.split("/")[-1]
                img.save(dst_test_data + name)
                url_count += 1
                print(f"Immagine {name} salvata, contatore={url_count}!")
        except Exception as e:
            print(f"Errore: {e}")
            pass