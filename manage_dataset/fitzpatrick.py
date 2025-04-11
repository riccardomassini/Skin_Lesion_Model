from collections import defaultdict
from requests import get
from io import BytesIO
from PIL import Image
import pandas as pd
import random
import os

def fitzpatrick_training_data():
    dst_train_data = "TRAIN_DATA/PAN/"
    dst_test_data = "TEST_DATA/PAN/"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    df = pd.read_csv("ALL_DATA/fitzpatrick17k.csv")
    classes = list(set(df["label"].tolist()))

    class_counts = defaultdict(int)
    url_dict = defaultdict(list)
    max_for_class = 500
    n_for_class = 20

    for _, row in df.iterrows():
        class_counts[row["label"]] += 1
        url_dict[row["label"]].append(row["url"])

    for key in url_dict.keys():
        random.shuffle(url_dict[key])

    url_selected = defaultdict(list)

    for key, value in url_dict.items():
        url_selected[key] = value[:max_for_class]

    for key, url_list in url_selected.items():
        url_count = 0
        for url in url_list:
            try:
                name = url.split("/")[-1]
                if name not in os.listdir(dst_train_data) and name not in os.listdir(dst_test_data):
                    image_resp = get(url, headers=headers).content
                    img = Image.open(BytesIO(image_resp))
                    name = url.split("/")[-1]
                    img.save(dst_train_data + name)
                    url_count += 1

                    print(f"Immagine {name} salvata per la classe {key}, contatore={url_count}!")

                    if url_count >= n_for_class:
                        break

                else:
                    print("Già utilizzata...")

            except Exception:
                print("Errore...")
                pass



def fitzpatrick_test_data():
    dst_train_data = "TRAIN_DATA/PAN/"
    dst_test_data = "TEST_DATA/PAN/"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    df = pd.read_csv("ALL_DATA/fitzpatrick17k.csv")
    classes = list(set(df["label"].tolist()))

    class_counts = defaultdict(int)
    url_dict = defaultdict(list)
    max_for_class = 100
    n_for_class = 4

    for _, row in df.iterrows():
        class_counts[row["label"]] += 1
        url_dict[row["label"]].append(row["url"])

    for key in url_dict.keys():
        random.shuffle(url_dict[key])

    url_selected = defaultdict(list)

    for key, value in url_dict.items():
        url_selected[key] = value[:max_for_class]

    for key, url_list in url_selected.items():
        url_count = 0
        for url in url_list:
            try:
                name = url.split("/")[-1]
                if name not in os.listdir(dst_train_data) and name not in os.listdir(dst_test_data):
                    image_resp = get(url, headers=headers).content
                    img = Image.open(BytesIO(image_resp))
                    name = url.split("/")[-1]
                    img.save(dst_test_data + name)
                    url_count += 1

                    print(f"Immagine {name} salvata per la classe {key}, contatore={url_count}!")

                    if url_count >= n_for_class:
                        break

                else:
                    print("Già utilizzata...")

            except Exception:
                print("Errore...")
                pass