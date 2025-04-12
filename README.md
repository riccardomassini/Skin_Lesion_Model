# 🧠 Deep Learning Classifier for Medical Image Categories

## 📌 Project Overview

This project focuses on training and testing a deep learning model to classify images into **five categories** related to dermatological and non-medical contexts. The classifier is designed to distinguish between different image types based on source and content.

### 🧑‍💻 Requirements

To run this project, you'll need the following software and libraries installed:

- **Python**: A modern version of Python (preferably 3.7 or higher).
- **PyTorch**: A deep learning framework used for model training and inference.
- **Torchvision**: A library that provides pre-trained models and datasets.

You can install Python from the official website ([Python Installation](https://www.python.org/downloads/)) and PyTorch along with Torchvision by following the official guide: [PyTorch/Torchvision Installation](https://pytorch.org/get-started/locally/)

It is recommended to create a **virtual environment** to avoid conflicts with other packages and to manage dependencies more easily, especially due to the number of libraries that need to be installed. You can create a virtual environment using the following commands:

For **Windows**:
```bash
python -m venv your_venv  
.\your_venv\Scripts\activate
```

For **Linux/macOS**:
```bash
python3 -m venv your_venv  
source your_venv/bin/activate
```

Additionally, make sure to install the required Python packages by running:
```bash
pip install -r requirements.txt
```

This will install all the additional libraries needed to run the project.

Before starting training or testing, make sure to **generate and organize the dataset** by following the instructions in the [Dataset Setup](#-dataset-setup) section below.  
Once all the required images are downloaded and placed in their respective folders, run the following script to prepare the final dataset structure:
```bash
python create_dataset.py
```

### 🔍 Classification Categories

| Category       | Description                                                                 |
|----------------|-----------------------------------------------------------------------------|
| `DERM`         | Dermatoscopic images of skin lesions                                        |
| `DERM_CLINIC`  | Clinical (smartphone) images of skin lesions                               |
| `PAN`          | Panoramic or body-part images that may contain multiple lesions             |
| `OTHER`        | Non-medical images                                                          |
| `LOW_QUALITY`  | Any type of image with poor visual quality (e.g., blurry, noisy, low-res)   |

---

## 📂 Dataset Setup

Follow the instructions below to download and prepare the datasets required for each category.

### 📁 `DERM` – Dermatoscopic Images

- **Links**: 
  - [ISIC 2018 Challenge](https://challenge.isic-archive.com/data/#2018) 
  - [ISIC 2019 Challenge](https://challenge.isic-archive.com/data/#2019)
- **Instructions**:
  1. **ISIC 2018 Dataset**:
     - Download Training Data for Task 3 (10015 images) and Place all the images into `ALL_DATA/ISIC2018_Task3_Training_Input/` directory.
     - Download Test Data for Task 3 (1512 images) and place all the images into `ALL_DATA/ISIC2018_Task3_Test_Input/` directory.

  2. **ISIC 2019 Dataset**:
     - Download Test Data for Task 1 (8238 images) and place all the images into `ALL_DATA/ISIC_2019_Training_Input/` directory.
     - Download Training Data for Task 1 (25331 images) and place all the images into `ALL_DATA/ISIC_2019_Test_Input/` directory.

### 📁 `DERM_CLINIC` – Clinical Smartphone Images

- **Links**:
  - [PAD-UFES-20](https://data.mendeley.com/datasets/zr7vgbcyr2/1) 
  - [Hospital Italiano de Buenos Aires - Skin Lesions Images (2019-2022)](https://api.isic-archive.com/doi/hospital-italiano-de-buenos-aires-skin-lesions-images-2019-2022/)
- **Instructions**:
  1. **PAD-UFES-20 Dataset**:
     - Download the zip file and place the images from the `imgs_part_1/` folder into `ALL_DATA/img_part_1/`. Then, do the same for the `imgs_part_2/` and `imgs_part_3/` folders, placing the respective images into `ALL_DATA/img_part_2/` and `ALL_DATA/img_part_3/`.

  2. **Hospital Italiano de Buenos Aires Dataset**:
     - Download the zip file with the description `The complete bundle of all images, metadata, and supplemental files related to this dataset` and place `images` folder and `metadata.cvs` into `ALL_DATA/hospital_buenos_aires/` directory.

### 📁 `DERM` and `DERM_CLINIC` – Dermatoscopic and Clinical Image Dataset

- **Link**:
  - [Derm7pt](https://derm.cs.sfu.ca/Download.html) 
- **Description**:
  This dataset contains images of skin lesions, divided into two main categories:
  1. **Dermatoscopic** (DERM): High-resolution images captured using dermoscopy, used for detailed analysis of skin lesion characteristics.
  2. **Clinical** (DERM_CLINIC): Clinical images taken using smartphones or conventional cameras, providing a general view of the skin lesions.

- **Instructions**:
  1. **Derm7pt Dataset**:
    - Download the zip file and place all files and folders into `ALL_DATA/release_v0/` directory.
    - Note: To download the dataset, you need to register on the website and wait for an email with your login credentials.

### 📁 `PAN` – Panoramic / Body-Part Images

- **Link**:
  - [Hospital Varese](https://drive.google.com/drive/folders/1SdL61x75OpEc-z0L_o7irB0uX3o2fTKV)
- **Instructions**:
  1. **Hospital Varese Dataset**:
     - Download the `train` and `test` folders and place them into `ALL_DATA/hospital_varese/` directory.

The remaining data will be automatically downloaded from the internet during the process, so there's no need to manually add them to the folder.

### 📁 `OTHER` – Non-Medical Images

- **Link**:
  - [ImageNet 1000 (mini)](https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000)
- **Instructions**:
  1. **ImageNet 1000 (mini) Dataset**:
     - Download data and place the `train` and `val` directories into `ALL_DATA/imagenet-mini/` directory.

### 📁 `LOW_QUALITY` – Low-Quality Images

- No manual download required.
- These images will be automatically generated by applying transformations (e.g., blur, noise) to the original datasets.

---

## 🏋️‍♂️ Training and Testing

Once the dataset is prepared and organized, you can start training and testing the deep learning model by running the following script:

```bash
python launch_training_test.py
```

### 📦 Configuration: Batch Size and Workers

You can adjust the `batch_size` and `num_workers` values as follows:

- `batch_size`: Number of samples processed in one forward/backward pass. A larger batch size can speed up training but requires more memory.
- `num_workers`: Number of subprocesses used to load data in parallel. Increasing this can speed up data loading, but it is limited by the number of CPU cores available on your machine.

The script will prompt you to enter these values interactively. The code will validate that both values are positive integers greater than 0. Here's how you can configure them:

```python
def get_positive_int(prompt):
    while True:
        try:
            value = int(input(prompt))
            if value > 0:
                return value
            else:
                print("Please enter a number greater than 0.")
        except ValueError:
            print("That's not a valid number. Please enter a valid integer.")
            
batch = get_positive_int("Choose batch_size: ")
workers = get_positive_int("Choose num_workers: ")
```

When running the script, you can customize the batch size and the number of workers for data loading to manage your system's resources. These parameters will be used to create the `train_loader` and `test_loader`:

```python
train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True, num_workers=workers)
test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=True, num_workers=workers)
```

### 🧑‍💻 Model Selection: Choose Your Model

The project supports various deep learning models, each with different levels of complexity. You can select the model interactively via Python, where the script will prompt you to choose the model based on your preferences.

Here's how to select your model:

#### Models Available:
1. **ResNet-18** (Lightweight)
   - A smaller model that works well for many tasks, offering a good balance of performance and efficiency.

2. **ResNet-50** (Medium Complexity)
   - A more powerful model compared to ResNet-18, but still relatively lightweight. It performs better on more complex tasks.

3. **Vision Transformers (ViT)** (Heavyweight)
   - A transformer-based model that has gained popularity for its performance in vision tasks. It requires significantly more computational resources compared to ResNets.

4. **Swin Transformer** (Very Heavyweight)
   - A more advanced transformer model that achieves state-of-the-art performance but is computationally expensive. Suitable for larger datasets or more complex tasks.

#### Model Selection Code

In the script, the user will be prompted to choose a model by entering a number. The model selected will be mapped to its corresponding configuration.

Here's the code to implement this:

```python
models_dict = {
    "resnet18": "resnet18",
    "resnet50": "resnet50",
    "vit": "vit_base_patch16_224",
    "swin": "swin_base_patch4_window7_224"
}

chose_dict = {
    "1": list(models_dict.keys())[0],
    "2": list(models_dict.keys())[1],
    "3": list(models_dict.keys())[2],
    "4": list(models_dict.keys())[3] 
}

chose = ""

while chose not in chose_dict:
    print("Choose a model:\n1 - ResNet-18\n2 - ResNet-50\n3 - Vision Transformer (ViT)\n4 - Swin Transformer")
    chose = input("Your choice: ")

selected_model = chose_dict[chose]
model_name = models_dict[selected_model]
```

### 📊 Memory Usage Estimation

As a rough guide, the memory consumption on the GPU will vary depending on the model and the chosen batch size. Here are the approximate memory usage estimates for different models with a batch size of 64 and 4 workers:

- **ResNet-18**: 
  - Using a batch size of 64 and 4 workers, this model will typically require around **2.5 - 3 GB** of GPU RAM.

- **ResNet-50**:
  - With the same batch size and number of workers, ResNet-50 will need around **4.5 - 5 GB** of GPU RAM.

- **Vision Transformer (ViT)**:
  - The ViT model is more memory-intensive and will generally require **8 - 10 GB** of GPU RAM with a batch size of 64 and 4 workers.

- **Swin Transformer**:
  - Swin Transformer is one of the most computationally expensive models and may require **10 - 12 GB** or more of GPU RAM depending on the specific configuration.

**Important Note for CPU Usage**:  
Running these models on the **CPU** is not recommended, especially with large models like ViT or Swin, as it could be very slow and consume excessive memory. CPUs are not optimized for the parallelized operations needed for deep learning tasks, which can lead to significantly longer training times and memory bottlenecks.

For best performance, it is recommended to use a **batch size of at least 32-64** and **2-4 workers** to ensure optimal use of GPU resources without overloading memory. If you run into memory issues, you can reduce the batch size or use a smaller model.

---
