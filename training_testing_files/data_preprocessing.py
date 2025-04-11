from torchvision import transforms
import random
import torch
import timm

class RandomResize:
    def __init__(self, resize_min=224, resize_max=280):
        self.resize_min = resize_min
        self.resize_max = resize_max

    def __call__(self, img):
        resize_size = random.randint(self.resize_min, self.resize_max)
        resize_transforms = transforms.Resize(resize_size)
        img = resize_transforms(img)
        return img
    
def get_preprocess(model_name):
    model = timm.create_model(model_name, pretrained=True, num_classes=5)

    '''train_preprocess = transforms.Compose([
        RandomResize(250, 350),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))], p=0.5),
        transforms.RandomApply(transforms=[transforms.RandomRotation(degrees=(-60, 60))], p=0.5),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])'''

    train_preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return model, train_preprocess, test_preprocess