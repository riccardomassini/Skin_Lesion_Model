from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
import random
import os


class ProbabilisticDegradationDataset(Dataset):
    def __init__(self, root_dir, target_class_mapping,
                 degradation_probability=0.2,
                 low_quality_class_name='LOW_QUALITY',
                 transform=None, degrader=None):

        self.root_dir = root_dir
        self.target_class_mapping = target_class_mapping
        self.degradation_probability = degradation_probability
        self.low_quality_label = target_class_mapping[low_quality_class_name]
        self.low_quality_class_name = low_quality_class_name
        self.transform = transform
        self.degrader = degrader
        self.image_paths = []
        self.image_original_labels = []

        base_class_names_in_dir = [name for name in target_class_mapping.keys() if name != low_quality_class_name]

        for class_name in base_class_names_in_dir:
            class_original_label = target_class_mapping[class_name]
            class_dir = os.path.join(self.root_dir, class_name)

            for fname in os.listdir(class_dir):
                self.image_paths.append(os.path.join(class_dir, fname))
                self.image_original_labels.append(class_original_label)

        self.num_original_images = len(self.image_paths)
        self.classes = list(target_class_mapping.keys())
        self.num_total_classes = len(self.classes)

        # print(f"Training Dataset: Initialized for {self.num_total_classes} target classes: {self.classes}")
        # print(f"Training Dataset: Expecting folders for {base_class_names_in_dir} in {root_dir}")
        # print(f"Training Dataset: Found {self.num_original_images} images.")
        # print(f"Training Dataset: Each image has a {self.degradation_probability*100:.1f}% chance of being degraded to label '{self.low_quality_class_name}' ({self.low_quality_label}).")


    def __len__(self):
        return self.num_original_images

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        original_label = self.image_original_labels[idx]
        degrade_this_image = random.random() < self.degradation_probability

        image_pil = Image.open(img_path).convert("RGB")

        if degrade_this_image and self.degrader:
            image_np = np.array(image_pil)
            degraded_np = self.degrader(image_np)
            image = Image.fromarray(degraded_np)
            label = self.low_quality_label
        else:
            image = image_pil
            label = original_label

        if self.transform:
            image = self.transform(image)

        return image, label


class DegradedImageTransform:
    def __init__(self):
        self.transforms = {
            'motion_blur': self.motion_blur,
            'gaussian_blur': self.gaussian_blur,
            'brightness': self.brightness,
            'noisiness': self.noisiness,
            'chromatic_aberration': self.chromatic_aberration,
            'pixelation': self.pixelation
        }
        # print("DegradedImageTransform initialized with effects:", list(self.transforms.keys()))

    def __call__(self, image_np: np.ndarray) -> np.ndarray:
        if not isinstance(image_np, np.ndarray):
            if isinstance(image_np, Image.Image):
                 image_np = np.array(image_np.convert("RGB"))

        if image_np.dtype != np.uint8:
            if image_np.dtype in [np.float32, np.float64] and image_np.min() >= 0 and image_np.max() <= 1:
                 image_np = (image_np * 255).astype(np.uint8)
            else:
                 image_np = np.clip(image_np, 0, 255).astype(np.uint8)

        image_np = self._ensure_rgb(image_np)

        transform_name = random.choice(list(self.transforms.keys()))
        try:
            degraded_image = self.transforms[transform_name](image_np.copy())
            degraded_image = self._ensure_rgb(degraded_image)
            if degraded_image.dtype != np.uint8:
                 degraded_image = np.clip(degraded_image, 0, 255).astype(np.uint8)
            return degraded_image
        except Exception as e:
            print(f"Error during degradation '{transform_name}': {e}. Returning original image.")
            return self._ensure_rgb(image_np).astype(np.uint8)


    def _ensure_rgb(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif len(image.shape) == 3 and image.shape[-1] == 1:
             image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[-1] != 3:
             print(f"Warning: Unexpected number of channels {image.shape[-1]}. Attempting to select first 3.")
             image = image[..., :3]

        if image.dtype != np.uint8:
             if image.dtype in [np.float32, np.float64]:
                 print(f"Warning: Image became {image.dtype} in _ensure_rgb. Clipping and converting to uint8.")
                 image = np.clip(image, 0, 255)
             image = image.astype(np.uint8)

        return image

    def motion_blur(self, image: np.ndarray) -> np.ndarray:
        kernel_size = random.randint(20, 60)
        kernel_motion_blur = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        center = (kernel_size - 1) // 2
        kernel_motion_blur[center, :] = np.ones(kernel_size, dtype=np.float32)
        kernel_motion_blur = kernel_motion_blur / kernel_size
        return cv2.filter2D(image, -1, kernel_motion_blur)

    def gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        blur_amount = random.randint(8, 15) * 2 + 1
        return cv2.GaussianBlur(image, (blur_amount, blur_amount), 0)

    def brightness(self, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        delta_v = random.uniform(0.2, 0.3) * 100
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + delta_v, 0, 255)
        bright_image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return bright_image

    def noisiness(self, image: np.ndarray) -> np.ndarray:
        noise_type = random.choice(['gaussian', 'salt_and_pepper'])
        h, w, c = image.shape

        image_float = image.astype(np.float32) / 255.0

        if noise_type == 'gaussian':
            std_dev = random.uniform(0.05, 0.20)
            noise = np.random.normal(0, std_dev, image_float.shape).astype(np.float32)
            noisy_image = image_float + noise

        elif noise_type == 'salt_and_pepper':
            amount = random.uniform(0.05, 0.20)
            s_vs_p = 0.5
            noisy_image = np.copy(image_float)
            num_salt = np.ceil(amount * image.size * s_vs_p / c)
            coords = [np.random.randint(0, i, int(num_salt)) for i in (h, w)]
            noisy_image[coords[0], coords[1], :] = 1.0
            num_pepper = np.ceil(amount * image.size * (1. - s_vs_p) / c)
            coords = [np.random.randint(0, i, int(num_pepper)) for i in (h, w)]
            noisy_image[coords[0], coords[1], :] = 0.0

        noisy_image = np.clip(noisy_image, 0.0, 1.0)
        return (noisy_image * 255).astype(np.uint8)

    def chromatic_aberration(self, image: np.ndarray) -> np.ndarray:
        offset = random.randint(5, 20)
        aberrated_image = image.copy()
        axis = random.choice([0, 1])
        channels_to_shift = random.sample([0, 1, 2], 2)

        aberrated_image[:, :, channels_to_shift[0]] = np.roll(image[:, :, channels_to_shift[0]], offset, axis=axis)
        aberrated_image[:, :, channels_to_shift[1]] = np.roll(image[:, :, channels_to_shift[1]], -offset, axis=axis)
        return aberrated_image

    def pixelation(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        scale_factor = random.uniform(0.05, 0.15)

        new_height = max(1, int(height * scale_factor))
        new_width = max(1, int(width * scale_factor))

        try:
            image_small = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            return cv2.resize(image_small, (width, height), interpolation=cv2.INTER_NEAREST)
        except Exception as e:
            print(f'Error during pixelation (scale={scale_factor:.2f}): {e}. Returning original.')
            return image