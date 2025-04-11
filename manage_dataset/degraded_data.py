from manage_dataset.degrade import DegradedImageTransform
from pathlib import Path
from PIL import Image
import numpy as np
import random


def degraded_test_data():
    num_images_per_class = 225
    src_path = "TEST_DATA/"
    source_dir = Path(src_path)
    low_quality_dir = source_dir / 'LOW_QUALITY'
    low_quality_dir.mkdir(parents=True, exist_ok=True)

    classes = ['DERM', 'DERM_CLINIC', 'PAN', 'OTHER']

    for class_name in classes:
        class_path = source_dir / class_name
        images = list(class_path.glob("*"))
        selected_images = random.sample(images, min(num_images_per_class, len(images)))

        for img in selected_images:
            img_pil = Image.open(img)
            np_image = np.array(img_pil)
            degraded_img = DegradedImageTransform()(np_image)

            if degraded_img.dtype != np.uint8:
                degraded_img = np.clip(degraded_img, 0, 255).astype(np.uint8)

            degraded_img = Image.fromarray(degraded_img)
            degraded_img.save(low_quality_dir / img.name)