import cv2
import numpy as np


def resize_image(*, image: np.ndarray, size: int) -> np.ndarray:
    image_dims = image.shape[1::-1]
    aspect_ratio = image_dims[0] / image_dims[1]
    new_size = (int(round(size * aspect_ratio)), size)
    resized = cv2.resize(image, new_size, interpolation=cv2.INTER_LANCZOS4)
    return resized
