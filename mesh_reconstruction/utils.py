import numpy as np


def tile_images(images):
    rows = int(images.shape[0] ** 0.5)
    cols = int(images.shape[0] ** 0.5)
    image_size = images.shape[2]
    if images.ndim == 3:
        image = np.zeros((rows * image_size, cols * image_size))
    else:
        image = np.zeros((rows * image_size, cols * image_size, images.shape[1]))
        images = images.transpose((0, 2, 3, 1))
    for i in range(rows):
        for j in range(cols):
            image[i * image_size:(i + 1) * image_size, j * image_size:(j + 1) * image_size] = images[i * cols + j]
    return image
