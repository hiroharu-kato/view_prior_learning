import chainer
import chainer.functions as cf
import math
import numpy as np
import imageio


def to_gpu(data, device=None):
    if isinstance(data, tuple) or isinstance(data, list):
        return [chainer.cuda.to_gpu(d, device) for d in data]
    else:
        return chainer.cuda.to_gpu(data)


def imread(filename):
    return imageio.imread(filename).astype('float32') / 255.


def create_textures(num_faces, texture_size=16, flatten=False):
    if not flatten:
        tile_width = int((num_faces - 1.) ** 0.5) + 1
        tile_height = int((num_faces - 1.) / tile_width) + 1
    else:
        tile_width = 1
        tile_height = num_faces
    textures = np.ones((3, tile_height * texture_size, tile_width * texture_size), 'float32')

    vertices = np.zeros((num_faces, 3, 2), 'float32')  # [:, :, XY]
    face_nums = np.arange(num_faces)
    column = face_nums % tile_width
    row = face_nums / tile_width
    vertices[:, 0, 0] = column * texture_size
    vertices[:, 0, 1] = row * texture_size
    vertices[:, 1, 0] = column * texture_size
    vertices[:, 1, 1] = (row + 1) * texture_size - 1
    vertices[:, 2, 0] = (column + 1) * texture_size - 1
    vertices[:, 2, 1] = (row + 1) * texture_size - 1
    vertices = vertices.reshape((num_faces * 3, 2))
    faces = np.arange(num_faces * 3).reshape((num_faces, 3)).astype('int32')

    return vertices, faces, textures


def radians(degrees):
    pi = 3.14159265359
    return degrees / 180. * pi


def get_points_from_angles(distance, elevation, azimuth, degrees=True):
    if isinstance(distance, float) or isinstance(distance, int):
        if degrees:
            elevation = math.radians(elevation)
            azimuth = math.radians(azimuth)
        return (
            distance * math.cos(elevation) * math.sin(azimuth),
            distance * math.sin(elevation),
            -distance * math.cos(elevation) * math.cos(azimuth))
    else:
        if degrees:
            elevation = radians(elevation)
            azimuth = radians(azimuth)
        return cf.stack([
            distance * cf.cos(elevation) * cf.sin(azimuth),
            distance * cf.sin(elevation),
            -distance * cf.cos(elevation) * cf.cos(azimuth),
        ]).transpose()
