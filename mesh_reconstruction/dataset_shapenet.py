import cStringIO
import numpy as np
import os

import chainer
import imageio
import neural_renderer
import tqdm
import pickle


def decode_image(image):
    return imageio.imread(cStringIO.StringIO(image)).transpose((2, 0, 1)).astype('float32') / 255.


def process_images(images, threshold=0.1):
    # for RGB, rescale from [0, 1] to [-1, 1] and change background to gray
    mask = images[:, -1]
    images_rgb = 2 * images[:, :3] - 1
    images[:, :3] = mask[:, None, :, :] * images_rgb

    # binarize silhouettes
    images[:, 3][images[:, 3] >= threshold] = 1
    images[:, 3][images[:, 3] <= threshold] = 0

    return images


class ShapeNet(object):
    def __init__(self, directory, class_ids, set_name, num_views=20, single_view_training=False, device=None):
        self.name = 'shapenet'
        self.image_size = 224
        self.num_views_max = 20

        self.directory = directory
        self.class_ids = class_ids
        self.set_name = set_name
        self.num_views = num_views
        self.single_view_training = single_view_training
        self.device = device

        self.images = {}
        self.viewpoints = {}
        self.voxels = {}
        self.num_data = {}
        self.object_ids = {}

        loop = tqdm.tqdm(class_ids)
        for class_id in loop:
            loop.set_description('Loading dataset')
            filename = os.path.join(directory, '%s_%s.pkl' % (class_id, set_name))
            data = pickle.load(open(filename))
            self.images.update(data['images'])
            self.viewpoints.update(data['viewpoints'])
            self.voxels.update(data['voxels'])
            self.num_data[class_id] = len(data['voxels'])
            self.object_ids[class_id] = [key[9:] for key in data['voxels'].keys()]

        self.skip_ids = [
            '187f32df6f393d18490ad276cd2af3a4',  # invalid image
            '391fa4da294c70d0a4e97ce1d10a5ae6',  # invalid image
            '50cdaa9e33fc853ecb2a965e75be701c',  # failed to load
        ]

    def get_single_data_train(self, class_id, object_id, render_id, flip, color_shuffle):
        key = str('%s_%s_%d' % (class_id, object_id, render_id))
        image = decode_image(self.images[key])
        viewpoint = self.viewpoints[key].copy()

        if flip:
            image = image[:, :, ::-1]
            viewpoint[-1] *= -1
        image[:3] = image[:3][color_shuffle]

        return image, viewpoint

    def get_random_batch(self, batch_size):
        labels = np.zeros(batch_size, 'int32')
        images_a = np.zeros((batch_size, 4, self.image_size, self.image_size), 'float32')
        viewpoints_a = np.zeros((batch_size, 3), 'float32')
        if not self.single_view_training:
            images_b = np.zeros((batch_size, 4, self.image_size, self.image_size), 'float32')
            viewpoints_b = np.zeros((batch_size, 3), 'float32')

        for i in range(batch_size):
            class_id = np.random.choice(self.class_ids)
            object_id = np.random.choice(self.object_ids[class_id])
            while object_id in self.skip_ids:
                object_id = np.random.choice(self.object_ids[class_id])
            if self.num_views != 1:
                render_id_a, render_id_b = np.random.permutation(self.num_views)[:2]
            else:
                render_id_a = 0
            flip = np.random.rand() < 0.5
            color_shuffle = np.random.permutation(3)

            labels[i] = self.class_ids.index(class_id)
            images_a[i], viewpoints_a[i] = self.get_single_data_train(
                class_id, object_id, render_id_a, flip, color_shuffle)
            if not self.single_view_training:
                images_b[i], viewpoints_b[i] = self.get_single_data_train(
                    class_id, object_id, render_id_b, flip, color_shuffle)

        if self.device is not None:
            images_a = chainer.cuda.to_gpu(images_a, self.device)
            viewpoints_a = chainer.cuda.to_gpu(viewpoints_a, self.device)
        images_a = process_images(images_a)
        viewpoints_a = neural_renderer.get_points_from_angles(
            viewpoints_a[:, 0], viewpoints_a[:, 1], viewpoints_a[:, 2] + 90).data
        if not self.single_view_training:
            if self.device is not None:
                images_b = chainer.cuda.to_gpu(images_b, self.device)
                viewpoints_b = chainer.cuda.to_gpu(viewpoints_b, self.device)
            images_b = process_images(images_b)
            viewpoints_b = neural_renderer.get_points_from_angles(
                viewpoints_b[:, 0], viewpoints_b[:, 1], viewpoints_b[:, 2] + 90).data

            return images_a, images_b, viewpoints_a, viewpoints_b, labels
        else:
            return images_a, None, viewpoints_a, None, labels

    def get_all_batches_for_evaluation(self, batch_size, class_id, num_views=None):
        # returns 20 views per object
        if num_views is None:
            num_views = self.num_views_max
        num_objects = self.num_data[class_id]
        object_ids = np.repeat(self.object_ids[class_id], num_views)
        render_ids = np.tile(np.arange(num_views), num_objects)
        for batch_num in range((len(object_ids) - 1) / batch_size + 1):
            batch_size2 = min(len(object_ids) - batch_num * batch_size, batch_size)
            images_in = np.zeros((batch_size2, 4, self.image_size, self.image_size), 'float32')
            voxels = np.zeros((batch_size2, 32, 32, 32), 'float32')

            for i in range(batch_size2):
                object_id = object_ids[batch_num * batch_size + i]
                render_id = render_ids[batch_num * batch_size + i]
                key_image = str('%s_%s_%d' % (class_id, object_id, render_id))
                key_voxel = str('%s_%s' % (class_id, object_id))

                images_in[i] = decode_image(self.images[key_image])
                voxels[i] = self.voxels[key_voxel].astype('float32')

            if self.device is not None:
                images_in = chainer.cuda.to_gpu(images_in, self.device)
                voxels = chainer.cuda.to_gpu(voxels, self.device)
            images_in = process_images(images_in)

            yield images_in, voxels
